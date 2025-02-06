import os
import threading
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

# ignore warning showing from dino
warnings.filterwarnings("ignore")

from py_utils.src.utils import is_connect_to_network
from py_utils.src.utils_torch import (
    freeze_model,
    is_all_frozen,
    is_any_frozen,
    unfreeze_model,
)

# if your workspace does not have internet connection
# please download the model from github and save it to your local cache
# and change this path if necessary
_cache_root = "/home/{}/.cache/torch/hub/".format(os.environ["USER"])
_cache_root += "facebookresearch_dinov2_main"


def _get_dino(model="dinov2_vitg14", source="github", repo=None):
    """
    Wrapper for DINOv2

    dinov2_vits14 (small)
    dinov2_vitb14 (base)
    dinov2_vitl14 (large)
    dinov2_vitg14 (giant)
    """

    if source not in ["github", "local"]:
        raise ValueError('source must be either "github" or "local"')

    if source == "github" and is_connect_to_network():
        return torch.hub.load("facebookresearch/dinov2", model)

    if not is_connect_to_network():
        source = "local"

    if repo is None:
        repo = _cache_root

    return torch.hub.load(repo, model, source=source)


def get_dino(model, freeze=True, unfreeze_last_n_layers=0):

    dino_model = _get_dino(model)

    if freeze:
        freeze_model(dino_model)
        assert is_all_frozen(dino_model)
    else:
        assert ~is_any_frozen(dino_model)

    N = len(dino_model.blocks)
    n_layers = max(unfreeze_last_n_layers, 0)
    n_layers = N - n_layers

    for block in dino_model.blocks[n_layers:]:
        unfreeze_model(block)

    return dino_model


class ExtractDINO(nn.Module):

    def __init__(self, dino, layer=39):

        super(ExtractDINO, self).__init__()
        self.dino = dino
        self.num_features = dino.num_features
        self.patch_size = dino.patch_size
        self.embed_dim = dino.embed_dim
        self.layer = layer

        self.Q = None
        self.K = None
        self.V = None
        self.token = None

        self._cls_handle = None
        self._qkv_handle = None
        self._cls_hook_output = {}
        self._qkv_hook_output = {}

        self.frozen_mode = is_all_frozen(dino)

    def _del_handle(self):
        self._cls_handle.remove()
        self._qkv_handle.remove()

    def _set_handle(self):

        layer = self.layer

        # https://pytorch.org/docs/stable/generated/
        # torch.nn.modules.module.register_module_forward_hook.html

        # applied threading for pytorch data parallel
        # https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/7

        def cls_foo(module, inputs, outputs):
            self._cls_hook_output[threading.get_native_id()] = outputs

        def qkv_foo(module, inputs, outputs):
            self._qkv_hook_output[threading.get_native_id()] = outputs

        cls_handle = self.dino.blocks[layer].register_forward_hook(cls_foo)

        qkv_handle = self.dino.blocks[layer].attn.qkv.register_forward_hook(
            qkv_foo
        )

        self._cls_handle = cls_handle
        self._qkv_handle = qkv_handle

    def _forward(self, x):

        # [TODO] not sure why, but it works on both 1 and 2 GPUs env...
        # need to set register_forward_hook for each forward pass
        self._set_handle()

        assert len(x.shape) == 4
        batch, _, row, col = x.shape

        res = self.dino(x)

        thread_id = threading.get_native_id()

        qkv = self._qkv_hook_output[thread_id][:, 1:, ...]
        token = self._cls_hook_output[thread_id][:, 1:, ...]

        del self._qkv_hook_output[thread_id]
        del self._cls_hook_output[thread_id]

        # [TODO] not sure why, but it works on both 1 and 2 GPUs env...
        # need to delete register_forward_hook for each forward pass,
        self._del_handle()

        n = self.num_features

        # (batch, row * col / patch_size / patch_size, num_features)
        Q = qkv[..., :n]
        K = qkv[..., n : 2 * n]
        V = qkv[..., -n:]

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        V = F.normalize(V, dim=-1)
        token = F.normalize(token, dim=-1)
        res = F.normalize(res, dim=-1)

        new_shp = (batch, row // self.patch_size, col // self.patch_size, -1)
        self.Q = Q.reshape(new_shp)
        self.K = K.reshape(new_shp)
        self.V = V.reshape(new_shp)
        self.token = token.reshape(new_shp)

        return res

    def forward(self, x):

        if not self.frozen_mode:
            return self._forward(x)

        # ensure dino is in a state suitable for evaluation or inference
        # (dropout is turned off, batch normalization uses running statistics).
        if self.dino.training:
            self.dino.eval()

        # ensure that computational graph is not built,
        # saving memory and computational resources
        with torch.no_grad():
            x = self._forward(x)

        return x


class TwoDino(nn.Module):

    def __init__(self, dino_1, layer_1, dino_2, layer_2):

        super(TwoDino, self).__init__()

        self.dino_1 = ExtractDINO(dino_1, layer=layer_1)
        self.dino_2 = ExtractDINO(dino_2, layer=layer_2)

    def forward(self, img_1, img_2):

        x1 = self.dino_1(img_1)
        x2 = self.dino_2(img_2)

        return x1, x2

    @property
    def QKV_token_1(self):
        return self.dino_1.Q, self.dino_1.K, self.dino_1.V, self.dino_1.token

    @property
    def QKV_token_2(self):
        return self.dino_2.Q, self.dino_2.K, self.dino_2.V, self.dino_2.token

    @staticmethod
    def _get_facet(facet, QKVToken):
        Q, K, V, token = QKVToken

        if facet == "query":
            return Q
        if facet == "key":
            return K
        if facet == "value":
            return V
        if facet == "token":
            return token
        raise ValueError(f"{facet} do not support yet.")

    def get_facet_from_dino1(self, facet):
        return self._get_facet(facet, self.QKV_token_1)

    def get_facet_from_dino2(self, facet):
        return self._get_facet(facet, self.QKV_token_2)
