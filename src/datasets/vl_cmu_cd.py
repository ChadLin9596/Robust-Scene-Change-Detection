import copy
import glob
import os

import numpy as np
from PIL import Image

_file_path = os.path.split(os.path.realpath(__file__))[0]


class VL_CMU_CD:
    """
    each image should be <group_id>_1_<seq_id>_<angle>.png
    """

    def __init__(self, root, mode="train"):

        mode = mode.lower()
        assert mode in {"train", "test", "val"}
        if mode in {"train", "val"}:
            postfix = "train"
        else:
            postfix = "test"

        self.mode = mode

        # for simplicity, this class do not perform any security check
        # the assumptions are:
        # 1. [t0, t1, mask] folder must contain in root / train|test
        # 2. all files in [t0, t1, mask] use the exactly same file name.
        self.root = os.path.join(root, postfix)
        self.t0_path = os.path.join(self.root, "t0")
        self.t1_path = os.path.join(self.root, "t1")
        self.mask_path = os.path.join(self.root, "mask")

        filenames = glob.glob(os.path.join(self.t0_path, "*.png"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)

        if mode in {"train", "val"}:

            path = os.path.join(_file_path, f"indices/vl-cmu-cd.{mode}.index")
            with open(path, "r") as fd:
                indices = fd.read().splitlines()
            filenames = [i for i in filenames if i.split("_")[0] in indices]

        self._filenames = np.array(filenames)

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        # designed for torch.utils.data.DataLoader, supporting fetching
        # a data sample for a given key.
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

        t0_image = self.get_img_0(idx)
        t1_image = self.get_img_1(idx)
        mask_image = self.get_mask(idx)

        return t0_image, t1_image, mask_image

    def loc(self, key):
        if (
            not isinstance(key, slice)
            and np.shape(key) == ()
            and not (isinstance(key, np.ndarray) and key.dtype == np.bool_)
        ):
            key = np.unique(key)

        other = copy.copy(self)
        other._filenames = self._filenames[key]
        return other

    def get_img_0(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t0_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_img_1(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t1_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_mask(self, idx):
        filename = self._filenames[idx]
        mask_path = os.path.join(self.mask_path, filename)
        mask_image = Image.open(mask_path)
        mask_image = np.array(mask_image) / 255.0
        mask_image = mask_image > 0.0
        mask_image = mask_image.astype(np.float32)
        return mask_image

    @property
    def filenames(self):
        return self._filenames

    @property
    def group_ids(self):
        return np.array([i.split("_")[0] for i in self._filenames])

    @property
    def angles(self):
        x = [i.split(".")[0] for i in self._filenames]
        x = [int(i.split("_")[-1]) for i in x]
        return np.array(x)

    @property
    def seq_ids(self):
        x = [i.split(".")[0] for i in self._filenames]
        x = [i.split("_")[-2] for i in x]
        return np.array(x)

    @property
    def figsize(self):
        return np.array([512, 512])


class Diff_VL_CMU_CD:

    def __init__(self, root, mode="train", adjacent_distance=1):

        t0_dataset = VL_CMU_CD(root, mode=mode)
        t1_dataset = VL_CMU_CD(root, mode=mode)

        assert adjacent_distance != 0

        if mode in ["train", "val"]:
            stride = 4
        else:
            stride = 1

        if adjacent_distance > 0:
            I = slice(None, -adjacent_distance * stride)
            J = slice(adjacent_distance * stride, None)
        else:
            I = slice(-adjacent_distance * stride, None)
            J = slice(None, adjacent_distance * stride)

        t0_dataset = t0_dataset.loc(I)
        t1_dataset = t1_dataset.loc(J)

        I = t0_dataset.group_ids == t1_dataset.group_ids

        t0_dataset = t0_dataset.loc(I)
        t1_dataset = t1_dataset.loc(I)

        assert np.all(t0_dataset.group_ids == t1_dataset.group_ids)
        assert np.all(t0_dataset.angles == t1_dataset.angles)

        self.t0_dataset = t0_dataset
        self.t1_dataset = t1_dataset

        self._filenames = None

    def __len__(self):
        return len(self.t0_dataset)

    def __getitem__(self, idx):
        t0_image = self.t0_dataset.get_img_0(idx)
        t1_image = self.t1_dataset.get_img_1(idx)
        mask_image = self.t0_dataset.get_mask(idx)

        return t0_image, t1_image, mask_image

    def loc(self, key):
        other = copy.copy(self)
        other.t0_dataset = self.t0_dataset.loc(key)
        other.t1_dataset = self.t1_dataset.loc(key)
        return other

    def get_img_0(self, idx):
        return self.t0_dataset.get_img_0(idx)

    def get_img_1(self, idx):
        return self.t1_dataset.get_img_1(idx)

    def get_mask(self, idx):
        return self.t0_dataset.get_mask(idx)

    @property
    def filenames(self):

        if self._filenames is not None:
            return self._filenames

        names = zip(
            self.t0_dataset.group_ids,
            self.t0_dataset.seq_ids,
            self.t1_dataset.seq_ids,
            self.t0_dataset.angles,
        )

        names = [f"{i}_1_{j}.{k}_{l}.png" for i, j, k, l in names]
        self._filenames = np.array(names)
        return self._filenames

    @property
    def group_ids(self):
        return self.t0_dataset.group_ids

    @property
    def angles(self):
        return self.t0_dataset.angles

    @property
    def seq_ids(self):

        ids = zip(
            self.t0_dataset.seq_ids,
            self.t1_dataset.seq_ids,
        )

        ids = [f"{i}.{j}" for i, j in ids]
        return np.array(ids)

    @property
    def figsize(self):
        return np.array([512, 512])