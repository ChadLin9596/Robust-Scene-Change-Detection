# Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms


### Paper

* [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11128568)
* [arXiv link](http://arxiv.org/abs/2409.16850)

### An Introduction Video (3 minutes)

[![](https://img.youtube.com/vi/KX2E8Q5D-Fk/0.jpg)](https://www.youtube.com/watch?v=KX2E8Q5D-Fk)


### Installation

```bash
# clone main repo and corresponding submodule
$ git clone https://github.com/ChadLin9596/Robust-Scene-Change-Detection --recursive

# or
$ git clone https://github.com/ChadLin9596/Robust-Scene-Change-Detection
$ cd <this repository>
$ git submodule init
$ git submodule update

# create a Python 3.9.6 (or other env can run DinoV2) virtual environment
$ source <directory of virtual environment>/bin/activate
(env)$ cd <this repository>
(env)$ pip install -r requirements.txt

# install
(env)$ pip install -e thirdparties/py_utils
(env)$ pip install -e .

```

### Datasets

* download [VL-CMU-CD](https://huggingface.co/datasets/Flourish/VL-CMU-CD/blob/main/VL-CMU-CD-binary255.zip) & [PSCD](https://kensakurada.github.io/pscd/term_of_use.html) datasets

* update the both dataset directories to `Robust-Scene-Change-Detection/data_factory`

### Example usage

* unittest
    ``` bash
    $ cd <this repository>/src/unittest
    $ python -m unittest
    ```

* loading a model and test (please check [inference.ipynb](examples/inference.ipynb))
    ```python
    import torch
    import robust_scene_change_detect.models as models

    B = 1
    H = 504  # need to be 14 * n
    W = 504  # need to be 14 * n

    # load model
    model = models.get_model_from_pretrained("dino_2Cross_CMU")
    model = model.cuda().eval()
    model.module.upsample.size = (H, W)

    # load image
    t0 = torch.rand(B, 3, H, W).cuda()
    t1 = torch.rand(B, 3, H, W).cuda()

    with torch.no_grad():
        pred = model(t0, t1)  # B, H, W, 2
        pred = pred.argmax(dim=-1)  # B, H, W
    ```

* training
    ```bash
    # modify the configuration in scripts/configs/train.yml
    $ python <this repository>/src/scripts/train.py \
        <this repository>/src/scripts/configs/train.yml
    ```

* fine-tune
    ```bash
    # modify the configuration in scripts/configs/fine_tune.yml
    $ python <this repository>/src/scripts/fine_tune.py \
        <this repository>/src/scripts/configs/fine_tune.yml
    ```

* evaluation
    ```bash
    $ python <this repository>/src/scripts/evaluate.py \
        <checkpoint directory>/<name>.pth
    ```

* qualitive results
    ```bash
    $ python <this repository>/scripts/visualize.py \
        <checkpoint directory>/best.val.pth \
        --option <option> \
        --output <directory for qualitive results>
    ```

    | options           | comments                            |
    | ----------------- | ----------------------------------- |
    | VL-CMU-CD         | aligned                             |
    | PSCD              | aligned                             |
    | VL-CMU-CD-diff_1  | unaligned (adjacent distance == 1)  |
    | VL-CMU-CD-diff_-1 | unaligned (adjacent distance == -1) |
    | VL-CMU-CD-diff_2  | unaligned (adjacent distance == 2)  |
    | VL-CMU-CD-diff_-2 | unaligned (adjacent distance == -2) |

### Pretrained Weight

* Train on VL-CMU-CD

| name             | train on VL-CMU-CD    | train on diff VL-CMU-CD   | fine-tune on PSCD   |
| ---------------- | :-------------------: | :-----------------------: | :-----------------: |
| ours (DinoV2)    | [dinov2.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) | [dinov2.2CrossAttn.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.Diff-CMU.pth) | [dinov2.2CrossAttn.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.PSCD.pth) |
| ours (Resnet-18) | [resnet18.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) | / | [resnet18.2CrossAttn.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.PSCD.pth) |
| [C-3PO](https://github.com/DoctorKey/C-3PO) | [resnet18_id_4_deeplabv3_VL_CMU_CD](https://github.com/DoctorKey/C-3PO) | [baseline.c3po.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.c3po.Diff-CMU.pth) | [baseline.c3po.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.c3po.PSCD.pth) |
| [DR-TANet](https://github.com/Herrccc/DR-TANet) | [baseline.drtanet.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.CMU.pth) | [baseline.drtanet.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.Diff-CMU.pth) | [baseline.drtanet.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.PSCD.pth) |
| [CDNet](https://github.com/kensakurada/sscdnet) | [baseline.cdnet.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.cdnet.CMU.pth) | [baseline.cdnet.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.cdnet.Diff-CMU.pth) | / |
| [TransCD](https://github.com/wangle53/TransCD) | [VL-CMU-CD -> Res-SViT_E1_D1_16.pth](https://github.com/wangle53/TransCD) | / | / |

* backbone v.s. comparator

| backbone  | comparator         | train on VL-CMU-CD |
| --------- | ------------------ | ------------------ |
| DinoV2    | Co-Attention       | [dinov2.CoAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.CoAttn.CMU.pth) |
| DinoV2    | Temporal Attention | [dinov2.TemporalAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.TemporalAttn.CMU.pth) |
| DinoV2    | MTF                | [dinov2.MTF.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.MTF.CMU.pth) |
| DinoV2    | 1 CrossAttn        | [dinov2.1CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.1CrossAttn.CMU.pth) |
| DinoV2    | 2 CrossAttn        | [dinov2.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) |
| Resnet-18 | 2 CrossAttn        | [resnet18.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) |

### Changelogs

##### v0.1.0

* Modularize whole package to `robust_scene_change_detect` (can be installed by `pip install -e`)
* Remove relative path setting for easier inference
* Remove evaluation scripts for baselines.
* Support torch hub loading to automatically download checkpoints

##### V0.0.0

* Release datasets module
* Release models module
* Release train/fine-tune/evaluation/visualize scripts
* Release pretraining weight
* Examples of inference on new scenes

### TODO

* [x] release source code
    * [x] release datasets module
    * [x] release models module
    * [x] release train/fine-tune/evaluation/visualize scripts
* [x] release pretraining weight
* [x] examples of inference on new scenes
* [x] support torch hub
* [ ] ~~support Hugging Face (Q3 2025 or earlier...)~~ (no longer needed)
* [x] refactor to `master` and keep baseline scripts into `master-w-baselines` branch (like C3PO)

### BibTex

```
@inproceedings{lin2025robust,
  title        = {Robust scene change detection using visual foundation models and cross-attention mechanisms},
  author       = {Lin, Chun-Jung and Garg, Sourav and Chin, Tat-Jun and Dayoub, Feras},
  booktitle    = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages        = {8337--8343},
  year         = {2025},
  organization = {IEEE}
}
```