# Test-time Domain Adaptation for Monocular Depth Estimation
Code release for the paper
[*Test-time Domain Adaptation for Monocular Depth Estimation*](https://path_to_arxiv)
(accepted in ICRA 2023).

## Prerequisites
Our code is tested on python==3.6.13 with pytorch==1.7.1

Other fundamental packages: tensorboardX, timm, mmcv, numpy, scipy, opencv, matplotlib, pillow, tqdm, json and pickle

Our code is partially based on [monodepth2](https://github.com/nianticlabs/monodepth2) and [NeWCRFs](https://github.com/aliyun/NeWCRFs)

## Datasets
[KITTI](https://www.cvlibs.net/datasets/kitti/) dataset is used for training the source models.
We use the (refined) depth prediction set of the KITTI dataset; please download as instructed on their website.

[DDAD](https://github.com/TRI-ML/DDAD) dataset can be used for training the source models (train set)
as well as adaptation from KITTI source models. After downloading the dataset as instructed,
please install the [tool](https://github.com/TRI-ML/dgp) for data loading.

[Waymo](https://waymo.com/open/) dataset is used for adaptation from source models.
We experiment on the perception dataset v1.2.0; please refer to the [tutorials](https://github.com/waymo-research/waymo-open-dataset)
to install the waymo_open_dataset tool for dataset loading and parsing.

## Getting Started
### Training the Source Models
Please download one of the trained [SwinTransformer]{} backbones and put it in ./models.
Our experiments are conducted with a 224x224 [Swin-L]{https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth}.

To train the supervised source model on the KITTI dataset, run:

```Shell
python train.py --model_name kitti_sup --width 704 --height 352 --data_path PATH_TO_KITTI
```

on DDAD:

```Shell
python train.py --model_name dgp_sup --dataset dgp --width 640 --height 384 --data_path PATH_TO_ddad.json
```

for self-supervised source model on KITTI, run:

```Shell
python train.py --model_name kitti_unsup --monodepth --width 1216 --height 352 --data_path PATH_TO_KITTI
```

on DDAD:

```Shell
python train.py --model_name dgp_unsup --monodepth --dataset dgp --width 960 --height 608 --data_path PATH_TO_ddad.json
```

Other training options please refer to ./options.py

The trained source models will then be saved in ./exp_logs/MODEL_NAME

### Adaptation
To adapt the source models trained on KITTI to DDAD test set, run:

```Shell
python adaptation.py --model_name kitti2dgp --dataset dgp --load_weights_folder ./exp_logs/kitti_sup/models/weights_19 --models_to_load encoder depth --reg_path ./exp_logs/kitti_unsup/models/weights_19 --thres 0.4 --learning_rate 1e-5 --num_workers 0
```

To adapt to Waymo, change --dataset from "dgp" to "waymo".

The accuracy of the source models as well as the online adaptation will be shown.


## Citation
If you find our work useful, please kindly cite:

```
@inproceedings{Li_ICRA2023,
    title={Test-time Domain Adaptation for Monocular Depth Estimation},
    author={Zhi Li and Shaoshuai Shi and Bernt Schiele and Dengxin Dai},
    booktitle = {International Conference on Robotics and Automation (ICRA)},
    year={2023}
}
```

## Contact
For questions, clarifications, please get in touch with:

Zhi Li
[zhili@mpi-inf.mpg.de](zhili@mpi-inf.mpg.de)
