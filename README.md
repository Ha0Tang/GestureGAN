[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/GestureGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/GestureGAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/GestureGAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gesturegan-for-hand-gesture-to-gesture/gesture-to-gesture-translation-on-ntu-hand)](https://paperswithcode.com/sota/gesture-to-gesture-translation-on-ntu-hand?p=gesturegan-for-hand-gesture-to-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gesturegan-for-hand-gesture-to-gesture/gesture-to-gesture-translation-on-senz3d)](https://paperswithcode.com/sota/gesture-to-gesture-translation-on-senz3d?p=gesturegan-for-hand-gesture-to-gesture)

## Contents
  - [GestureGAN for Controllable Image-to-Image Translation](#GestureGAN-for-Controllable-Image-to-Image-Translation)
  - [Installation](#Installation)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Generating Images Using Pretrained Model](#Generating-Images-Using-Pretrained-Model)
  - [Training New Models](#Training-New-Models)
  - [Testing](#Testing)
  - [Code Structure](#Code-Structure)
  - [Evaluation](#Evaluation)
  - [Acknowledgments](#Acknowledgments)
  - [Related Projects](#Related-Projects)
  - [Citation](#Citation)
  - [Contributions](#Contributions)
  - [Collaborations](#Collaborations)

![GestureGAN demo](https://github.com/Ha0Tang/GestureGAN/blob/master/imgs/gesture_results.gif)
GestureGAN for hand gesture-to-gesture tranlation task. Given an image and some novel hand skeletons, GestureGAN is able
to generate the same person but with different hand gestures.

![GestureGAN demo](https://github.com/Ha0Tang/GestureGAN/blob/master/imgs/view_results.gif)
GestureGAN for cross-view image tranlation task. Given an image and some novel semantic maps, GestureGAN is able
to generate the same scene but with different viewpoints.

## GestureGAN for Controllable Image-to-Image Translation

### GestureGAN Framework
![Framework](./imgs/gesturegan_framework.jpg)

### Comparison with State-of-the-Art Image-to-Image Transaltion Methods
![Framework Comparison](./imgs/comparison.jpg)

### [Conference paper](https://arxiv.org/abs/1808.04859) | [Extended paper](https://arxiv.org/abs/1912.06112) | [Project page](http://disi.unitn.it/~hao.tang/project/GestureGAN.html) | [Slides](http://disi.unitn.it/~hao.tang/uploads/slides/GestureGAN_MM18.pptx) | [Poster](http://disi.unitn.it/~hao.tang/uploads/posters/GestureGAN_MM18.pdf)

GestureGAN for Hand Gesture-to-Gesture Translation in the Wild.<br>
[Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, [Wei Wang](https://weiwangtrento.github.io/)<sup>1,2</sup>, [Dan Xu](http://www.robots.ox.ac.uk/~danxu/)<sup>1,3</sup>, [Yan Yan](https://userweb.cs.txstate.edu/~y_y34/)<sup>4</sup> and [Nicu Sebe](http://disi.unitn.it/~sebe/)<sup>1</sup>. <br> 
<sup>1</sup>University of Trento, Italy, <sup>2</sup>EPFL, Switzerland, <sup>3</sup>University of Oxford, UK, <sup>4</sup>Texas State University, USA.<br>
In ACM MM 2018 (**Oral** & **Best Paper Candidate**).<br>
The repository offers the official implementation of our paper in PyTorch.

### [License](./LICENSE.md)
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
Copyright (C) 2018 University of Trento, Italy.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [bjdxtanghao@gmail.com](bjdxtanghao@gmail.com).

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/GestureGAN
cd GestureGAN/
```

This code requires PyTorch 0.4.1 and python 3.6+. Please install dependencies by
```bash
pip install -r requirements.txt (for pip users)
```
or 

```bash
./scripts/conda_deps.sh (for Conda users)
```

To reproduce the results reported in the paper, you would need two NVIDIA GeForce GTX 1080 Ti GPUs or two NVIDIA TITAN Xp GPUs.

## Dataset Preparation

For hand gesture-to-gesture translation tasks, we use NTU Hand Digit and Creative Senz3D datasets.
For cross-view image translation task, we use Dayton and CVUSA datasets.
These datasets must be downloaded beforehand. Please download them on the respective webpages. In addition, we put a few sample images in this [code repo](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/samples). Please cite their papers if you use the data. 

**Preparing NTU Hand Digit Dataset**. The dataset can be downloaded in this [paper](https://rose.ntu.edu.sg/Publications/Documents/Action%20Recognition/Robust%20Part-Based%20Hand%20Gesture.pdf). After downloading it we adopt [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate hand skeletons and use them as training and testing data in our experiments. Note that we filter out failure cases in hand gesture estimation for training and testing. Please cite their papers if you use this dataset. Train/Test splits for Creative Senz3D dataset can be downloaded from [here](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/ntu_split). Download images and the crossponding extracted hand skeletons of this dataset:
```bash
bash ./datasets/download_gesturegan_dataset.sh ntu_image_skeleton
```
Then run the following MATLAB script to generate training and testing data:
```bash
cd datasets/
matlab -nodesktop -nosplash -r "prepare_ntu_data"
```

**Preparing Creative Senz3D Dataset**. The dataset can be downloaded [here](https://lttm.dei.unipd.it//downloads/gesture/#senz3d). After downloading it we adopt [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate hand skeletons and use them as training data in our experiments. Note that we filter out failure cases in hand gesture estimation for training and testing. Please cite their papers if you use this dataset. Train/Test splits for Creative Senz3D dataset can be downloaded from [here](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/senz3d_split). Download images and the crossponding extracted hand skeletons of this dataset:
```bash
bash ./datasets/download_gesturegan_dataset.sh senz3d_image_skeleton
```
Then run the following MATLAB script to generate training and testing data:
```bash
cd datasets/
matlab -nodesktop -nosplash -r "prepare_senz3d_data"
```

**Preparing Dayton Dataset**. The dataset can be downloaded [here](https://github.com/lugiavn/gt-crossview). In particular, you will need to download dayton.zip. 
Ground Truth semantic maps are not available for this datasets. We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset.
Train/Test splits for Dayton dataset can be downloaded from [here](https://github.com/Ha0Tang/SelectionGAN/tree/master/datasets/dayton_split).

**Preparing CVUSA Dataset**. The dataset can be downloaded [here](https://drive.google.com/drive/folders/0BzvmHzyo_zCAX3I4VG1mWnhmcGc), which is from the [page](http://cs.uky.edu/~jacobs/datasets/cvusa/). After unzipping the dataset, prepare the training and testing data as discussed in [SelectionGAN](https://arxiv.org/abs/1904.06807). We also convert semantic maps to the color ones by using this [script](https://github.com/Ha0Tang/SelectionGAN/blob/master/scripts/convert_semantic_map_cvusa.m).
Since there is no semantic maps for the aerial images on this dataset, we use black images as aerial semantic maps for placehold purposes.

Or you can directly download the prepared Dayton and CVUSA data from [here](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1#dataset-preparation).

**Preparing Your Own Datasets**. Each training sample in the dataset will contain {Ix,Iy,Cx,Cy}, where Ix=image x, Iy=image y, Cx=Controllable structure of image x, and Cy=Controllable structure of image y.
Of course, you can use GestureGAN for your own datasets and tasks, such landmark-guided facial experssion translation and keypoint-guided person image generation.

## Generating Images Using Pretrained Model

Once the dataset is ready. The result images can be generated using pretrained models.

1. You can download a pretrained model (e.g. ntu_gesturegan_twocycle) with the following script:

```
bash ./scripts/download_gesturegan_model.sh ntu_gesturegan_twocycle
```
The pretrained model is saved at `./checkpoints/[type]_pretrained`. Check [here](https://github.com/Ha0Tang/GestureGAN/blob/master/scripts/download_gesturegan_model.sh) for all the available GestureGAN models.

2. Generate images using the pretrained model.
```bash
python test.py --dataroot [path_to_dataset] \
	--name [type]_pretrained \
	--model [gesturegan_model] \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize [BS] \
	--loadSize [LS] \
	--fineSize [FS] \
	--no_flip
```

`[path_to_dataset]` is the path to the dataset. Dataset can be one of `ntu`, `senz3d`, `dayton_a2g`, `dayton_g2a` and `cvusa`. `[type]_pretrained` is the directory name of the checkpoint file downloaded in Step 1, which should be one of `ntu_gesturegan_twocycle_pretrained`, `senz3d_gesturegan_twocycle_pretrained`, `dayton_a2g_64_gesturegan_onecycle_pretrained`, `dayton_g2a_64_gesturegan_onecycle_pretrained`, `dayton_a2g_gesturegan_onecycle_pretrained`, `dayton_g2a_gesturegan_onecycle_pretrained` and `cvusa_gesturegan_onecycle_pretrained`. 
`[gesturegan_model]` is the directory name of the model of GestureGAN, which should be one of `gesturegan_twocycle` or `gesturegan_onecycle`.
If you are running on CPU mode, change `--gpu_ids 0` to `--gpu_ids -1`. For [`BS`, `LS`, `FS`], please see `Training` and `Testing` sections.

Note that testing requires large amount of disk storage space. If you don't have enough space, append `--saveDisk` on the command line.
    
3. The outputs images are stored at `./results/[type]_pretrained/` by default. You can view them using the autogenerated HTML file in the directory.

## Training New Models

New models can be trained with the following commands.

1. Prepare dataset. 

2. Train.

For NTU dataset:
```bash
export CUDA_VISIBLE_DEVICES=3,4;
python train.py --dataroot ./datasets/ntu \
	--name ntu_gesturegan_twocycle \
	--model gesturegan_twocycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0,1 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--lambda_L1 800 \
	--cyc_L1 0.1 \
	--lambda_identity 0.01 \
	--lambda_feat 1000 \
	--display_id 0 \
	--niter 10 \
	--niter_decay 10
```

For Senz3D dataset:
```bash
export CUDA_VISIBLE_DEVICES=5,7;
python train.py --dataroot ./datasets/senz3d \
	--name senz3d_gesturegan_twocycle \
	--model gesturegan_twocycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0,1 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--lambda_L1 800 \
	--cyc_L1 0.1 \
	--lambda_identity 0.01 \
	--lambda_feat 1000 \
	--display_id 0 \
	--niter 10 \
	--niter_decay 10
```

For CVUSA dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./dataset/cvusa \
	--name cvusa_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 15 \
	--niter_decay 15
```

For Dayton (a2g direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/dayton_a2g \
	--name dayton_a2g_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 20 \
	--niter_decay 15
```

For Dayton (g2a direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=1;
python train.py --dataroot ./datasets/dayton_g2a \
	--name dayton_g2a_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 20 \
	--niter_decay 15
```

For Dayton (a2g direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/dayton_a2g \
	--name dayton_a2g_64_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 50 \
	--niter_decay 50
```

For Dayton (g2a direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=1;
python train.py --dataroot ./datasets/dayton_g2a \
	--name dayton_g2a_64_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 50 \
	--niter_decay 50
```

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `export CUDA_VISIBLE_DEVICES=[GPU_ID]`. Note that train `gesturegan_onecycle` only needs one GPU, while train `gesturegan_twocycle` needs two GPUs.

To view training results and loss plots on local computers, set `--display_id` to a non-zero value and run `python -m visdom.server` on a new terminal and click the URL [http://localhost:8097](http://localhost:8097/).
On a remote server, replace `localhost` with your server's name, such as [http://server.trento.cs.edu:8097](http://server.trento.cs.edu:8097).

### Can I continue/resume my training? 
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train --which_epoch <int> --epoch_count <int+1>` flag. The program will then load the model based on epoch `<int>` you set in `--which_epoch <int>`. Set `--epoch_count <int+1>` to specify a different starting epoch count.


## Testing

Testing is similar to testing pretrained models.

For NTU dataset:
```bash
python test.py --dataroot ./datasets/ntu \
	--name ntu_gesturegan_twocycle \
	--model gesturegan_twocycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip
```

For Senz3D dataset:
```bash
python test.py --dataroot ./datasets/senz3d \
	--name senz3d_gesturegan_twocycle \
	--model gesturegan_twocycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip
```

For CVUSA dataset:
```bash
python test.py --dataroot ./datasets/cvusa \
	--name cvusa_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip
```

For Dayton (a2g direction, 256) dataset:
```bash
python test.py --dataroot ./datasets/dayton_a2g \
	--name dayton_a2g_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip
```

For Dayton (g2a direction, 256) dataset:
```bash
python test.py --dataroot ./datasets/dayton_g2a \
	--name dayton_g2a_gesturegan_onecycle \
	--model gesturegan_onecycle \ 
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip
```

For Dayton (a2g direction, 64) dataset:
```bash
python test.py --dataroot ./datasets/dayton_a2g \
	--name dayton_g2a_64_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip
```

For Dayton (g2a direction, 64) dataset:
```bash
python test.py --dataroot ./datasets/dayton_g2a \
	--name dayton_g2a_64_gesturegan_onecycle \
	--model gesturegan_onecycle \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip
```

Use `--how_many` to specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `models/gesturegan_onecycle_model.py`, `models/gesturegan_twocycle_model.py`: creates the networks, and compute the losses.
- `models/networks/`: defines the architecture of all models for GestureGAN.
- `options/`: creates option lists using `argparse` package.
- `data/`: defines the class for loading images and controllable structures.
- `scripts/evaluation`: several evaluation codes.

## Evaluation

We use several metrics to evaluate the quality of the generated images:

- Hand gesture-to-gesture translation: [Inception Score (IS)](https://github.com/openai/improved-gan) or [Here](https://github.com/Ha0Tang/GestureGAN/tree/master/scripts/evaluation/IS) **|** [Fréchet Inception Distance (FID)](https://github.com/bioinf-jku/TTUR): `pip install tensorflow-gpu==1.14` **|** [PSNR](https://github.com/Ha0Tang/GestureGAN/blob/master/scripts/evaluation/compute_psnr.lua), need install `Lua` **|** [Fréchet ResNet Distance (FRD)](https://github.com/Ha0Tang/GestureGAN/tree/master/scripts/evaluation/FRD), need install `MATLAB 2016+`
- Cross-view image translation: [Inception Score (IS)](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1/scripts/evaluation/compute_topK_KL.py), need install `python 2.7` **|** [Accuracy](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1/scripts/evaluation/compute_accuracies.py), need install `python 2.7` **|** [KL score](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1/scripts/evaluation/KL_model_data.py), need install `python 2.7` **|** 
[SSIM, PSNR, SD](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1/scripts/evaluation/compute_ssim_psnr_sharpness.lua), need install `Lua` **|** [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

## Acknowledgments
This source code is inspired by [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN). We want to thank the NVIDIA Corporation for the donation of the TITAN Xp GPUs used in this work.

## Related Projects
**[BiGraphGAN](https://github.com/Ha0Tang/BiGraphGAN) | [XingGAN](https://github.com/Ha0Tang/XingGAN) | [C2GAN](https://github.com/Ha0Tang/C2GAN) | [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) | [Guided-I2I-Translation-Papers](https://github.com/Ha0Tang/Guided-I2I-Translation-Papers)**

## Citation
If you use this code for your research, please cite our papers.

GestureGAN
```
@article{tang2019unified,
  title={Unified Generative Adversarial Networks for Controllable Image-to-Image Translation},
  author={Tang, Hao and Liu, Hong and Sebe, Nicu},
  journal={IEEE Transactions on Image Processing (TIP)},
  year={2020}
}

@inproceedings{tang2018gesturegan,
  title={GestureGAN for Hand Gesture-to-Gesture Translation in the Wild},
  author={Tang, Hao and Wang, Wei and Xu, Dan and Yan, Yan and Sebe, Nicu},
  booktitle={ACM MM},
  year={2018}
}
```

If you use the original [BiGraphGAN](https://github.com/Ha0Tang/BiGraphGAN), [XingGAN](https://github.com/Ha0Tang/XingGAN), [C2GAN](https://github.com/Ha0Tang/C2GAN), and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) model, please cite the following papers:

BiGraphGAN
```
@article{tang2022bipartite,
  title={Bipartite Graph Reasoning GANs for Person Pose and Facial Image Synthesis},
  author={Tang, Hao and Shao, Ling and Torr, Philip HS and Sebe, Nicu},
  journal={International Journal of Computer Vision (IJCV)},
  year={2022}
}

@inproceedings{tang2020bipartite,
  title={Bipartite Graph Reasoning GANs for Person Image Generation},
  author={Tang, Hao and Bai, Song and Torr, Philip HS and Sebe, Nicu},
  booktitle={BMVC},
  year={2020}
}
```

XingGAN
```
@inproceedings{tang2020xinggan,
  title={XingGAN for Person Image Generation},
  author={Tang, Hao and Bai, Song and Zhang, Li and Torr, Philip HS and Sebe, Nicu},
  booktitle={ECCV},
  year={2020}
}
```

C2GAN
```
@article{tang2021total,
  title={Total Generate: Cycle in Cycle Generative Adversarial Networks for Generating Human Faces, Hands, Bodies, and Natural Scenes},
  author={Tang, Hao and Sebe, Nicu},
  journal={IEEE Transactions on Multimedia (TMM)},
  year={2021}
}

@inproceedings{tang2019cycleincycle,
  title={Cycle In Cycle Generative Adversarial Networks for Keypoint-Guided Image Generation},
  author={Tang, Hao and Xu, Dan and Liu, Gaowen and Wang, Wei and Sebe, Nicu and Yan, Yan},
  booktitle={ACM MM},
  year={2019}
}
```

SelectionGAN
```
@article{tang2022multi,
  title={Multi-Channel Attention Selection GANs for Guided Image-to-Image Translation},
  author={Tang, Hao and Torr, Philip HS and Sebe, Nicu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022}
}

@inproceedings{tang2019multi,
  title={Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Wang, Yanzhi and Corso, Jason J and Yan, Yan},
  booktitle={CVPR},
  year={2019}
}
```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([bjdxtanghao@gmail.com](bjdxtanghao@gmail.com)).

## Collaborations
I'm always interested in meeting new people and hearing about potential collaborations. If you'd like to work together or get in contact with me, please email bjdxtanghao@gmail.com.
___
*It does not matter how slowly you go as long as you do not stop.*
