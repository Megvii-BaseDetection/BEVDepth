## Introduction
BEVDepth is a new 3D object detector with a trustworthy depth
estimation. For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2206.10092).

<img src="assets/vis.png" width="1000" >

## Updates!!
* 【2022/06/21】 We have released our paper on [Arxiv](https://arxiv.org/abs/2206.10092).

## Coming soon

## Benchmark

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**Step 1.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)(v0.18.0).

**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```
**Step 3.** Install BEVDepth(gpu required).
```shell
python setup.py develop
```

### Data preparation
**Step 0.** Download nuScenes official dataset. and symlink the dataset root to `./data/`. The directory will be as follows.
```
bevdepth
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## Cite BEVDepth
If you use BEVDepth in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{li2022bevdepth,
  title={BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection},
  author={Li, Yinhao and Ge, Zheng and Yu, Guanyi and Yang, Jinrong and Wang, Zengran and Shi, Yukang and Sun, Jianjian and Li, Zeming},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```
