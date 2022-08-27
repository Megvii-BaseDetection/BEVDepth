## BEVDepth
BEVDepth is a new 3D object detector with a trustworthy depth
estimation. For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2206.10092).

<img src="assets/backbone.png" width="1000" >

## Updates!!
* 【2022/06/23】 We submitted our result without extra data on [nuScenes Detection Task](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera) and achieved the SOTA.
* 【2022/06/21】 We released our paper on [Arxiv](https://arxiv.org/abs/2206.10092).
* 【2022/04/11】 We submitted our result on [nuScenes Detection Task](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera) and achieved the SOTA.

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**Step 1.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)(v1.0.0rc4).

**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```
**Step 3.** Install BEVDepth(gpu required).
```shell
python setup.py develop
```

### Data preparation
**Step 0.** Download nuScenes official dataset.

**Step 1.** Symlink the dataset root to `./data/`.
```
ln -s [nuscenes root] ./data/
```
The directory will be as follows.
```
BEVDepth
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
**Step 2.** Prepare infos.
```
python scripts/gen_info.py
```
**Step 3.** Prepare depth gt.
```
python scripts/gen_depth_gt.py
```

### Tutorials
**Train.**
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
**Eval.**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

### Benchmark
|Exp |EMA| CBGS |mAP |mATE| mASE | mAOE |mAVE| mAAE | NDS | weights |
| ------ | :---: | :---: | :---:       |:---:     |:---:  | :---: | :----: | :----: | :----: | :----: |
|[R50](exps/bev_depth_lss_r50_256x704_128x128_24e_2key_ema.py)| | |0.3304| 0.7021| 0.2795| 0.5346| 0.5530| 0.2274| 0.4355 | [github](https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth)
|[R50](exps/bev_depth_lss_r50_256x704_128x128_24e_2key_ema.py)|√ | |0.3329 |  0.6832     |0.2761 | 0.5446 | 0.5258 | 0.2259 | 0.4409 | [github](https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key_ema.pth)
|[R50](exps/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.py)| |√ |0.3484| 0.6159| 0.2716| 0.4144| 0.4402| 0.1954| 0.4805 | [github](https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth)
|[R50](exps/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.py)|√  |√ |0.3589 |  0.6119     |0.2692 | 0.5074 | 0.4086 | 0.2009 | 0.4797 | [github](https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.pth) |

## FAQ

### EMA
- The results are differnt between evaluation during training and evaluation from ckpt.

Due to the working mechanism of EMA, the model parameters saved by ckpt are different from the model parameters used in the training stage.

- EMA exps are unable to resume training from ckpt.

We used the customized EMA callback and this function is not supported for now.

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
