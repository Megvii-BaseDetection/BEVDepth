## MatrixVT
MatrixVT is a novel View Transformer for BEV paradigm with high efficiency and without customized operators. For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2211.10593).

## Try MatrixVT on CPU/GPU
```
python matrixvt.py
```
## Train & Val BEVDepth with MatrixVT
```
python matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py --amp_backend native -b 8 --gpus 8
```
```
python matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```
## Cite BEVDepth
If you use MatrixVT in your research / project, please cite our work by using the following BibTeX entry:

```latex
@article{zhou2022matrixvt,
  title={MatrixVT: Efficient Multi-Camera to BEV Transformation for 3D Perception},
  author={Zhou, Hongyu and Ge, Zheng and Li, Zeming and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2211.10593},
  year={2022}
}
```
