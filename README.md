# DMRNet : Towards Effective Feature Learning for One-Step Person Search

Code for the paper ["Decoupled and Memory-Reinforced Networks: Towards Effective Feature Learning for One-Step Person Search"](https://arxiv.org/abs/2102.10795).


## Installation
This project is developed upon [RepPoints](https://github.com/microsoft/RepPoints) and [MMdetection](https://github.com/open-mmlab/mmdetection). Here we provide a step-by-step installation. 
```bash
conda create --name dmrnet
source activate dmrnet

pip install cython tqdm pandas sklearn
pip install torch==1.4.0 torchvision==0.5.0
pip install mmcv==0.2.14

git clone https://github.com/nuannuanhcc/DMRNet.git ChuchuHan
cd DMRNet/mmdetection
pip install -v -e .
```
## Dataset
Download [CUHK-SYSU](https://drive.google.com/file/d/1uVtxdNG-RaKzN8fGIJj4XzI9rGU55oeh/view?usp=sharing) and [PRW](https://drive.google.com/file/d/1hoFfjVqqBtdZnYl0ihJc6TG2ARcWtQ5X/view?usp=sharing) to `DMRNet/data/`
```bash
cd DMRNet && mkdir data
cd data
unzip sysu.zip && unzip prw.zip
```
## Experiments
1. Train
```bash
sh run.sh
```
2. Test
```bash
sh erun.sh
```
## Results and models

The results on CUHK-SYSU and PRW are shown in the table below.

| Dataset | Detector | mAP | Rank-1 | Download |
| :----: | :-------: | :------: | :-----: | :------: |
| CUHK-SYSU | RepPoints | 93.21% | 94.31% |[model](https://drive.google.com/file/d/1ITVbbSrZKc9aa_aYqD3K-3HE4qMJ49aV/view?usp=sharing) |
| PRW | RepPoints | 47.05% | 83.52% |[model](https://drive.google.com/file/d/13ZNF-aSN4V9F4ApbTyWAW3SS3h0w7_BS/view?usp=sharing) |

## Citation
Please cite the following paper in your publications if it helps your research:
```
@inproceedings{han2021decoupled,
  title={Decoupled and Memory-Reinforced Networks: Towards Effective Feature Learning for One-Step Person Search},
  author={Han, Chuchu and Zheng, Zhedong and Gao, Changxin and Sang, Nong and Yang, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={2},
  pages={1505--1512},
  year={2021}
}
```