# CVPR 2023 1st foundation model challenge-Track2
# Leaderboard A: 3rd Place Solution

#### [Competition](https://aistudio.baidu.com/aistudio/competition/detail/891/0/introduction)

## HARDWARE & SOFTWARE

Ubuntu 22.04

CPU: 13900k

GPU: 1 * 4090, 24G

Python: 3.9.13

Pytorch: 2.0.0+cu118

## Arch
```
|-- CVPR/
|   |-- models
|      |-- ...
|   |-- utils
|      |-- ...
|   |-- ...
|-- data/
|   |-- train
|      |-- train_images
|         |-- ...
|      |-- train_label.txt
|   |-- test
|      |-- test_images
|         |-- ...
|      |-- test_label.txt
|   |-- val
|      |-- val_images
|         |-- ...
|      |-- val_label.txt
|-- ...
```

## Data Preparation
1. Download data from the [official link](https://aistudio.baidu.com/aistudio/datasetdetail/203278)

2. Run **data_analyzing.ipynb** to explore the dataset and do caption->label mapping, dataset merging, etc.

3. Run **Data_preparing.ipynb** to split dataset with stratified Kfold for local validation.

## Pretrained Models
1. EVA02_CLIP_L_336_psz14_s6B(visual) and EVA02_CLIP_L_psz14_s4B(visual) from [EVA](https://github.com/baaivision/EVA)
 
2. eva02_large_patch14_448.mim_m38m_ft_in22k_in1k from [timm](https://github.com/huggingface/pytorch-image-models)
 
3. ConvNext-XXLarge-soup from [open_clip_torch](https://github.com/mlfoundations/open_clip)

## Training
1. Car Classification Training:
```bash
!CUDA_VISIBEL_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
CVPR/train_car_cl.py \
--csv-dir data/train_val_cars_type**{task}**_10fold.csv \                     task in [type, color, brand]
--config-name 'config_eva_vit_car_cl' \
--image-size 224**{size}** \                                                  size in [224, 336]
--epochs 10 \
--init-lr 3e-5 \
--batch-size 32 \
--num-workers 8 \
--nbatch_log 300 \
--warmup_epochs 0 \
--fold 1
```


## Contact
Email: 3579628328@qq.com
