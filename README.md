# CVPR 2023 1st foundation model challenge-Track2
# Leaderboard A: 3rd Place Solution

#### [Competition on kaggle](https://aistudio.baidu.com/aistudio/competition/detail/891/0/introduction)
#### [ECCV 2022 Instance-Level Recognition workshop](https://ilr-workshop.github.io/ECCVW2022/)

## Code arch
|   
|
|
|
|
## HARDWARE & SOFTWARE

Ubuntu 22.04

CPU: 13900k

GPU: 1 * 4090, 24G

Python: 3.9.13

Pytorch: 2.0.0+cu118

## Data Preparation
1. Download data from the [official link](https://aistudio.baidu.com/aistudio/datasetdetail/203278)

2. Run **data_analyzing.ipynb** to explore the dataset and do caption->label mapping, dataset merging, etc.

3. Run **Data_preparing.ipynb** to split dataset with stratified Kfold for local validation.

4. Run **Data_Merge.ipynb** to merge all the csvs, and do sampling and resamping. Will get **final_data_224_sample_balance.csv**. 


## Pretrained Models
1. EVA02_CLIP_L_336_psz14_s6B(visual) and EVA02_CLIP_L_psz14_s4B(visual) from [EVA](https://github.com/baaivision/EVA)
 
2. eva02_large_patch14_448.mim_m38m_ft_in22k_in1k from [timm](https://github.com/huggingface/pytorch-image-models)
 
3. ConvNext-XXLarge-soup from [open_clip_torch](https://github.com/mlfoundations/open_clip)

## Training
2. Training:
```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python -m torch.distributed.launch --nproc_per_node=6 \
./GUIE/train.py \
--csv-dir ./final_data_224_sample_balance_fold.csv \
--config-name 'vit_224' \
--image-size 224 \
--batch-size 32 \
--num-workers 10 \
--init-lr 1e-4 \
--n-epochs 10 \
--cpkt_epoch 10 \
--n_batch_log 300 \
--warm_up_epochs 1 \
--fold 1
```


## Contact
Email: 3579628328@qq.com
