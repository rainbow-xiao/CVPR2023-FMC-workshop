# [CVPR 2023 1st foundation model challenge-Track2](https://aistudio.baidu.com/aistudio/competition/detail/891/0/introduction)
# Leaderboard A: 3rd Place Solution


# HARDWARE & SOFTWARE

Ubuntu 22.04

CPU: 13900k

GPU: 1 * 4090, 24G

Python: 3.9.13

Pytorch: 2.0.0+cu118

# Arch
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

# Pipeline
1. Download data from the [official link](https://aistudio.baidu.com/aistudio/datasetdetail/203278)

2. Run **data_analyzing.ipynb** to explore the dataset and do caption->label mapping, dataset merging, etc.

3. Run **Data_preparing.ipynb** to split dataset with stratified Kfold for local validation.

4. Train 3 models for 3 sub-cl-tasks of car.

5. Train 12 models for 12 sub-cl-tasks of pedestrian.

6. Train a Pedestrian-Car general classification Model.

7. Split pedestrian and car of test data with model trained in step 6.

8. Inference of sub-cl-tasks of car with models trained in step 4, and retrival top10 imgs.

9. Inference of sub-cl-tasks of people with models trained in step 5, and retrival top10 imgs.

10. Merge results from step 8 and step 9 to make **submission.json**.

# Pretrained Models
1. EVA02_CLIP_L_336_psz14_s6B(visual) and EVA02_CLIP_L_psz14_s4B(visual) from [EVA](https://github.com/baaivision/EVA)
 
2. eva02_large_patch14_448.mim_m38m_ft_in22k_in1k from [timm](https://github.com/huggingface/pytorch-image-models)
 
3. ConvNext-XXLarge-soup from [open_clip_torch](https://github.com/mlfoundations/open_clip)

# Training
1. Car Classification Training:
```bash
!CUDA_VISIBEL_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
CVPR/train_car_cl.py \
--csv-dir data/train_val_cars_*{task}*_10fold.csv \             task in [type, color, brand]
--config-name *{cfg}* \                      cfg in [config_eva_vit_car_cl, config_eva_02_car_cl, config_conv_car_cl]
--image-size *{size}* \                      size in [224 (eva-l), 280 (conv), 336 (eva-l-336), 448 (eva02-448)]
--epochs 10 \
--init-lr 3e-5 \
--batch-size 32 \
--num-workers 8 \
--nbatch_log 300 \
--warmup_epochs 0 \
--fold 1
```
---
2. Pedestrian Classification Training:
```bash
!CUDA_VISIBEL_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
CVPR/train_people_cl.py \
--csv-dir data/train_val_peoples_code_fold_*{task}*.csv \          task in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
--config-name *{cfg}* \                   cfg in [config_eva_vit_car_cl, config_eva_02_car_cl, config_conv_car_cl]
--image-size *{size}* \                   size in [224 (eva-l), 280 (conv), 336 (eva-l-336), 448 (eva02-448)]
--epochs 16 \
--init-lr 3e-5 \
--batch-size 32 \
--num-workers 8 \
--nbatch_log 300 \
--warmup_epochs 1 \
--fold 1
```
---
3. Pedestrian-Car Classification Training:
```bash
!CUDA_VISIBEL_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
CVPR/train_people_car_cl.py \
--csv-dir data/train_val_cl_20fold.csv \
--config-name 'config_eva_vit_people_car_cl' \
--image-size 224 \
--epochs 11 \
--init-lr 3e-5 \
--batch-size 32 \
--num-workers 8 \
--nbatch_log 300 \
--warmup_epochs 0 \
--fold 1
```

# Inference
1. Pedestrian Classification Inference(Emsenble):
```bash
!python CVPR/inference_car_emsenble.py \
--csv-dir data/test/test_car.csv \
--config_names config_eva_02_car_cl config_conv_car_cl \
--image-sizes 448 320 \
--model-weights 0.8 0.2 \
--model_paths1 output/car/eva_02-car-type/eva_02-448_best_ep5.pth output/car/conv-car-type/convnext_xxlarge_best_ep5.pth \
--model_paths2 output/car/eva_02-car-color/eva_02-448_best_ep5.pth output/car/conv-car-color/convnext_xxlarge_best_ep5.pth \
--model_paths3 output/car/eva_02-car-brand/eva_02-448_best_ep5.pth output/car/conv-car-brand/convnext_xxlarge_best_ep5.pth \
--batch-size 32 \
--num-workers 8 \
```
---
2. Pedestrian Classification Inference(Emsenble):
```bash
!python CVPR/inference_people_cl_emsenble.py \
--csv-dir data/test/test_people.csv \
--config_names config_eva_vit_people_cl config_eva_02_people_cl \
--image-sizes 224 448 \
--model-weights 0.5 0.5 \
--batch-size 32 \
--model_path1 \
    output/people/eva-l-people-0/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-1/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-2/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-3/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-4/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-5/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-6/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-7/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-8/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-9/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-10/eva-cl-l_best_ep8.pth \
    output/people/eva-l-people-11/eva-cl-l_best_ep8.pth \
--model_path2 \
    output/people/eva02-l-448-people-0/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-1/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-2/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-3/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-4/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-5/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-6/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-7/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-8/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-9/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-10/eva_02-448_best_ep8.pth \
    output/people/eva02-l-448-people-11/eva_02-448_best_ep8.pth \
--num-workers 8 \
--nbatch_log 300 \
```
---
3. Pedestrian-Car General Classification Inference:
```bash
!python CVPR/inference_people_car_cl.py \
--test_data_path data/test/test_images \
--config-name config_eva_vit_people_car_cl \
--image-size 224 \
--batch-size 32 \
--model_path output/cl/eva-l/eva-cl-l_best_ep2.pth \
--num-workers 8 \
--nbatch_log 300 \
```
## Contact
Email: 3579628328@qq.com
