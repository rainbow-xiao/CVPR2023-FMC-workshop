import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
from glob import glob
import pandas as pd
import os

class People_Car_cl_Dataset(Dataset):
    def __init__(self, df, fold, mode, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)

        self.images = self.df['img_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        # image = self.mirror_padding(image)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index]).long()
        return image, label

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        if self.mode == 'train':
            transforms=(A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=16, border_mode=0, p=0.6),
            ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms
    
class People_Dataset(Dataset):
    def __init__(self, df, fold, mode, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)

        self.images = self.df['img_path'].tolist()
        self.labels = self.df['code'].tolist()
        for i in range(len(self.labels)):
            self.labels[i] = list(map(lambda x:float(x), self.labels[i].strip('[|]').split(',')))
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        # image = self.mirror_padding(image)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.Tensor(self.labels[index])
        return image, label

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        if self.mode == 'train':
            transforms=(A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.6),
            ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms

class People_cl_Dataset(Dataset):
    def __init__(self, df, fold, mode, tag, img_size):
        self.mode = mode
        self.tag = tag
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)
        self.labels = self.get_labels(self.df['code'].tolist())
        self.images = self.df['img_path'].tolist()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.process(image)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index]).long()
        return image, label

    def process(self, image):
        h, w = image.shape[:2]
        if self.tag in ['3', '4']:
            if h>w:
                e = h//2
                image = image[:e]
        elif self.tag in ['5', '6', '7', '9']:
            if h>w:
                s = h//10
                e = s*9
                image = image[s:e]
        elif self.tag in ['11']:
            if h>w:
                s = h//5
                image = image[s:]
        return image

    def mirror_padding(self, img):
        if img.shape[0]>img.shape[1]:
            s1, s2 = img.shape[:2]
            match = (s1-s2)%img.shape[1]!=0
            n = 1+(s1-s2)//img.shape[1]
            if match:
                n += 1
            img = np.concatenate([img if i%2==0 else img[:, ::-1, :] for i in range(n)], axis=1)
            if match:
                clip1 = (s2*n-s1)//2
                clip2 = s2*n-s1-clip1
                img = img[:, clip1:-clip2]
        elif img.shape[0]<img.shape[1]:
            s1, s2 = img.shape[:2]
            match = (s2-s1)%img.shape[0]!=0
            n = 1+(s2-s1)//img.shape[0]
            if match:
                n += 1
            img = np.concatenate([img if i%2==0 else img[::-1, :, :] for i in range(n)], axis=0)
            if match:
                clip1 = (s1*n-s2)//2
                clip2 = s1*n-s2-clip1
                img = img[clip1:-clip2, :]
        return img

    def get_labels(self, labels):
        for i in range(len(labels)):
            if self.tag == '0':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[:2]
            elif self.tag == '1':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[2:5]
            elif self.tag == '2':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[5:8]
            elif self.tag == '3':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[8:10]
            elif self.tag == '4':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[10:12]
            elif self.tag == '5':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[12:14]
            elif self.tag == '6':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[14:17]
            elif self.tag == '7':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[17:19]
            elif self.tag == '8':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[19:21]
            elif self.tag == '9':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[21:26]
            elif self.tag == '10':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[26:28]
            elif self.tag == '11':
                labels[i] = list(map(lambda x:float(x), labels[i].strip('[|]').split(',')))[28:]
            else:
                raise FileNotFoundError(f'No such tag{self.tag}')
        return labels
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img

    def get_transforms(self,):
        if self.mode == 'train':
            if self.tag in ['0', '1', '2']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag in ['3', '4']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag in ['5', '6', '7']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag in ['8', '9']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag in ['10']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag in ['11']:
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms
    
class Car_cl_Dataset(Dataset):
    def __init__(self, df, fold, mode, tag, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)
        self.tag = tag
        self.images = self.df['img_path'].tolist()
        self.labels = self.df[tag].tolist()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index]).long()
        return image, label

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img

    def get_transforms(self,):
        if self.mode == 'train':
            if self.tag == 'type':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag == 'color':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=16, border_mode=0, p=0.6),
                ]))
            elif self.tag == 'brand':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.6),
                ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms

class Car_cl_balance_Dataset(Dataset):
    def __init__(self, df, fold, mode, tag, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)
        self.tag = tag
        self.images = self.df['img_path'].to_numpy()
        self.labels = self.df[tag].to_numpy()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()
        self.step = 0

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index]).long()
        return image, label

    def update(self,):
        index = self.df[f'step_{self.step}'].to_numpy()==True
        self.images = self.df['img_path'].to_numpy()[index]
        self.labels = self.df[self.tag].to_numpy()[index]
        self.step = min(2, self.step+1)

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img

    def get_transforms(self,):
        if self.mode == 'train':
            if self.tag == 'type':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.3, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag == 'color':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.3, rotate_limit=10, border_mode=0, p=0.6),
                ]))
            elif self.tag == 'brand':
                transforms=(A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.3, rotate_limit=10, border_mode=0, p=0.6),
                ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms

class Inference_Dataset(Dataset):
    def __init__(self, mode, img_size, tag=None):
        self.df = pd.read_csv('output/test_people_car_cl_pred.csv')
        self.tag = tag
        self.mode = mode
        if mode == 'people':
            self.images = self.df[self.df['label']==1]['img_path'].tolist()
        else:
            self.images = self.df[self.df['label']==0]['img_path'].tolist()
        self.images = self.images[:100]
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        if self.mode == 'people' and self.tag!=None:
            image = self.process(image)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        return image, self.images[index]

    def process(self, image):
        h, w = image.shape[:2]
        if self.tag in ['3', '4']:
            if h>w:
                e = h//2
                image = image[:e]
        elif self.tag in ['5', '6', '7', '9']:
            if h>w:
                s = h//10
                e = s*9
                image = image[s:e]
        elif self.tag in ['11']:
            if h>w:
                s = h//5
                image = image[s:]
        return image

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        return A.Compose([A.Resize(self.img_size, self.img_size)])

class People_Car_Inference_Dataset(Dataset):
    def __init__(self, test_data_path, img_size):
        self.images = glob(os.path.join(test_data_path, '*'))
        # self.images = self.images[:100]
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        return image, self.images[index]

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        return A.Compose([A.Resize(self.img_size, self.img_size)])

class Car_Pseudo_Label_Dataset(Dataset):
    def __init__(self, df, tag, img_size):
        self.df = df[df[tag].isna()].reset_index(drop=True)
        self.images = self.df['img_path'].tolist()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = self.images[index]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        return image

    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img

    def get_transforms(self,):
        transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms