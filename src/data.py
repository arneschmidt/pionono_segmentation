import os
import torch
import numpy as np
import pandas as pd
import cv2
import random

from torch.utils import data

import albumentations as albu
import utils.globals as globals
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.preprocessing import get_preprocessing_fn_without_normalization


def get_training_augmentation():
    aug_config = globals.config['data']['augmentation']
    if aug_config['use_augmentation']:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Blur(blur_limit=aug_config['gaussian_blur_kernel'], p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=aug_config['brightness_limit'],
                                          contrast_limit=aug_config['contrast_limit'],
                                          p=1.0),
            albu.HueSaturationValue(hue_shift_limit=aug_config['hue_shift_limit'],
                                    sat_shift_limit=aug_config['sat_shift_limit'],
                                    p=1.0),
            albu.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), shear=[-5, 5],
                        rotate=[-360, 360], interpolation=cv2.INTER_CUBIC, cval=[255, 255, 255], cval_mask=4, p=1.0)
        ]
        composed_transform = albu.Compose(train_transform)
    else:
        composed_transform = None
    return composed_transform


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# =============================================

# class SupervisedDataset(torch.utils.data.Dataset):
#     """Supervised Dataset. Read images, apply augmentation and preprocessing transformations.
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing
#             (e.g. normalization, shape manipulation, etc.)
#     """
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             augmentation=None,
#             preprocessing=None
#     ):
#         img_ids = os.listdir(images_dir)
#         mask_ids = os.listdir(masks_dir)
#         self.ids = np.intersect1d(img_ids, mask_ids)
#         if self.ids.size == 0:
#             raise Exception('Empty data generator because no images with masks were found.')
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#         self.class_no = globals.config['data']['class_no']
#         self.class_values = self.set_class_values(self.class_no)
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#
#     def __getitem__(self, i):
#
#         # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], 0)
#         if mask is None:
#             raise Exception('Empty mask! Path: ' + self.masks_fps[i])
#
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype('float')
#
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#         return image, mask, self.ids[i], 0
#
#     def __len__(self):
#         return len(self.ids)
#
#     def set_class_values(self, class_no):
#         if globals.config['data']['ignore_last_class']:
#             class_values = list(range(class_no + 1))
#         else:
#             class_values = list(range(class_no))
#         return class_values


class Dataset(torch.utils.data.Dataset):
    """Crowdsourced_Dataset Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        image_path (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            data_path,
            image_path,
            masks_dirs,
            augmentation=None,
            preprocessing=None,
            repeat_images=None,
            repeat_factor=1,
            annotator_ids='auto',
            _set = None
    ):

        image_path = os.path.join(data_path, image_path)

        mask_paths = [os.path.join(data_path, m) for m in masks_dirs]
        self.annotators = masks_dirs
        self.mask_paths = mask_paths

        self.ids = self.get_valid_ids(os.listdir(image_path), mask_paths, repeat_images, repeat_factor)
        self.images_fps = [os.path.join(image_path, image_id) for image_id in self.ids]

        self.annotators_no = len(self.annotators)
        self.class_no = globals.config['data']['class_no']
        self.class_values = self.set_class_values(self.class_no)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.annotator_ids = annotator_ids

        if globals.config['data']['ignore_last_class']:
            self.ignore_index = int(self.class_no) # deleted class is always set to the last index
        else:
            self.ignore_index = -100 # this means no index ignored

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size_image, _, _ = image.shape
        indexes = np.random.permutation(self.annotators_no)
        for ann_index in indexes:
            ann_path = self.mask_paths[ann_index]
            mask_path = os.path.join(ann_path, self.ids[i])
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                id = self.mask_paths.index(ann_path)
                if self.annotator_ids == 'auto':
                    annotator_id = id
                else:
                    annotator_id = self.annotator_ids[id]
                break

                # print("Exist ", mask_path)
            else:
                # print("Not exist ", mask_path)
                continue

        # apply augmentations
        if self.augmentation:
            # print("Augmentation!")
            sample = self.augmentation(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        mask = [(mask == v) for v in self.class_values]
        mask = np.stack(mask, axis=-1).astype('float')

        # apply preprocessing
        if self.preprocessing:
            # print("Preprocessing!")
            sample = self.preprocessing(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        return image, mask, self.ids[i], annotator_id

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']:
            class_values = list(range(class_no + 1))
        else:
            class_values = list(range(class_no))
        return class_values

    def get_valid_ids(self, image_ids, mask_paths, repeat_images=None, repeat_factor=0):
        """
        Returns all image ids that have at least one corresponding annotated mask
        """
        all_masks = []
        for p in range(len(mask_paths)):
            mask_ids = os.listdir(mask_paths[p])
            for m in mask_ids:
                all_masks.append(m)
        all_unique_masks = np.unique(all_masks)
        valid_ids = np.intersect1d(image_ids, all_unique_masks)

        if repeat_images is not None:
            repeat_ids = np.intersect1d(image_ids, repeat_images)
            for i in range(repeat_factor):
                valid_ids = np.concatenate([valid_ids, repeat_ids], axis=0)

        return valid_ids

def get_data():
    config = globals.config
    batch_size = config['model']['batch_size']
    normalization = config['data']['normalization']

    if normalization:
        encoder_name = config['model']['encoder']['backbone']
        encoder_weights = config['model']['encoder']['weights']
        preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
    else:
        preprocessing_fn = get_preprocessing_fn_without_normalization()

    preprocessing = get_preprocessing(preprocessing_fn)

    print("Annotators: ")
    print(*config['data']['train']['masks'], sep="\n")

    # load all available annotators for training in one loader
    train_dataset = Dataset(config['data']['path'],
                            config['data']['train']['images'],
                            config['data']['train']['masks'],
                            augmentation=get_training_augmentation(),
                            preprocessing = preprocessing,
                            repeat_images=config['data']['repeat_train_images'],
                            repeat_factor=config['data']['repeat_factor'])
    annotators = train_dataset.annotators

    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # validation: create separate loaders for each annotator
    val_masks = config['data']['val']['masks']
    validateloaders = []
    for a in range(len(val_masks)):
        validate_dataset = Dataset(config['data']['path'],
                                   config['data']['val']['images'],
                                   [val_masks[a]],
                                   preprocessing=preprocessing,
                                   annotator_ids=[a])
        validateloaders.append(data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=batch_size, drop_last=False))
    validate_data = (val_masks, validateloaders)

    # test: create separate loaders for each annotator
    test_masks = config['data']['test']['masks']
    testloaders = []
    for a in range(len(test_masks)):
        test_dataset = Dataset(config['data']['path'],
                               config['data']['test']['images'],
                               [test_masks[a]],
                               preprocessing=preprocessing,
                               annotator_ids=[a])
        testloaders.append(data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                           drop_last=False))
    test_data = (test_masks, testloaders)

    return trainloader, validate_data, test_data, annotators
