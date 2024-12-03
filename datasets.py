import torch
import random
import os, cv2
import warnings
import numpy as np
import pandas as pd
import rasterio as rio
import albumentations as album
from utils_analyse import one_image
from utils_visualize import visualize
from torch.utils.data import DataLoader
from utils_UI import print_error_message
from utils import colour_code_segmentation


warnings.filterwarnings("ignore", message="TIFFReadDirectory: Warning, Unknown field with tag *")
warnings.filterwarnings("ignore")


def to_tensor(x, **kwargs):
    if len(x.shape)==2:#this is a mask so do nothing
        return x.astype('float32')
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def get_training_augmentation():
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
        album.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.5),
        album.ElasticTransform(p=0.5),
        album.GridDistortion(p=0.5),
        album.RandomBrightnessContrast(p=0.5),
        album.HueSaturationValue(p=0.5),
        album.GaussNoise(p=0.5),
        album.RandomFog(p=0.2),
        album.RandomRain(p=0.2),
        album.RGBShift(p=0.5),
    ]
    return album.Compose(train_transform)

def get_training_augmentation_with_rand_crop():
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
        album.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.5),
        album.ElasticTransform(p=0.5),
        album.GridDistortion(p=0.5),
        album.RandomBrightnessContrast(p=0.5),
        album.HueSaturationValue(p=0.5),
        album.GaussNoise(p=0.5),
        album.RandomFog(p=0.2),
        album.RandomRain(p=0.2),
        album.RGBShift(p=0.5),
        album.RandomCrop(height=256, width=256, always_apply=True)
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True)
    ]
    return album.Compose(test_transform)


def get_test_augmentation():
    # do nothing!
    return album.Compose([])


# This is for random cropping option both datasets
class BuildingsDataset(torch.utils.data.Dataset):
    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
            args = None):


        if args.ds_id == 3: #GBSS
            with open(images_dir, 'r') as f:
                self.image_paths = [line.strip() for line in f.readlines()]

            with open(masks_dir, 'r') as f:
                self.mask_paths = [line.strip() for line in f.readlines()]
        else: #other datasets
            self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
            self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.images_dir = images_dir

    def __getitem__(self, i):
        # print(self.image_paths[i], i)
        src = rio.open(self.image_paths[i])
        image = src.read()
        image = image.transpose(1, 2, 0)

        if "/massachusetts-buildings-dataset/" in self.images_dir:
            mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
            mask = np.all(mask == [255, 255, 255], axis=-1).astype(int)
        elif "/WHU/" in self.images_dir:
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)
            mask[mask != 0] = 1
        elif "/Inria/" in self.images_dir:
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)
            mask[mask != 0] = 1
        elif "/GBSS/" in self.images_dir:
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)
            mask[mask != 0] = 1

        # apply augmentations
        if self.augmentation:
            image = image.astype('uint8')  # Convert to uint8 format
            mask = mask.astype('uint8')  # Convert to uint8 format
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return torch.tensor(image),  torch.tensor(mask)

    def __len__(self):
        return len(self.image_paths)



def create_GBSS_dataset(args):
    ds_id = 3 # please do not change! this is hard coded

    if os.name == 'posix':
        # print("This is a Linux system.")
        DATA_DIR = '/home/kerim/Datasets/GBSS/'
    else:
        print_error_message("This is another system. That I do not know", True)

    if args.isAISurrey:
        DATA_DIR = "./GBSS/"
        # DATA_DIR = "/vol/research/ak0084_datasets/Datasets/GBSS/"
        # print(os.listdir('.'))


    x_train_dir = os.path.join(DATA_DIR, 'splits/train_images.txt')
    y_train_dir = os.path.join(DATA_DIR, 'splits/train_labels.txt')

    x_valid_dir = os.path.join(DATA_DIR, 'splits/valid_images.txt')
    y_valid_dir = os.path.join(DATA_DIR, 'splits/valid_labels.txt')

    x_test_dir = os.path.join(DATA_DIR, 'splits/test_images.txt')
    y_test_dir = os.path.join(DATA_DIR, 'splits/test_labels.txt')

    class_dict = pd.read_csv(DATA_DIR+"/label_class_dict.csv")

    dataset, train_dataset, valid_dataset, tst_dataset, args = create_splits_given_proc_mode(x_train_dir, y_train_dir,
                                                                                       x_valid_dir, y_valid_dir,
                                                                                       x_test_dir, y_test_dir, DATA_DIR, args,class_dict)
    return dataset, train_dataset, valid_dataset, tst_dataset, args


def create_Inria_dataset(args):
    ds_id = 2 # please do not change! this is hard coded

    if os.name == 'posix':
        # print("This is a Linux system.")
        DATA_DIR = '/home/kerim/Datasets/Inria/'
    else:
        print_error_message("This is another system. That I do not know", True)

    if args.isAISurrey:
        # DATA_DIR = "/mnt/fast/nobackup/users/ak0084/Datasets/Inria/"
        DATA_DIR = "./Inria/"

    x_train_dir = os.path.join(DATA_DIR, 'train/Image')
    y_train_dir = os.path.join(DATA_DIR, 'train/Mask')

    x_valid_dir = os.path.join(DATA_DIR, 'val/Image')
    y_valid_dir = os.path.join(DATA_DIR, 'val/Mask')

    x_test_dir = os.path.join(DATA_DIR, 'test/Image')
    y_test_dir = os.path.join(DATA_DIR, 'test/Mask')

    #you need to add counterpart for windows [feature]
    class_dict = pd.read_csv(DATA_DIR+"/label_class_dict.csv")


    dataset, train_dataset, valid_dataset, tst_dataset, args = create_splits_given_proc_mode(x_train_dir, y_train_dir,
                                                                                       x_valid_dir, y_valid_dir,
                                                                                       x_test_dir, y_test_dir, DATA_DIR, args, class_dict)
    return dataset, train_dataset, valid_dataset, tst_dataset,  args


def create_WHU_dataset(args):
    ds_id = 1 # please do not change! this is hard coded

    if os.name == 'posix':
        # print("This is a Linux system.")
        DATA_DIR = '/home/kerim/Datasets/WHU/'
    else:
        print_error_message("This is another system. That I do not know", True)

    if args.isAISurrey:
        # DATA_DIR = "/mnt/fast/nobackup/users/ak0084/Datasets/WHU/"
        DATA_DIR = "./WHU/"

    x_train_dir = os.path.join(DATA_DIR, 'train/Image')
    y_train_dir = os.path.join(DATA_DIR, 'train/Mask')

    x_valid_dir = os.path.join(DATA_DIR, 'val/Image')
    y_valid_dir = os.path.join(DATA_DIR, 'val/Mask')

    x_test_dir = os.path.join(DATA_DIR, 'test/Image')
    y_test_dir = os.path.join(DATA_DIR, 'test/Mask')

    class_dict = pd.read_csv(DATA_DIR+"/label_class_dict.csv")

    dataset, train_dataset, valid_dataset, tst_dataset, args = create_splits_given_proc_mode(x_train_dir, y_train_dir,
                                                                                       x_valid_dir, y_valid_dir,
                                                                                       x_test_dir, y_test_dir, DATA_DIR, args,class_dict)
    return dataset, train_dataset, valid_dataset, tst_dataset,  args

def create_massachusetts_dataset(args):
    ds_id = 0 # please do not change! this is hard coded

    if os.name == 'posix':
        # print("This is a Linux system.")
        DATA_DIR = '/home/kerim/Datasets/massachusetts-buildings-dataset/tiff/'
    else:
        print_error_message("This is another system. That I do not know", True)

    if args.isAISurrey:
        DATA_DIR = "./massachusetts-buildings-dataset/tiff/"

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'test_labels')

    #you need to add counterpart for windows [feature]
    class_dict = pd.read_csv(DATA_DIR.replace("/tiff/","")+"/label_class_dict.csv")


    dataset, train_dataset, valid_dataset, tst_dataset, args = create_splits_given_proc_mode(x_train_dir, y_train_dir,
                                                                                       x_valid_dir, y_valid_dir,
                                                                                       x_test_dir, y_test_dir, DATA_DIR, args, class_dict)
    return dataset, train_dataset, valid_dataset, tst_dataset,  args


def create_splits_given_proc_mode(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir,x_test_dir, y_test_dir, DATA_DIR, args, class_dict):


    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'building']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    args.select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    if args.train_ds_proc_id==0:#random cropping
        dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=args.select_class_rgb_values,args=args)

        # Get train and val dataset instances
        train_dataset = BuildingsDataset(
            x_train_dir, y_train_dir,
            augmentation=get_training_augmentation_with_rand_crop(),#get_validation_augmentation()
            preprocessing=get_preprocessing(preprocessing_fn=None),
            class_rgb_values=args.select_class_rgb_values,args=args
        )


    elif args.train_ds_proc_id==1:#guided/selected cropping
        dataset = BuildingsDatasetCustom(x_train_dir, y_train_dir, class_rgb_values=None, augmentation=None,
                                         preprocessing=None, images_dir=DATA_DIR, ds_id=args.ds_id,args=args)


        # Get train and val dataset instances
        train_dataset = BuildingsDatasetCustom(
            x_train_dir, y_train_dir, class_rgb_values=None, augmentation=None,  preprocessing=None, images_dir=DATA_DIR,
            ds_id=args.ds_id,args=args)


    # the same validation and test data for both data processing modes (random and selective modes)
    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir,
        class_rgb_values=args.select_class_rgb_values,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn=None), args=args
    )

    # no random cropping!!!
    tst_dataset = BuildingsDataset(
        x_test_dir, y_test_dir,
        class_rgb_values=args.select_class_rgb_values,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn=None), args=args
    )

    return  dataset, train_dataset, valid_dataset, tst_dataset, args

# this should be used for both datasets!!
class BuildingsDatasetCustom(torch.utils.data.Dataset):
    def __init__(
            self,
            x_dir,
            y_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
            images_dir=None,
            ds_id = -1,
            args = None):

        if args.ds_id == 3: #GBSS
            with open(x_dir, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]

            with open(y_dir, 'r') as f:
                mask_paths = [line.strip() for line in f.readlines()]
        else: #other datasets
            image_paths = [os.path.join(x_dir, image_id) for image_id in sorted(os.listdir(x_dir))]
            mask_paths = [os.path.join(y_dir, image_id) for image_id in sorted(os.listdir(y_dir))]

        self.image_paths = sorted(image_paths)
        self.mask_paths =  sorted(mask_paths)

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.images_dir = images_dir

        self.min_pixel_count = args.min_pixel_count
        self.max_pixel_count = args.max_pixel_count
        self.ds_id = ds_id
        self.MAX_NUM_CROPS = args.MAX_NUM_CROPS
        self.visualize_crops = args.visualize_crops
        self.visualize_image = args.visualize_image

    def __getitem__(self, i):

        while True:  # to handle issues where no object is found in the entire big image!!
            src = rio.open(self.image_paths[i])
            image = src.read()
            image = image.transpose(1, 2, 0)

            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)

            if self.ds_id==0: #Massachusetts
                mask = np.all(mask == [255, 255, 255], axis=-1).astype(int)
            elif self.ds_id==1: #WHU
                mask[mask != 0] = 1
            elif self.ds_id==2: #Inria
                mask[mask != 0] = 1
            elif self.ds_id==3: #GBSS
                mask[mask != 0] = 1
            else:
                print_error_message("Dataset id is wrong, please check!",True)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']


            # select a crop from the big image
            if random.random() > 0:
                meta_data, crop_image, crop_mask = one_image(self.image_paths[i], image, mask, min_pixel_count=self.min_pixel_count,
                                             max_pixel_count=self.max_pixel_count, visualize_image=self.visualize_image, visualize_crops=self.visualize_crops,
                                             ds_names=None, ds_id=self.ds_id, clss_rgb=None,MAX_NUM_CROPS=self.MAX_NUM_CROPS)

                # print(meta_data)
                if crop_image is None or len(meta_data) == 0: # when not object was found in the big image
                    i = random.randint(0, len(self.image_paths) - 1)
                else:
                    return torch.tensor(crop_image).permute(2,0,1).float(), torch.tensor(crop_mask).float()
            else: #  return a random crop
                x = random.randint(0, image.shape[0] - 256)
                y = random.randint(0, image.shape[1] - 256)

                # Perform the crop
                crop_image = image[y:y + 256, x:x + 256,:]
                crop_mask = mask[y:y + 256, x:x + 256]
                return torch.tensor(crop_image).permute(2, 0, 1).float(), torch.tensor(crop_mask).float()

    def __len__(self):
        return len(self.image_paths)

def select_dataset(args):
    if args.ds_id == 0:
        dataset, train_dataset, valid_dataset, tst_dataset, args = create_massachusetts_dataset(args)
    elif args.ds_id == 1:
        dataset, train_dataset, valid_dataset, tst_dataset, args = create_WHU_dataset(args)
    elif args.ds_id == 2:
        dataset, train_dataset, valid_dataset, tst_dataset, args = create_Inria_dataset(args)
    elif args.ds_id == 3:
        dataset, train_dataset, valid_dataset, tst_dataset, args = create_GBSS_dataset(args)

    else:
        print_error_message('Error - Dataset id is not valid!', True)

    for ix in range(0):
        random_idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[random_idx]

        visualize(
            ds_name=args.ds_names[args.ds_id] + "original(cropped)",
            original_image=image,
            ground_truth_mask=colour_code_segmentation(mask, args),
            one_hot_encoded_mask=mask
        )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.vl_batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=False, pin_memory=True)
    tst_loader = DataLoader(tst_dataset, batch_size=args.tst_batch_size, shuffle=True, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True)

    return train_dataset, valid_dataset, tst_dataset, args, train_loader, valid_loader, tst_loader