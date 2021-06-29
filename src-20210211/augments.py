import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

""" オーグメンテーション """
def getTrainAugs(**kwargs) -> A.Compose:
    """
    学習時のオーグメンテーション
    """
    # parameter:
    name = kwargs['name'] if 'name' in kwargs else 'default'
    size = kwargs['size'] if 'size' in kwargs else 512
    # process:
    aug = None
    if name is None or name == 'default':
        aug = [
            A.Resize(size, size, p=1.00),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=20, p=0.50),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    elif name == 'ex1':
        aug = [
            A.OneOf([
                A.JpegCompression(quality_lower=95, quality_upper=100, p=0.40),
                A.Downscale(scale_min=0.10, scale_max=0.15, p=0.20),
            ], p=1.00),
            A.RandomResizedCrop(size, size, scale=(0.9, 1.0), p=1.00),
            A.HorizontalFlip(p=0.50),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=20, p=0.50),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.75),
            A.OneOf([
                A.RandomGamma(p=0.50),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.50),
            ], p=1.00),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.50),
                A.OpticalDistortion(distort_limit=0.50, shift_limit=0.05, p=0.50),
                A.ElasticTransform(alpha=3, p=0.50),
            ], p=0.30),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.50),
                A.GaussianBlur(p=0.50),
                A.MotionBlur(p=0.50),
                A.MedianBlur(p=0.40),
            ], p=0.50),
            A.CoarseDropout(max_holes=4, max_height=48, max_width=48, min_height=16, min_width=16, fill_value=0, p=0.50),
            A.IAAPiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', p=0.20),
            A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.20),
            A.Resize(size, size, p=1.00),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    elif name == 'ex2':
        aug = [
            A.OneOf([
                A.NoOp(),
                A.RandomShadow(),
                A.RandomSnow(),
                A.RandomFog(),
            ]),
            A.OneOf([
                A.NoOp(),
                A.CLAHE(),
                A.RGBShift(),
                A.Posterize(),
                A.ToSepia(p=0.2),
                A.RandomGamma(),
                A.RandomBrightnessContrast(),
            ]),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.JpegCompression(quality_lower=85, quality_upper=100),
                A.Downscale(scale_min=0.75, scale_max=0.95),
            ]),
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=255, p=0.5),
            ]),
            A.OneOf([
                A.OpticalDistortion(p=0.6),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
                A.IAAPiecewiseAffine(p=0.7),
                A.RandomGridShuffle(grid=(4, 4), p=0.7),
            ]),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=20, p=0.50),
            A.Resize(size, size, p=1.00),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.RandomRotate90(p=0.50),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    elif name == 'ex3':
        aug = [
            A.OneOf([
                A.NoOp(),
                A.Sequential([
                    A.RandomCrop(448, 448, p=1.0),
                    A.Resize(512, 512),
                ]),
                A.Sequential([
                    A.RandomCrop(464, 464, p=1.0),
                    A.Resize(512, 512),
                ]),
                A.Sequential([
                    A.RandomCrop(480, 480, p=1.0),
                    A.Resize(512, 512),
                ]),
                A.Sequential([
                    A.RandomCrop(496, 496, p=1.0),
                    A.Resize(512, 512),
                ]),
                A.Resize(512, 512),
            ]),
            A.OneOf([
                A.NoOp(),
                A.RandomSunFlare(),
                A.RandomShadow(),
                A.RandomSnow(),
                A.RandomFog(),
            ]),
            A.OneOf([
                A.CLAHE(),
                A.RGBShift(),
                A.Posterize(),
                A.ToSepia(),
                A.ToGray(),
                A.RandomGamma(),
                A.RandomContrast(),
                A.RandomBrightness(),
                A.RandomBrightnessContrast(),
                ColorFilter(),
            ]),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.JpegCompression(quality_lower=85, quality_upper=100),
                A.Downscale(scale_min=0.75, scale_max=0.95),
            ]),
            A.OneOf([
                A.NoOp(),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=64, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=128, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=192, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=255, p=0.5),
            ]),
            A.OneOf([
                A.OpticalDistortion(p=0.6),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
                A.IAAPiecewiseAffine(p=0.7),
                A.RandomGridShuffle(grid=(4, 4), p=0.7),
            ]),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=20, p=0.50),
            A.Resize(size, size, p=1.00),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.RandomRotate90(p=0.50),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    elif name == 'ex4':
        aug = [
            A.OneOf([
                A.NoOp(),
                A.RandomSunFlare(),
                A.RandomShadow(),
                A.RandomSnow(),
                A.RandomFog(),
            ]),
            A.OneOf([
                A.CLAHE(),
                A.RGBShift(),
                A.Posterize(),
                A.ToSepia(),
                A.ToGray(),
                A.RandomGamma(),
                A.RandomContrast(),
                A.RandomBrightness(),
                A.RandomBrightnessContrast(),
                ColorFilter(),
                A.Sequential([
                    ColorFilter(p=1.0),
                    ColorFilter(p=1.0),
                ]),
            ]),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.JpegCompression(quality_lower=85, quality_upper=100),
                A.Downscale(scale_min=0.75, scale_max=0.95),
            ]),
            A.OneOf([
                A.NoOp(),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=64, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=128, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=192, p=0.5),
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=255, p=0.5),
            ]),
            A.OneOf([
                A.OpticalDistortion(p=0.6),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
                A.IAAPiecewiseAffine(p=0.7),
                A.RandomGridShuffle(grid=(4, 4), p=0.7),
            ]),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=20, p=0.50),
            A.Resize(size, size, p=1.00),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.RandomRotate90(p=0.50),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    else:
        raise NameError('指定されたオーグメンテーションは定義されていません. (name={})'.format(name))
    if aug is None:
        raise NameError('指定されたオーグメンテーションは定義されていません. (name={})'.format(name))
    return A.Compose(aug, p=1.0)

def getValidAugs(**kwargs) -> A.Compose:
    """
    検証時のオーグメンテーション
    """
    # parameter:
    name = kwargs['name'] if 'name' in kwargs else 'default'
    size = kwargs['size'] if 'size' in kwargs else 512
    # process:
    aug = None
    if name is None or name == 'default':
        aug = [
            A.Resize(size, size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ]
    else:
        raise NameError('指定されたオーグメンテーションは定義されていません. (name={})'.format(name))
    if aug is None:
        raise NameError('指定されたオーグメンテーションは定義されていません. (name={})'.format(name))
    return A.Compose(aug, p=1.0)

""" オーグメンテーションモジュール """
class ColorFilter(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(ColorFilter, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        u = np.random.randint(0, w)
        v = np.random.randint(0, h)
        x = np.random.randint(0, w - u)
        y = np.random.randint(0, h - v)
        a = np.random.rand() * 0.5
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        buf = np.zeros([v, u, 3], dtype=np.int32)
        buf[:, :, 0] = np.clip((1.0 - a) * img[y:y+v, x:x+u, 0] + a * r, 0, 255)
        buf[:, :, 1] = np.clip((1.0 - a) * img[y:y+v, x:x+u, 1] + a * g, 0, 255)
        buf[:, :, 2] = np.clip((1.0 - a) * img[y:y+v, x:x+u, 2] + a * b, 0, 255)
        img[y:y+v, x:x+u, :] = buf.astype(np.uint8)
        return img
