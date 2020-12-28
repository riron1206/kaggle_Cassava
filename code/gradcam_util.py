# https://qiita.com/bamboo-nova/items/082f71b96b9aca0d5df5
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class GradcamUtil:
    """
    https://github.com/vickyliin/gradcam_plus_plus-pytorch でGrad-Cam
    Usage:
        a_transform = A.Compose(
            [
                A.CenterCrop(height, width, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
        model_paths = "./model_seed_0_fold_0.ckpt"
        model = CassavaLite().load_from_checkpoint(model_path)
        target_layer = model.net.layer4

        grad_util = GradcamUtil(model, target_layer, height, width, a_transform=a_transform)

        im_path = "../input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
        grad_images, pred_max_label_idx = grad_util(im_path)
    """

    def __init__(self, model, target_layer, height, width, a_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.a_transform = a_transform  # albumentations
        self.height = height
        self.width = width
        self.model.eval()

    def load_img_cv2(self, im_path):
        """ファイルパスからcv2で画像ロード"""
        x_orig = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x_orig = cv2.cvtColor(x_orig, cv2.COLOR_BGR2RGB)

        if self.a_transform is None:
            x = torch.from_numpy(x_orig.astype(np.float32)).clone()
        else:
            x = self.a_transform(image=x_orig)["image"]

        x = x.unsqueeze(0)  # 4次元化

        return x, x_orig

    def run_gradcam(self, x, x_orig, model, target_layer, class_idx=None):
        """gradcam実行してヒートマップなどの画像を返す
        class_idx=Noneならスコア最大クラスの結果を返す"""

        # 可視化(visualize_cam)は 1/255 + 4次元化 しないとおかしくなる
        x_orig = x_orig / 255
        transform = A.Compose(
            [A.Resize(self.height, self.width), ToTensorV2(p=1.0)], p=1.0
        )
        x_orig = transform(image=x_orig)["image"]
        x_orig = x_orig.unsqueeze(0)

        gradcam = GradCAM(model, target_layer)
        gradcam_pp = GradCAMpp(model, target_layer)

        mask, logit = gradcam(x, class_idx=class_idx)
        heatmap, result = visualize_cam(mask, x_orig)  # gradcam heatmap, gradcam image

        mask_pp, logit = gradcam_pp(x)
        heatmap_pp, result_pp = visualize_cam(
            mask_pp, x_orig
        )  # gradcam++ heatmap, gradcam++ image

        grad_images = [
            torch.squeeze(x_orig).cpu(),
            heatmap,
            heatmap_pp,
            result,
            result_pp,
        ]

        return grad_images, logit

    def show_gradcam(self, grad_images, pred_max_label_idx=""):
        """run_gradcam()の画像並べて表示"""
        fig = plt.figure(figsize=(18, 16))
        for i, (im, title) in enumerate(
            zip(
                grad_images,
                [
                    "orig",
                    "Grad-Cam heatmap",
                    "Grad-Cam++ heatmap",
                    "Grad-Cam",
                    "Grad-Cam++",
                ],
            )
        ):
            ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
            im = im.numpy().transpose(1, 2, 0)
            plt.title(title + " pred_id=" + str(pred_max_label_idx))
            plt.imshow(im)
        plt.show()
        plt.clf()
        plt.close()

    def __call__(self, im_path, class_idx=None, is_show=True):
        """画像ファイル1件gradcam実行"""
        x, x_orig = self.load_img_cv2(im_path)
        grad_images, logit = self.run_gradcam(
            x, x_orig, self.model, self.target_layer, class_idx=class_idx
        )

        pred_max_label_idx = logit.max(1)[-1].numpy()[0]

        if is_show:
            self.show_gradcam(grad_images, pred_max_label_idx=pred_max_label_idx)

        return grad_images, logit
