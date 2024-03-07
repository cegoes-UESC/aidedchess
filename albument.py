import cv2 as cv
import numpy as np
import albumentations as A


def augment(image, isCoco):

    transform = A.Compose(
        [
            A.Normalize(),
            A.GaussianBlur(p=0.25),
            A.GaussNoise((0, 1), p=0.2),
            A.ColorJitter(),
            A.ShiftScaleRotate(
                p=0.2, border_mode=cv.BORDER_TRANSPARENT, rotate_method="ellipse"
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.Defocus(p=0.2),
            A.Perspective(p=0.15),
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco" if isCoco else "yolo"),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    trans = transform(image=image)
    trans_image = trans["image"]
    h, w = trans_image.shape[:2]

    ones = np.ones((4, 1), dtype=np.int32)
