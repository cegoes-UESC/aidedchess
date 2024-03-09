import cv2 as cv
import numpy as np
import albumentations as A


def augment(image, bboxes, kpts, isCoco):

    transform = A.Compose(
        [
            A.Normalize(),
            A.GaussianBlur(p=0.25),
            A.GaussNoise((0, 1), p=0.2),
            A.ColorJitter(),
            A.ShiftScaleRotate(
                p=0.5, border_mode=cv.BORDER_TRANSPARENT, rotate_method="ellipse"
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.Defocus(p=0.2),
            A.Perspective(p=0.15),
            A.VerticalFlip(p=0.25),
            A.HorizontalFlip(p=0.25),
        ],
        bbox_params=A.BboxParams(
            format="coco" if isCoco else "yolo",
            label_fields=["class_labels"],
            check_each_transform=False,
        ),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    trans = transform(
        image=image,
        bboxes=bboxes[0],
        keypoints=kpts[0],
        class_labels=bboxes[1],
        keypoints_classes=kpts[1],
    )
    trans_image = trans["image"]
    h, w = trans_image.shape[:2]

    kps = np.array(trans["keypoints"])
    ones = np.ones((kps.shape[0], 1), dtype=np.int32)

    kps = np.hstack([kps, ones])
    out = (kps[:, 0] < 0) | (kps[:, 0] > w) | (kps[:, 1] < 0) | (kps[:, 1] > h)
    kps[out] = 0

    return trans["image"], trans["bboxes"], kps, trans["class_labels"]
