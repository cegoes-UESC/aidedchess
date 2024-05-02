import cv2 as cv
import numpy as np
import albumentations as A


def augment(image, targets, arguments, length, isCoco):

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
        ),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        additional_targets=targets,
    )

    isValid = False

    while not isValid:
        isValid = True
        trans = transform(image=image, **arguments)
        trans_image = trans["image"]
        h, w = trans_image.shape[:2]

        keypoints, bboxes, classes = [], [], []

        for i in range(length):
            if i == 0:
                kkey, bkey = "keypoints", "bboxes"
            else:
                kkey, bkey = "keypoints" + str(i - 1), "bboxes" + str(i - 1)

            kps = np.array(trans[kkey], np.float32)
            ones = np.ones((kps.shape[0], 1), dtype=np.int32)

            kps = np.hstack([kps, ones])
            out = (kps[:, 0] < 0) | (kps[:, 0] > w) | (kps[:, 1] < 0) | (kps[:, 1] > h)
            kps[out] = 0

            hasKpts = kps[:, 2] != 0
            hasKpts = np.any(hasKpts)

            hasOffScreen = kps[:, 2] == 0
            hasOffScreen = np.any(hasOffScreen)

            if hasOffScreen:
                isValid = False
                break

            if len(trans[bkey]) != 0 and hasKpts:
                keypoints.append(kps)
                bboxes.append(trans[bkey][0][:4])
                classes.append(trans[bkey][0][4])

    return trans["image"], bboxes, np.array(keypoints, np.float32), classes
