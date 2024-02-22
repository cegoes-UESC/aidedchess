import cv2 as cv
from pathlib import Path
import albumentations as A
import numpy as np
from time import time_ns

path = Path("datasets/chess")

images_path = Path(path / "images")
labels_path = Path(path / "labels")

train_images_path = Path(images_path / "train")
train_labels_path = Path(labels_path / "train")

imagesj = train_images_path.glob("*.jpg")
imagesJ = train_images_path.glob("*.JPG")

images = list(imagesj) + list(imagesJ)

for im in images:
    filename = im.name.split(".")[0]
    label_name = filename + ".txt"

    labels = Path(train_labels_path / label_name)

    label_file = open(labels.absolute(), "r")
    image = cv.imread(str(im.absolute()), cv.IMREAD_COLOR)
    h, w = image.shape[0:2]

    cv.namedWindow("image", cv.WINDOW_GUI_NORMAL)
    bs, ks = [], []
    targets = {}
    arguments = {}
    size = 0
    for item_idx, l in enumerate(label_file):
        size = size + 1
        if item_idx != 0:
            targets["bboxes" + str(item_idx - 1)] = "bboxes"
            targets["keypoints" + str(item_idx - 1)] = "keypoints"

        labels_content = l.split(" ")
        labels_content = labels_content[:-1]
        _class = int(labels_content[0])

        points = list(map(float, labels_content[1:]))
        bbs = points[0:4]
        bbs.append(_class)
        bs.append(bbs)

        kpts1, kpts2, kpts3, kpts4 = (
            points[4:6],
            points[7:9],
            points[10:12],
            points[13:15],
        )

        kpts = [kpts1, kpts2, kpts3, kpts4]
        for idx, (kw, kh) in enumerate(kpts):
            kpts[idx] = [int(kw * w), int(kh * h)]

        if item_idx != 0:
            arguments["keypoints" + str(item_idx - 1)] = kpts
            arguments["bboxes" + str(item_idx - 1)] = bs
        else:
            arguments["keypoints"] = kpts
            arguments["bboxes"] = bs

    transform = A.Compose(
        [
            A.Normalize(),
            A.ColorJitter(),
            A.GaussianBlur(p=0.25),
            A.RandomBrightnessContrast(p=0.2),
            A.Defocus(p=0.2),
            A.Perspective(p=0.15),
            A.ShiftScaleRotate(
                p=0.25, border_mode=cv.BORDER_TRANSPARENT, rotate_method="ellipse"
            ),
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.1),
            A.GaussNoise((0, 1)),
        ],
        bbox_params=A.BboxParams(format="yolo"),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        additional_targets=targets,
    )

    trans = transform(image=image, **arguments)
    trans_image = trans["image"]

    h, w = trans_image.shape[:2]

    ones = np.ones((4, 1), dtype=np.int32)

    out_name = filename + "_" + str(time_ns())
    out_path = Path(str(train_images_path / out_name) + ".jpg")
    out_im = cv.imwrite(str(out_path.absolute()), trans_image)
    out_labels = Path(str(train_labels_path / out_name) + ".txt")
    out_labels_file = open(out_labels.absolute(), "w")

    for i in range(size):

        kb = "bboxes"

        bxx = trans[kb]

        bxx = bxx[i]

        _cx = bxx[4]
        bx = bxx[0:4]

        real_out = []
        real_out.append(_cx)
        real_out.extend(bx)

        key = "keypoints" + (str(i - 1) if i != 0 else "")

        kps = trans[key]

        kps = list(map(lambda x: (int(x[0]), int(x[1])), kps))
        kps = np.hstack([kps, ones])

        out = (kps[:, 0] < 0) | (kps[:, 0] > w) | (kps[:, 1] < 0) | (kps[:, 1] > h)
        kps[out] = 0

        kps = kps.reshape(12)

        real_out.extend(kps)
        line = " ".join(list(map(str, real_out)))

        out_labels_file.write(line + "\n")
        out_labels_file.flush()

    out_labels_file.close()
    cv.imshow("image", trans_image)
