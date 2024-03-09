import cv2 as cv
import numpy as np
from pathlib import Path
from time import time_ns
from albument import augment

path = Path("datasets/chess")

images_path = Path(path / "images")
labels_path = Path(path / "labels")

train_images_path = Path(images_path / "train")
train_labels_path = Path(labels_path / "train")

imagesj = train_images_path.glob("*.jpg")
imagesJ = train_images_path.glob("*.JPG")

images = list(imagesj) + list(imagesJ)

out_images_folder = Path("datasets/albumented/images/train")
out_labels_folder = Path("datasets/albumented/labels/train")

for im in images:
    filename = im.name.split(".")[0]
    label_name = filename + ".txt"

    labels = Path(train_labels_path / label_name)

    label_file = open(labels.absolute(), "r")
    image = cv.imread(str(im.absolute()), cv.IMREAD_COLOR)
    h, w = image.shape[0:2]

    bboxes, bboxes_class, keypoints, keypoints_class = [], [], [], []

    for item_idx, l in enumerate(label_file):

        labels_content = l.split(" ")
        labels_content = labels_content[:-1]
        _class = int(labels_content[0])

        points = list(map(float, labels_content[1:]))
        bboxes.append(points[0:4])
        bboxes_class.append(_class)

        kpts1, kpts2, kpts3, kpts4 = (
            points[4:6],
            points[7:9],
            points[10:12],
            points[13:15],
        )

        kpts = [kpts1, kpts2, kpts3, kpts4]
        for idx, (kw, kh) in enumerate(kpts):
            kpts[idx] = [int(kw * w), int(kh * h)]

        keypoints.extend(kpts)
        keypoints_class.append(_class)

    kpts_np = np.array(keypoints, np.float32)

    b = [bboxes, bboxes_class]
    k = [kpts_np, keypoints_class]

    im, bboxes, kpts, classes = augment(image, b, k, False)
    h, w = im.shape[:2]

    kpts = kpts.reshape((len(classes), 4, 3))

    # for b in bboxes:
    #     w2 = (b[2] * w) / 2
    #     h2 = (b[3] * h) / 2
    #     cv.rectangle(
    #         im,
    #         (int(b[0] * w - w2), int(b[1] * h - h2)),
    #         (int(b[0] * w + w2), int(b[1] * h + h2)),
    #         (255, 0, 0),
    #         20,
    #     )

    # for g in kpts:
    #     for k in g:
    #         cv.circle(im, (int(k[0]), int(k[1])), 10, (0, 0, 255), 10)

    for i, g in enumerate(kpts):
        for idx, k in enumerate(g):
            kpts[i, idx, 0] = k[0] / w
            kpts[i, idx, 1] = k[1] / h

    out_name = filename + "_" + str(time_ns())
    out_path = Path(str(out_images_folder / out_name) + ".jpg")
    cv.imwrite(str(out_path.absolute()), im * 255)
    out_labels = Path(str(out_labels_folder / out_name) + ".txt")
    out_labels_file = open(out_labels.absolute(), "w")

    args = []
    for idx, i in enumerate(classes):
        args = []
        args.append(i)

        for b in bboxes[idx]:
            args.append(b)

        for k in kpts[idx]:
            args.append(k[0])
            args.append(k[1])
            args.append(int(k[2]))
        line = list(map(str, args))
        out_labels_file.write(" ".join(line) + "\n")
        out_labels_file.flush()

    out_labels_file.close()
    print("Image saved")
