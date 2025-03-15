from pathlib import Path
from time import time_ns

import cv2 as cv

from .albument import augment


def augment_yolo():

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

    LEN_IMAGES = len(images)

    for idx, im in enumerate(images):
        print(f"=== Processing image {idx+1}/{LEN_IMAGES} ===")
        filename = im.name.split(".")[0]
        label_name = filename + ".txt"

        labels = Path(train_labels_path / label_name)

        label_file = open(labels.absolute(), "r")
        image = cv.imread(str(im.absolute()), cv.IMREAD_COLOR)
        h, w = image.shape[0:2]

        bboxes, keypoints, visibility = [], [], []

        targets, arguments = {}, {}

        size = 0

        for item_idx, label in enumerate(label_file):
            size = size + 1
            bboxes, keypoints = [], []

            if item_idx != 0:
                targets["bboxes" + str(item_idx - 1)] = "bboxes"
                targets["keypoints" + str(item_idx - 1)] = "keypoints"

            labels_content = label.split(" ")
            _class = int(labels_content[0])

            points = list(map(float, labels_content[1:]))
            bboxes.extend(points[0:4])
            bboxes.append(_class)

            kpts1, kpts2, kpts3, kpts4 = (
                points[4:6],
                points[7:9],
                points[10:12],
                points[13:15],
            )

            v1, v2, v3, v4 = points[6], points[9], points[12], points[15]

            kpts = [kpts1, kpts2, kpts3, kpts4]
            for idx, (kw, kh) in enumerate(kpts):
                kpts[idx] = [int(kw * w), int(kh * h)]

            keypoints.extend(kpts)
            visibility.append([v1, v2, v3, v4])

            if item_idx == 0:
                arguments["bboxes"] = [bboxes]
                arguments["keypoints"] = keypoints
            else:
                arguments["bboxes" + str(item_idx - 1)] = [bboxes]
                arguments["keypoints" + str(item_idx - 1)] = keypoints

        im, bboxes, kpts, classes = augment(image, targets, arguments, size, False)

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
                if k[2] != 0:
                    kpts[i, idx, 2] = visibility[i][idx]

        out_name = filename + "_" + str(time_ns())
        out_path = Path(str(out_images_folder / out_name) + ".jpg")
        cv.imwrite(str(out_path.absolute()), im * 255)
        out_labels = Path(str(out_labels_folder / out_name) + ".txt")
        out_labels_file = open(out_labels.absolute(), "w")

        args = []
        for idx, i in enumerate(classes):
            args = []
            args.append(i)

            for b_idx, b in enumerate(bboxes[idx]):
                if (b_idx == 2 and b == 0) or (b_idx == 3 and b == 0):
                    b = 0.1
                args.append(b)

            for k in kpts[idx]:
                args.append(k[0])
                args.append(k[1])
                args.append(int(k[2]))
            line = list(map(str, args))
            out_labels_file.write(" ".join(line) + "\n")
            out_labels_file.flush()

        out_labels_file.close()
