import cv2 as cv
from pathlib import Path
import albumentations as A

transform = A.Compose(
    [
        A.RandomCrop(width=1024, height=1024),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo"),
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

path = Path("datasets/split")

images_path = Path(path / "images")
labels_path = Path(path / "labels")

train_images_path = Path(images_path / "train")

imagesj = train_images_path.glob("*.jpg")
imagesJ = train_images_path.glob("*.JPG")

images = list(imagesj) + list(imagesJ)

for im in images:
    filename = im.name.split(".")[0]
    label_name = filename + ".txt"

    labels = Path(labels_path / "train" / label_name)
    label_file = open(labels.absolute(), "r")
    image = cv.imread(str(im.absolute()), cv.IMREAD_COLOR)
    h, w = image.shape[0:2]

    cv.namedWindow("image", cv.WINDOW_GUI_NORMAL)
    bs, ks = [], []
    for l in label_file:
        labels_content = l.split(" ")
        labels_content = labels_content[:-1]
        _class = int(labels_content[0])
        # objects[0].append(_class)
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

        ks.extend(kpts)

    trans = transform(image=image, bboxes=bs, keypoints=ks)
    trans_image = trans["image"]

    for kk in trans["keypoints"]:
        cv.circle(trans_image, kk, 7, (0, 255, 0), 50, 10)
    cv.imshow("image", trans_image)
    cv.waitKey(0)
    print(labels)


# READ dataset root path (from the base, before images and labels)
# Read all images from images path
# Get label for the current image
# Read label file line by line
# Split by space
# Get 0 from the object' class
# Get 1:4, inclusive, for the BB
# Get, from 5, by pairs, the keypoint location
# Unnormalize kpt positions
# Apply transformation
# Renormalize kpt positions
# Save new image, with new name
# Write each object to a new file, line by line, save it
# Repeat for each image in dataset (train and val)


def albument(it=1):
    pass
