import cv2 as cv
from pathlib import Path
import albumentations as A


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

        if item_idx != 0:
            arguments["keypoints" + str(item_idx - 1)] = kpts
            arguments["bboxes" + str(item_idx - 1)] = bs
        else:
            arguments["keypoints"] = kpts
            arguments["bboxes"] = bs
    print(arguments)

    transform = A.Compose(
        [
            # A.RandomCrop(width=1024, height=1024),
            A.Normalize(),
            A.Perspective(),
            A.ColorJitter(),
            A.MotionBlur(),
            A.ChannelShuffle(),
            # A.GlassBlur(),
            A.GaussianBlur(),
            A.Rotate(border_mode=cv.BORDER_TRANSPARENT),
            A.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo"),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        additional_targets=targets,
    )

    trans = transform(image=image, **arguments)
    trans_image = trans["image"]

    for i in range(size):
        key = "keypoints" + (str(i - 1) if i != 0 else "")
        for kk in trans[key]:
            kk = list(map(int, kk))
            cv.circle(trans_image, kk, 7, (0, 255, 0), 20, 10)
    cv.imshow("image", trans_image)
    cv.waitKey(0)


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
