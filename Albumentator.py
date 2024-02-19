import albumentations as A
from torch.utils.data import Dataset

transform = A.Compose(
    [
        A.RandomCrop(width=330, height=330),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo"),
    keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=True, label_fields=["1", "2", "3", "4"]
    ),
)


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
