from pathlib import Path
import os

PATH = Path("datasets/albumented/")

images, labels = Path(PATH / "images/train"), Path(PATH / "labels/train")

images, labels = list(images.glob("*")), list(labels.glob("*"))

print(len(images), len(labels))

labels_n = list(map(lambda x: x.name.split(".")[0], labels))

for i in images:
    if i.name.split(".")[0] not in labels_n:
        os.remove(i.absolute())
        print(i.name)
