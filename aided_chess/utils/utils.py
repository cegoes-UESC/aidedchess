from pathlib import Path


def getLabel(image):
    name = image.name.split(".")[0]
    label_name = name + ".txt"

    labels_dir = Path("datasets/chess/labels/train")
    label = labels_dir / label_name
    return label.resolve()


def getBoardKeypoints(image):
    label = getLabel(image)

    label_f = open(label, "r")

    kpts = []

    for label_in in label_f:
        _ = label_in.split(" ")
        cls = _[0]
        if cls == "0":
            kpts = _[5:]
            break
    return kpts


def convertToPx(image, kpts):
    ks = []
    for idx, k in enumerate(kpts):
        if idx in [2, 5, 8, 11]:
            continue
        if idx in [0, 3, 6, 9]:
            k = float(k) * image.shape[1]
        else:
            k = float(k) * image.shape[0]
        ks.append(k)
    return ks
