import os

import pyheif
from PIL import Image


def convert_from_heif():
    # Get list of HEIF and HEIC files in directory
    directory = "/home/almir-dev/Documents/aided-chess/datasets/new/"
    files = [
        f for f in os.listdir(directory) if f.endswith(".HEIC") or f.endswith(".HEIF")
    ]

    if not os.path.exists(os.path.join(directory, "output")):
        os.makedirs(os.path.join(directory, "output"))

    # Convert each file to JPEG
    for filename in files:
        heif_file = pyheif.read(os.path.join(directory, filename))
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        image.save(
            os.path.join(directory, "output", os.path.splitext(filename)[0] + ".jpg")
        )
