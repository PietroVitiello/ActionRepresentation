from turtle import shape
from typing import List, Tuple
import numpy as np
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms as T

def getImages(frame0_dir: Path, frame1_dir: Path) -> Tuple[Image.Image, Image.Image]:
    f0 = Image.open(frame0_dir)
    f1 = Image.open(frame1_dir)
    return f0, f1

def im2arr(frame0: Image.Image, frame1: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    frame0 = np.asarray(frame0, dtype=np.int16)
    frame1 = np.asarray(frame1, dtype=np.int16)
    return frame0, frame1

def arr2Im(*images: np.ndarray) -> List[Image.Image]:
    images_arr = []
    for im in images:
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        images_arr.append(im)
    return images_arr

def resizeImages(frame0: Image.Image, frame1: Image.Image, new_side: int) -> Tuple[Image.Image, Image.Image]:
    resize_transform = T.Resize((new_side, new_side))
    frame0 = resize_transform(frame0)
    frame1 = resize_transform(frame1)
    return frame0, frame1

def turnGreyscale(frame0: Image.Image, frame1: Image.Image) -> Tuple[Image.Image, Image.Image]:
    frame0 = ImageOps.grayscale(frame0)
    frame1 = ImageOps.grayscale(frame1)
    return frame0, frame1

def generate_motionImage(
        frame0: np.ndarray,
        frame1: np.ndarray,
        scaling_factor: float = 5.0,
        threshold: int = 0
    ) -> np.ndarray:

    assert(threshold >= 0)
    motion_image = frame1 - frame0 #make sure that the type is not uint8
    motion_image *= scaling_factor #scale differences
    motion_image[motion_image<threshold] = 0 #remove small differences and negative values
    motion_image[motion_image>255] = 255 #make sure scaling did not create pixels >255
    return motion_image


def get_motionImage(frame0: Image.Image, frame1: Image.Image, greyscale: bool = False, resized_side: int = None) -> Image.Image :
    if greyscale:
        turnGreyscale(frame0, frame1)
    if resized_side is not None:
        resizeImages(frame0, frame1, resized_side)
    frame0, frame1 = im2arr(frame0, frame1)
    motion_image = generate_motionImage(frame0, frame1)
    motion_image = arr2Im(motion_image)[0]
    return motion_image
    



