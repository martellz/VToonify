import os
import io
import numpy as np
import onnxruntime as ort
import pooch
from pathlib import Path
from enum import Enum
from typing import List, Union, Dict, Tuple, Type
from cv2 import (
    BORDER_DEFAULT,
    COLOR_BGR2GRAY,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    THRESH_BINARY,
    cvtColor,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
    threshold
)
from PIL import Image
from PIL.Image import Image as PILImage

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


class SimpleSession:
    def __init__(self, model_name: str, inner_session: ort.InferenceSession):
        self.model_name = model_name
        self.inner_session = inner_session

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage) -> List[PILImage]:
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]


def new_session(model_name: str = "u2net") -> SimpleSession:
    session_class: Type[SimpleSession]
    md5 = "60024c5c889badc19c04ad937298a77b"
    url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    session_class = SimpleSession

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
        )
        session_class = SimpleSession
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
        session_class = SimpleSession
    # elif model_name == "u2net_cloth_seg":
    #     md5 = "2434d1f3cb744e0e49386c906e5a08bb"
    #     url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx"
    #     session_class = ClothSession
    elif model_name == "silueta":
        md5 = "55e59e0d8062d2f5d013f4725ee84782"
        url = (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"
        )
        session_class = SimpleSession

    u2net_home = os.getenv(
        "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    )

    fname = "u2net.onnx"
    full_path = Path(u2net_home).expanduser() / fname

    # Download and cache a single file locally.
    pooch.retrieve(
        url,
        f"md5:{md5}",
        fname=fname,
        path=Path(u2net_home).expanduser(),
        progressbar=True,
    )

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(
        "u2net",
        ort.InferenceSession(
            str(full_path),
            providers=['CPUExecutionProvider'],
            sess_options=sess_opts,
        ),
    )

session = new_session()

def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary
    return mask


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    only_mask: bool = False,
    post_process_mask: bool = True,
) -> Union[bytes, PILImage, np.ndarray]:
    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    masks = session.predict(img)
    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask

        else:
            cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()


def get_background_mask(image: PILImage) -> PILImage:
  image = remove(image)
  gray = cvtColor(np.asarray(image), COLOR_BGR2GRAY)
  ret, mask = threshold(gray, 0.5, 255, THRESH_BINARY)
  mask = 255 - mask
  return Image.fromarray(mask)


def rembg_dir(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if(not output_dir.exists()):
        output_dir.mkdir(parents=True)

    for i in os.listdir(image_dir):
        img = Image.open(image_dir / i)
        out = get_background_mask(img)
        out.save(output_dir / i)

if __name__ == '__main__':
    import sys
    if(len(sys.argv) < 3):
        print('usage: python {} image_dir output_dir'.format(sys.argv[0]))
        exit(0)
    rembg_dir(sys.argv[1], sys.argv[2])
