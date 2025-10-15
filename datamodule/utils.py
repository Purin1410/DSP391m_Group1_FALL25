from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
from PIL import Image
from torch import FloatTensor, LongTensor
from pathlib import Path

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
# - lazy_load=False: List[(str, Image.Image, List[str])]
# - lazy_load=True:  List[(str, Tuple[int,int], List[str])]
Data = List[Tuple[str, object, List[str]]]


# ---------------------------------------------------------------------
# Batch grouping
# ---------------------------------------------------------------------
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = 32e4,
    maxlen: int = 200,
    maxImagesize: int = 32e4,
):
    """
    Return data as:
    [
      ([fname1,...], [feature1,...], [label1,...]),
      ...
    ]
    """
    fname_batch, feature_batch, label_batch = [], [], []
    fname_total, feature_total, label_total = [], [], []

    biggest_image_size = 0
    data.sort(key=lambda x: x[1].size[0] * x[1].size[1] if hasattr(x[1], "size") else x[1][0] * x[1][1])

    i = 0
    for fname, fea, lab in data:
        if hasattr(fea, "size"):
            w, h = fea.size
            fea_arr = np.array(fea)
        else:
            w, h = fea
            fea_arr = fea  # metadata mode

        size = w * h
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)

        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(f"image: {fname} size: {w} x {h} =  bigger than {maxImagesize}, ignore")
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch, feature_batch, label_batch = [], [], []

            fname_batch.append(fname)
            feature_batch.append(fea_arr)
            label_batch.append(lab)
            i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


# ---------------------------------------------------------------------
# Helper: resolve image paths
# ---------------------------------------------------------------------
def _resolve_image_path(
    img_dir: Path,
    stem: str,
    exts: Sequence[str] = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
) -> Optional[Path]:
    p = img_dir / stem
    if p.exists():
        return p

    for ext in exts:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------
# Extract data
# ---------------------------------------------------------------------
def extract_data(
    root_dir: str,
    split: str,              
    caption_name: str = "caption.txt",
    img_subdir: str = "img",
    convert_mode: str = "L",
    lazy_load: bool = False
) -> Data:
    """
    Read from:
      root_dir/
        split/
          img/
          caption.txt
    caption.txt format: "<image_stem> token1 token2 ..."
    Return: 
      If lazy_load=False: [(img_name, Image, tokens)]
      If lazy_load=True:  [(img_name, (w,h), tokens)]
    """
    split_dir = Path(root_dir) / split
    cap_path = split_dir / caption_name
    img_dir = split_dir / img_subdir

    assert cap_path.exists(), f"Not found: {cap_path}"
    assert img_dir.exists(), f"Not found: {img_dir}"

    with cap_path.open("r", encoding="utf-8") as f:
        captions = f.readlines()

    data: Data = []
    missing = 0

    for line in captions:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        img_stem = parts[0]
        tokens = parts[1:]

        stem_no_ext = Path(img_stem).stem
        img_path = _resolve_image_path(img_dir, stem_no_ext) or _resolve_image_path(img_dir, img_stem)
        if img_path is None:
            missing += 1
            print(f"[WARN] Missing image for '{img_stem}' in {img_dir}")
            continue

        if lazy_load:
            with Image.open(img_path) as im:
                size = im.size
            data.append((str(img_path), size, tokens))
        else:
            with Image.open(img_path) as im:
                if convert_mode is not None:
                    im = im.convert(convert_mode)
                im = im.copy()
            data.append((img_path.stem, im, tokens))

    print(f"Extract data from dir: {split_dir}, size: {len(data)} (missing: {missing})")
    return data


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------
def build_validation_dataset(
    archive: str,
    folder: str,
    batch_size: int,
    batch_Imagesize: float,
    maxlen: int,
    maxImagesize: float,
    lazy_load: bool = False,
):
    if folder != "all":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
    else:
        data = []
        for folder_name in ["2014", "2016", "2019"]:
            data += extract_data(root_dir=archive, split=folder_name, lazy_load=lazy_load)

    return data_iterator(
        data=data,
        batch_size=batch_size,
        batch_Imagesize=batch_Imagesize,
        maxlen=maxlen,
        maxImagesize=maxImagesize,
    )


def build_train_dataset(
    archive: str,
    folder: str,
    batch_size: int,
    batch_Imagesize: float,
    maxlen: int,
    maxImagesize: float,
    lazy_load: bool = False,
):
    if folder == "train":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
        data = data_iterator(
            data=data,
            batch_size=batch_size,
            batch_Imagesize=batch_Imagesize,
            maxlen=maxlen,
            maxImagesize=maxImagesize,
        )
    return data


# ---------------------------------------------------------------------
# Batch class
# ---------------------------------------------------------------------
@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )
