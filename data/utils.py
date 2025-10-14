from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
from PIL import Image
from torch import FloatTensor, LongTensor
from pathlib import Path


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Data = List[Tuple[str, Image.Image, List[str]]]


def data_iterator(
    data: List[Tuple[str, Image.Image, List[str]]],
    batch_size: int,
    batch_Imagesize: int = 32e4,
    maxlen: int = 200,
    maxImagesize: int = 32e4,
    ):
    """
    return data as follow: 
    [
    ([fname1,fname2,fname3,...], [feature1,feature2,feature3,...], [label1,label2,label3,...]), 
    ([fname9,fname10,fname11,...], [feature9,feature10,feature11,...], [label9,label10,label11,...]), 
    ...]
    """
    fname_batch = []
    feature_batch = []
    label_batch = []

    fname_total = []
    feature_total = []
    label_total = []
    
    biggest_image_size = 0
    
    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
            fname_batch.append(fname)
            feature_batch.append(fea)
            label_batch.append(lab)
            i += 1
                
    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))

# def extract_data(archive: ZipFile, dir_name: str, free_memory = False) -> Data: # return data as follow: [(fname1, fea1, lab1), (fname2, fea2, lab2), ...]
#     """Extract all data need for a dataset from zip archive

#     Args:
#         archive (ZipFile):
#         dir_name (str): dir name in archive zip (eg: train, test_2014......)

#     Returns:
#         Data: list of tuple of image and formula
#     """
#     with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
#         captions = f.readlines()
#     data = []
#     for line in captions:
#         tmp = line.decode().strip().split()
#         img_name = tmp[0]
#         formula = tmp[1:]
#         with archive.open(f"data/{dir_name}/img/{img_name}.bmp", "r") as f:
#         # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
#             img = Image.open(f).copy()
#         data.append((img_name, img, formula))
#         if free_memory:
#             del img

#     print(f"Extract data from: {dir_name}, with data size: {len(data)}")

#     return data

def _resolve_image_path(img_dir: Path, stem: str,
                        exts: Sequence[str] = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> Optional[Path]:
    p = img_dir / stem
    if p.exists():
        return p

    for ext in exts:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def extract_data(
    root_dir: str,
    split: str,                     # 'train' hoặc '2014'/'2016'/'2019'
    caption_name: str = "caption.txt",
    img_subdir: str = "img",
    convert_mode: str = "L",        # "L" cho grayscale (giống .bmp gốc)
    free_memory: bool = False
) -> Data:
    """
    Read in folder:
      root_dir/
        split/
          img/
          caption.txt

    caption.txt format: "<image_stem> token1 token2 ..."

    Return: List[(img_name, PIL.Image, token_list)]
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
        parts = line.decode().strip().split()
        if len(parts) == 0:
            continue
        img_stem = parts[0]          # không nhất thiết đã có .ext
        tokens = parts[1:]

        stem_no_ext = Path(img_stem).stem
        img_path = _resolve_image_path(img_dir, stem_no_ext)
        if img_path is None:
            img_path = _resolve_image_path(img_dir, img_stem)

        if img_path is None:
            missing += 1
            print(f"[WARN] Missing image for '{img_stem}' in {img_dir}")
            continue

        with Image.open(img_path) as im:
            if convert_mode is not None:
                im = im.convert(convert_mode)
            im = im.copy()  # materialize in memory
        data.append((img_path.stem, im, tokens))
        if free_memory:
            del im

    print(f"Extract data from dir: {split_dir}, size: {len(data)} (missing: {missing})")
    return data

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



def build_validation_dataset(archive, 
                             folder: str,
                             batch_size: int,
                             batch_Imagesize,
                             maxlen, 
                             maxImagesize,
                             free_memory = False,
                             ):
    if folder != 'all':
        data = extract_data(root_dir = archive, split=folder, free_memory = free_memory)
    else:
        data = []
        for folder in ['2014', '2016', '2019']:
            data += extract_data(root_dir = archive, split=folder, free_memory = free_memory)
    return data_iterator(data = data, 
                         batch_size = batch_size,
                         batch_Imagesize = batch_Imagesize,
                         maxlen = maxlen,
                         maxImagesize = maxImagesize,
                         ) 

def build_train_dataset(archive, 
                        folder: str, 
                        batch_size: int, 
                        batch_Imagesize,
                        maxlen, 
                        maxImagesize,
                        free_memory: bool = False):
    if folder == 'train':
        data = extract_data(root_dir = archive, split=folder, free_memory = free_memory)
        data = data_iterator(data = data, 
                        batch_size = batch_size,
                        batch_Imagesize = batch_Imagesize,
                        maxlen = maxlen,
                        maxImagesize = maxImagesize,
                        )
    return data