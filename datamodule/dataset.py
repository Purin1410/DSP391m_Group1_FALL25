import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from .transforms import ScaleAugmentation, ScaleToLimitRange

class CROHMEDataset(Dataset):
    def __init__(self, 
                dataset,
                is_train: bool,
                scale_aug: bool,
                k_min: float,
                k_max: float,
                w_lo: float,
                w_hi: float,
                h_lo: float,
                h_hi: float,
                lazy_load: bool = False) -> None:
        super().__init__()
        self.dataset = dataset
        self.lazy_load = lazy_load

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(lo = k_min,
                                                hi = k_max)
                              )

        trans_list += [
            ScaleToLimitRange(w_lo=w_lo, 
                              w_hi=w_hi, 
                              h_lo=h_lo, 
                              h_hi=h_hi),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        fnames, imgs, captions = self.dataset[idx]
        
        processed_imgs = []
        if self.lazy_load:
            for item in imgs:
                if isinstance(item, tuple):
                    continue
                try:
                    im = Image.open(item).convert("L")
                    im_np = np.array(im)
                except Exception as e:
                    print(f"[WARN] Could not open image {item}: {e}")
                    continue
                processed_imgs.append(self.transform(im_np))
        else:
            for im in imgs:
                if not isinstance(im, np.ndarray):
                    im = np.array(im)
                processed_imgs.append(self.transform(im))

        return fnames, processed_imgs, captions

    def __len__(self):
        return len(self.dataset)