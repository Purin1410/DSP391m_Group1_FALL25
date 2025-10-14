import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset

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
                h_hi: float) -> None:
        super().__init__()
        self.dataset = dataset

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
        fname, img, caption = self.dataset[idx]

        img = [self.transform(im) for im in img]

        return fname, img, caption

    def __len__(self):
        return len(self.dataset)