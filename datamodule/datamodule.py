from typing import Optional
import pytorch_lightning as pl
from .dataset import CROHMEDataset
from torch.utils.data.dataloader import DataLoader
from .utils import (build_train_dataset, 
                    build_validation_dataset)
from .vocab import Vocab
from .utils import Batch
import torch

class CROHMEDatamodule(pl.LightningDataModule):
    shared_vocab: Optional[Vocab] = None  # class-level cache
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        data_config = self.config.data
        self.zipfile_path               = data_config.zipfile_path
        self.test_year                  = data_config.test_year
        self.train_batch_size           = data_config.train_batch_size
        self.eval_batch_size            = data_config.eval_batch_size
        self.num_workers                = data_config.num_workers
        self.scale_aug                  = data_config.scale_aug
        self.gpu_max_memory             = data_config.gpu_max_memory
        self.maxlen                     = self.config.model.max_len
        self.lazy_load                  = data_config.lazy_load
        self.k_min                      = data_config.k_min
        self.k_max                      = data_config.k_max
        self.w_lo                       = data_config.w_lo
        self.w_hi                       = data_config.w_hi
        self.h_lo                       = data_config.h_lo
        self.h_hi                       = data_config.h_hi
        self.pin_memory                 = data_config.pin_memory
        self.persistent_workers         = data_config.persistent_workers
        if CROHMEDatamodule.shared_vocab is None:
            CROHMEDatamodule.shared_vocab = Vocab(dict_path=data_config.dictionary_txt)
        self.vocab = CROHMEDatamodule.shared_vocab
        
        print(f"Load data from: {self.zipfile_path}")
    
    def collate_fn(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        fnames = batch[0]
        images_x = batch[1]
        seqs_y = [self.vocab.words2indices(x) for x in batch[2]]
        
        # images_x = [torch.as_tensor(s, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(s) else s for s in images_x]

        heights_x = [s.size(1) for s in images_x]
        widths_x = [s.size(2) for s in images_x]

        n_samples = len(heights_x)
        max_height_x = max(heights_x)
        max_width_x = max(widths_x)

        x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
        x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
        for idx, s_x in enumerate(images_x):
            x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
            x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

        # return fnames, x, x_mask, seqs_y
        return Batch(fnames, x, x_mask, seqs_y)

        

    def setup(self, stage: Optional[str] = None) -> None:
        # with ZipFile(self.zipfile_path) as archive:
        if stage == "fit" or stage is None:
            # Train_dataset
            self.train_dataset = CROHMEDataset(
                dataset = build_train_dataset(archive       = self.zipfile_path, 
                                            folder          = 'train', 
                                            batch_size      = self.train_batch_size,
                                            batch_Imagesize = self.gpu_max_memory,
                                            maxlen          = self.maxlen, 
                                            maxImagesize    = self.gpu_max_memory,
                                            lazy_load       = self.lazy_load,
                                            ),
                is_train    = True,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
            )
            # Val_dataset
            self.val_dataset = CROHMEDataset(
                dataset = build_validation_dataset(archive  = self.zipfile_path, 
                                            folder          = self.test_year, 
                                            batch_size      = self.eval_batch_size,
                                            batch_Imagesize = self.gpu_max_memory,
                                            maxlen          = self.maxlen, 
                                            maxImagesize    = self.gpu_max_memory,
                                            lazy_load     = self.lazy_load,
                                            ),
                is_train    = False,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                dataset = build_validation_dataset(archive  = self.zipfile_path, 
                                            folder          = self.test_year, 
                                            batch_size      = self.eval_batch_size,
                                            batch_Imagesize = self.gpu_max_memory,
                                            maxlen          = self.maxlen, 
                                            maxImagesize    = self.gpu_max_memory,
                                            lazy_load     = self.lazy_load,
                                            ),
                is_train    = False,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset             = self.train_dataset,
            shuffle             = True,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset             = self.val_dataset,
            shuffle             = False,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset             = self.test_dataset,
            shuffle             = False,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
        )