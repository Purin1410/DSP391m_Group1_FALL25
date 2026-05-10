import argparse
import os
import zipfile

import torch
from pytorch_lightning import seed_everything
from sconf import Config
from tqdm import tqdm

from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER


def main(config_path: str, ckp_path: str, output_zip: str = "result.zip"):
    config = Config(config_path)
    seed_everything(config.seed_everything, workers=True)

    dm = CROHMEDatamodule(config=config)
    dm.setup("test")
    test_dataloader = dm.test_dataloader()

    vocab_info = dm.vocab.get_info()

    model = LitCoMER.load_from_checkpoint(ckp_path, vocab_info=vocab_info)
    model.eval()
    model.cuda()

    exprate_recorder = model.exprate_recorder

    with zipfile.ZipFile(output_zip, "w") as zip_f:
        with torch.inference_mode():
            for batch in tqdm(test_dataloader, desc="Testing"):
                batch = batch.to("cuda")
                
                # Inference
                hyps = model.approximate_joint_search(batch.imgs, batch.mask)
                exprate_recorder([h.seq for h in hyps], batch.indices)
                
                img_bases = batch.img_bases
                preds = [vocab_info.words.indices2label(h.seq) for h in hyps]
                
                # Write to zip incrementally
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    exprate = exprate_recorder.compute()
    print(f"Validation ExpRate: {exprate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckp", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="result.zip", help="Output zip file")
    args = parser.parse_args()
    
    main(args.config, args.ckp, args.output)
