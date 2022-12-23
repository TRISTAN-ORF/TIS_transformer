import os
import torch
import pytorch_lightning as pl

from benchmark_scripts import h5pyDataModule, collate_fn, TranscriptSeqRiboEmbBench
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

## Inputs
h5py_path = '../../data/GRCh38_v107_ATG_mask.h5'
exp_path = 'transcript'
ribo_path = []
y_path = 'tis'
x_seq = 'seq'
ribo_offset = None
x_id_path = 'id'
contig_path = 'contig'
val = ['2','14']
test = ['1','7','13','19']

## Hyperparameters
dim = 48
depth = 8
heads = 8
dim_head = 16
local_attn_heads = 5
local_window_size = 256

## Benchmark
min_seq_len = 601

tis_tr = TranscriptSeqRiboEmbBench(True, False, 8, 0.001, 0.96, 1500, 30000, dim, depth, heads, dim_head, False, 80, 1000, True,
                                    torch.nn.ReLU(), False, 1, False, False, False, False, 0.1, 0.1, 0.1, local_attn_heads, local_window_size)

tr_loader = h5pyDataModule(h5py_path, exp_path, ribo_path, y_path, x_seq, ribo_offset, x_id_path, contig_path, val=val, test=test, 
                            max_transcripts_per_batch=400, min_seq_len=min_seq_len, max_seq_len=30000, num_workers=4, cond_fs=None, leaky_frac=0.05, collate_fn=collate_fn)

checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                        filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join('DNA_former_logs', 'benchmark_large'))
trainer = pl.Trainer(accelerator='gpu', devices=1, reload_dataloaders_every_n_epochs=1, 
                     callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=10)], logger=tb_logger)
trainer.fit(tis_tr, datamodule=tr_loader)
trainer.test(tis_tr, datamodule=tr_loader, ckpt_path='best')