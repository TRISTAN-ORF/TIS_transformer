import torch
import pytorch_lightning as pl

import torch.nn.functional as F
import torchmetrics as tm
from performer_pytorch import Performer
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding

import os

from src.transcript_loader import h5pyDataModule, collate_fn

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class TranscriptSeqRiboEmbCustom(pl.LightningModule):
    def __init__(self, x_seq, x_ribo, num_tokens, lr, decay_rate, warmup_steps, max_seq_len, dim, 
                 depth, heads, dim_head, causal, nb_features, feature_redraw_interval,
                 generalized_attention, kernel_fn, reversible, ff_chunks, use_scalenorm,
                 use_rezero, tie_embed, ff_glu, emb_dropout, ff_dropout, attn_dropout,
                 local_attn_heads, local_window_size):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, 
                                 causal=causal, nb_features=nb_features, 
                                 feature_redraw_interval=feature_redraw_interval,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                 reversible=reversible, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm,
                                 use_rezero=use_rezero, ff_glu=ff_glu, ff_dropout=ff_dropout, attn_dropout=attn_dropout, 
                                 local_attn_heads=local_attn_heads, local_window_size=local_window_size)

        self.val_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.val_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.test_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.test_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.ff_1 = torch.nn.Linear(dim,dim*2)
        self.ff_2 = torch.nn.Linear(dim*2,2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(emb_dropout)

        if x_ribo:
            self.ff_emb_1 = torch.nn.Linear(1,dim)
            self.ff_emb_2 = torch.nn.Linear(dim, 6*dim)
            self.ff_emb_3 = torch.nn.Linear(6*dim,dim)
            self.scalar_emb = torch.nn.Sequential(self.ff_emb_1, self.relu, self.ff_emb_2, self.relu, self.ff_emb_3)
            self.ribo_read_emb = torch.nn.Embedding(22, dim)
        
        if x_seq:
            self.nuc_emb = torch.nn.Embedding(num_tokens, dim)

        self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
        self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        
    def on_load_checkpoint(self, checkpoint):
        if 'mlm' in checkpoint.keys() and checkpoint['mlm']:
            state_dict = checkpoint['state_dict']
            for key in ['ff_2.weight', 'ff_2.bias', 'ff_1.weight', 'ff_1.bias']:
                state_dict.pop(key)
            checkpoint['state_dict'] = state_dict
            checkpoint['mlm'] = False
            
        
    def parse_embeddings(self, batch):
        xs = []
        if 'ribo_multi' in batch.keys():
            inp = batch['ribo_multi']
            # relative signal strength per position (21 read lengths)
            xs.append(self.scalar_emb(inp.sum(dim=-1).unsqueeze(-1)))
            x_mask = (inp.sum(dim=-1) != 147)
            # normalized read fraction per position
            x = torch.nan_to_num(torch.div(inp, inp.sum(axis=-1).unsqueeze(-1)))
            # linear combination between read length fraction and read length embedding
            #xs.append(torch.einsum('ikj,jl->ikl', [x, self.ribo_read_emb(torch.arange(1,22).to(x.device))]))
            xs.append(torch.einsum('ikj,jl->ikl', [x, self.ribo_read_emb.weight[1:]]))
            
        if 'ribo_single' in batch.keys():
            inp = batch['ribo_single']
            x_mask = (inp.squeeze(-1) != 7)
            xs.append(self.scalar_emb(inp))
            # linear combination between read length fraction (1) and read length embedding
            xs.append(self.ribo_read_emb(torch.zeros(inp.shape[:2], dtype=torch.int).to(inp.device)))
            
        if 'seq' in batch.keys():
            x_mask = batch['seq'] != 7
            xs.append(self.nuc_emb(batch['seq']))
            
        x_emb = torch.sum(torch.stack(xs), dim=0)
        
        return x_emb, x_mask
            
    def pad_edges(self, y_mask):
        shape = y_mask.shape[1]
        lens = y_mask.sum(dim=1)
        mask = torch.full_like(y_mask, True, dtype=torch.bool)
        mask[:,:301] = False
        for i in range(mask.shape[0]):
            mask[i, int(lens[i]-shape-299):] = False
            
        return mask
    
    def forward(self, batch, y_mask):
        x, x_mask = self.parse_embeddings(batch)
        x += self.pos_emb(x)
        x = self.dropout(x)
        
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb = layer_pos_emb, mask=x_mask)
        
        x = x[torch.logical_and(x_mask, y_mask)]
        x = x.view(-1, self.hparams.dim)
        
        x = F.relu(self.ff_1(x))
        x = self.ff_2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        y_mask = batch['y'] != 7
        y_mask = self.pad_edges(y_mask)
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss, batch_size=y_mask.sum())

        return loss
        
    def validation_step(self, batch, batch_idx):
        y_mask = batch['y'] != 7
        y_mask = self.pad_edges(y_mask)
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.val_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.val_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        
        self.log('val_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('val_prauc', self.val_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('val_rocauc', self.val_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
                
    def test_step(self, batch, batch_idx, ):
        y_mask = batch['y'] != 7
        y_mask = self.pad_edges(y_mask)
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.test_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.test_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)

        self.log('test_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('test_prauc', self.test_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('test_rocauc', self.test_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        
        splits = torch.cumsum(y_mask.sum(dim=1),0, dtype=torch.long).cpu()
        y_hat_grouped = torch.tensor_split(F.softmax(y_hat, dim=1)[:,1], splits)[:-1]
        y_true_grouped = torch.tensor_split(batch['y'][y_mask], splits)[:-1]
        #x_grouped = torch.tensor_split(batch['x'][y_mask], lens)
        
        #return y_hat_grouped, y_true_grouped, x_grouped, batch['x_id']
        return y_hat_grouped, y_true_grouped, batch['x_id']
    
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.test_targets = []
        self.labels = []
        
    def test_step_end(self, results):
        # this out is now the full size of the batch
        self.test_outputs = self.test_outputs + list(results[0])
        self.test_targets = self.test_targets + list(results[1])
        self.labels = self.labels + list(results[2])
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambda1 = lambda epoch: self.hparams.decay_rate
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)

        return [optimizer], [scheduler]
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        

## Inputs
h5py_path = 'GRCh38_v107.hdf5'
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
dim = 30
depth = 6
heads = 6
dim_head = 16
local_attn_heads = 4
local_window_size = 256

## Benchmark
min_seq_len = 601

tis_tr = TranscriptSeqRiboEmbCustom(True, False, 8, 0.001, 0.96, 1500,
        30000, dim, depth, heads, dim_head, False, 80, 1000, True, torch.nn.ReLU(), False, 1, False, False, False, False, 0.1, 0.1, 0.1, local_attn_heads, local_window_size)

tr_loader = h5pyDataModule(h5py_path, exp_path, ribo_path, y_path, x_seq, ribo_offset, 
                            x_id_path, contig_path, val=val, test=test, 
                            max_transcripts_per_batch=400, min_seq_len=min_seq_len, max_seq_len=30000, num_workers=4, cond_fs=None, leaky_frac=0.05, collate_fn=collate_fn)

checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                        filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join('DNA_former_logs', 'benchmark'))
trainer = pl.Trainer(accelerator='gpu', devices=1, reload_dataloaders_every_n_epochs=1, 
                     callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=10)], logger=tb_logger)
trainer.fit(tis_tr, datamodule=tr_loader)
trainer.test(tis_tr, datamodule=tr_loader, ckpt_path='best')