# This script contains deprecated functions from the transcript-tranformer
# package used to perform the benchmark

import torch
import h5py
import numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
import torchmetrics as tm
from performer_pytorch import Performer
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding

import os

from transcript_transformer.transcript_loader import DataLoader
from h5max import load_sparse

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def local_shuffle(data, lens=None):
    if lens is None:
        lens = np.array([ts[0].shape[0] for ts in data])
    elif type(lens) == list:
        lens = np.array(lens)
    # get split idxs representing spans of 400
    splits = np.arange(1,max(lens),400)
    # get idxs
    idxs = np.arange(len(lens))

    shuffled_idxs = []
    ### Local shuffle 
    for l, u in zip(splits, np.hstack((splits[1:],[999999]))):
        # mask between lower and upper
        mask = np.logical_and(l < lens, lens <= u)
        # get idxs within mask
        shuffled = idxs[mask]
        # randomly shuffle idxs
        np.random.shuffle(shuffled)
        # add to idxs all
        shuffled_idxs.append(shuffled)
    shuffled_idxs = np.hstack(shuffled_idxs)
    data = data[shuffled_idxs]
    lens = lens[shuffled_idxs]

    return data, lens

def bucket(data, lens, max_seq_len, max_transcripts_per_batch, min_seq_len=0):
    # split idx sites l
    l = []
    # idx pos
    num_samples = 0
    # filter invalid lens
    mask = np.logical_and(np.array(lens)<=max_seq_len, np.array(lens)>=min_seq_len)
    data = data[mask]
    lens = lens[mask]
    ### bucket batching
    while len(data) > num_samples:
        # get lens of leftover transcripts
        lens_set = lens[num_samples:]
        # calculate memory based on number and length of samples (+2 for transcript start/stop token)
        mask = (np.maximum.accumulate(lens_set)+2) * (np.arange(len(lens_set))+1) >= max_seq_len
        # obtain position where mem > max_memory
        mask_idx = np.where(mask)[0]
        # get idx to split
        if len(mask_idx) > 0 and (mask_idx[0] > 0):
            # max amount of transcripts per batch
            samples_d = min(mask_idx[0],max_transcripts_per_batch)
            num_samples += samples_d
            l.append(num_samples)       
        else:
            break
    # [:-1] not possible when trying to test all data
    return np.split(data, l)#[:-1]

class TranscriptSeqRiboEmbBench(pl.LightningModule):
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
            
    def bench_mask(self, y_mask, atg_mask):
        shape = y_mask.shape[1]
        lens = y_mask.sum(dim=1)
        mask = torch.full_like(y_mask, True, dtype=torch.bool)
        mask[:,:301] = False
        for i in range(mask.shape[0]):
            mask[i, int(lens[i]-shape-299):] = False
        
        mask = torch.logical_and(mask, atg_mask)
        
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
        y_mask = self.bench_mask(y_mask, batch['atg_mask'])
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss, batch_size=y_mask.sum())

        return loss
        
    def validation_step(self, batch, batch_idx):
        y_mask = batch['y'] != 7
        y_mask = self.bench_mask(y_mask, batch['atg_mask'])
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.val_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.val_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        
        self.log('val_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('val_prauc', self.val_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('val_rocauc', self.val_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
                
    def test_step(self, batch, batch_idx, ):
        y_mask = batch['y'] != 7
        y_mask = self.bench_mask(y_mask, batch['atg_mask'])
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.test_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.test_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)

        self.log('test_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('test_prauc', self.test_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('test_rocauc', self.test_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
    
    def predict_step(self, batch, batch_idx):
        y_mask = batch['y'] != 7
        y_mask = self.bench_mask(y_mask, batch['atg_mask'])
        y_true = batch['y'][y_mask].view(-1)
        
        y_hat = self(batch, y_mask)
        y_hat = F.softmax(y_hat, dim=1)[:,1]
        
        return y_hat.cpu().numpy(), y_true.cpu().numpy()
        
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
        
class h5pyDatasetBatchesBench(torch.utils.data.Dataset):
    def __init__(self, fh, ribo_paths, y_path, x_id_path, x_seq, ribo_offset, idx_adj, batches):
        super().__init__()
        self.fh = fh
        self.ribo_paths = ribo_paths
        self.y_path = y_path
        self.x_id_path = x_id_path
        self.x_seq = x_seq
        self.ribo_offset = ribo_offset
        self.idx_adj = idx_adj
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        # Transformation is performed when a sample is requested
        x_ids = []
        x_atgs = []
        xs = []
        ys = []
        for idx_conc in self.batches[index]:
            idx = idx_conc % self.idx_adj
            # get transcript IDs 
            x_ids.append(self.fh[self.x_id_path][idx])
            x_dict = {}
            # get seq data
            if self.x_seq:
                x_dict['seq'] = self.fh['seq'][idx]
                x_atgs.append(self.fh['atg_mask'][idx])
            # get ribo data
            if len(self.ribo_paths) > 0:
                # obtain data set and adjuster
                data_path = list(self.ribo_paths.keys())[idx_conc//self.idx_adj]
                x = load_sparse([idx], self.fh[data_path])[0].toarray()
                if self.ribo_offset:
                    col_names = np.array(self.fh[data_path]['col_names']).astype(str)
                    for col_key, shift in self.ribo_paths[data_path].items():
                        mask = col_names == col_key
                        if (shift != 0) and (shift > 0):
                            x[:shift, mask] = 0
                            x[shift:, mask] = x[:-shift, mask]
                        elif (shift != 0) and (shift < 0):
                            x[-shift:, mask] = 0
                            x[:-shift, mask] = x[shift:, mask]
                    x = x.sum(axis=1)
                    x = x/np.maximum(x.max(), 1)
                    x_dict['ribo_single'] = self.fh['seq'][idx]
                else:
                    x_dict['ribo_multi'] = x/np.maximum(np.sum(x, axis=1).max(), 1)
                
            xs.append(x_dict)
            ys.append(self.fh[self.y_path][idx])
            
        return [x_ids, xs, ys, x_atgs]
    
def collate_fn(batch):
    # In which cases is this true?
    if type(batch[0][0]) == list:
        batch = batch[0]
    lens = np.array([len(s) for s in batch[2]])
    max_len = max(lens)
    
    y_b = torch.LongTensor(np.array([np.pad(y,(1,1+l), constant_values=7) for y, l in zip(batch[2], max_len - lens)]))
    
    atg_b = torch.BoolTensor(np.array([np.pad(atg,(1,1+l), constant_values=False) for atg, l in zip(batch[3], max_len - lens)]))
    
    x_dict = {}
    for k in batch[1][0].keys():
        # if the entries are multidimensional: positions x read lengths (reads)
        if len(batch[1][0][k].shape) > 1:
            x_exp = [np.pad(x[k],((1,1),(0,0)), constant_values=((5,6),(0,0))) for x in batch[1]]
            x_exp = [np.pad(x,((0,l),(0,0)), constant_values=((0,7),(0,0))) for x, l in zip(x_exp, max_len - lens)]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float))
        
        # if the entries are single dimensional and float: positions (reads)
        elif batch[1][0][k].dtype == float:
            x_exp = [np.concatenate(([5], x[k], [6], [7]*l)) for x, l in zip(batch[1], max_len - lens)]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float)).unsqueeze(-1)
        
        # if the entries are single dimensional and string: positions (nucleotides)
        else:
            x_dict[k] = torch.LongTensor(np.array([np.concatenate(([5], x[k], [6], [7]*l)) for x, l in zip(batch[1], max_len - lens)], dtype=int))
            
    x_dict.update({'x_id':batch[0], 'y':y_b, 'atg_mask':atg_b})
    
    return x_dict

class h5pyDataModule(pl.LightningDataModule):
    def __init__(self, h5py_path, exp_path, ribo_paths, y_path, x_seq=False, ribo_offset=False, x_id_path='id', contig_path='contig', 
                 val=[], test=[], max_transcripts_per_batch=500, min_seq_len=0, max_seq_len=30000, num_workers=5, 
                 cond_fs=None, leaky_frac=0.05, collate_fn=collate_fn):
        super().__init__()
        self.ribo_paths = ribo_paths
        self.ribo_offset = ribo_offset
        if ribo_offset:
            assert len(list(ribo_paths.values())) > 0, f"No offset values present in ribo_paths input, check the function docstring"
        # number of datasets
        self.n_data = max(len(self.ribo_paths), 1)
        self.x_seq = x_seq
        self.y_path = y_path
        self.x_id_path = x_id_path
        self.h5py_path = h5py_path
        self.exp_path = exp_path
        self.contig_path = contig_path
        self.val_contigs = np.ravel([val])
        self.test_contigs = np.ravel([test])
        self.max_transcripts_per_batch = max_transcripts_per_batch
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_workers = num_workers
        self.cond_fs = cond_fs
        self.leaky_frac = leaky_frac
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.fh = h5py.File(self.h5py_path,'r')[self.exp_path]
        self.cond_mask = np.full(len(self.fh[self.x_id_path]), True)
        
        if self.cond_fs is not None:
            for key, cond in self.cond_fs.items():
                self.cond_mask = np.logical_and(self.cond_mask, cond(np.array(self.fh[key])))
            if self.leaky_frac > 0:
                leaky_abs = int(np.sum(self.cond_mask)*self.leaky_frac)
                leaky_idxs = np.random.choice(np.where(~self.cond_mask)[0], leaky_abs)
                self.cond_mask[leaky_idxs] = True
        
        contigs = np.unique(self.fh[self.contig_path]).astype(str)
        for ct in self.val_contigs:
            contigs = np.delete(contigs, np.where(contigs == str(ct)))
        for ct in self.test_contigs:
            contigs = np.delete(contigs, np.where(contigs == str(ct)))
        self.train_contigs = contigs
        print(f"Training contigs: {self.train_contigs}")
        print(f"Validation contigs: {self.val_contigs}")
        print(f"Test contigs: {self.test_contigs}")
        
        if stage == "fit" or stage is None:
            contig_mask = np.isin(self.fh[self.contig_path], np.array(self.train_contigs).astype('S'))
            mask = np.logical_and(self.cond_mask, contig_mask)
            self.tr_idx, self.tr_len, self.tr_idx_adj = self.prepare_sets(mask)
            print(f"Training set transcripts: {len(self.tr_idx)}")
            mask = np.isin(self.fh[self.contig_path], self.val_contigs.astype('S'))
            self.val_idx, self.val_len, self.val_idx_adj = self.prepare_sets(mask)
            print(f"Validation set transcripts: {len(self.val_idx)}")
        if stage == "test" or stage is None:
            mask = np.isin(self.fh[self.contig_path], self.test_contigs.astype('S'))
            self.te_idx, self.te_len, self.te_idx_adj = self.prepare_sets(mask)
            print(f"Test set transcripts: {len(self.te_idx)}")
            
    def prepare_sets(self, mask):
        # idx mask
        idx_temp = np.where(mask)[0]
        # set idx shift value if multiple riboseq datasets are present
        set_idx_adj = np.max(idx_temp)+1
        set_idx = np.ravel([np.where(mask)[0]+(set_idx_adj*i) for i in np.arange(self.n_data)])
        set_len = list(self.fh['tr_len'][mask])*self.n_data
        
        return set_idx, set_len, set_idx_adj

    def train_dataloader(self):
        batches = bucket(*local_shuffle(self.tr_idx, self.tr_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatchesBench(self.fh, self.ribo_paths, self.y_path, self.x_id_path, self.x_seq, self.ribo_offset, self.tr_idx_adj, batches), 
                          collate_fn=collate_fn, num_workers=self.num_workers, shuffle=True, batch_size=1)

    def val_dataloader(self):
        batches = bucket(*local_shuffle(self.val_idx, self.val_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatchesBench(self.fh, self.ribo_paths, self.y_path, self.x_id_path, self.x_seq, self.ribo_offset, self.val_idx_adj, batches), 
                         collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)

    def test_dataloader(self):
        batches = bucket(*local_shuffle(self.te_idx, self.te_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatchesBench(self.fh, self.ribo_paths, self.y_path, self.x_id_path, self.x_seq, self.ribo_offset, self.te_idx_adj, batches),
                          collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)