import pytorch_lightning as pl
import torchmetrics as tm
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class TISRover(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()

        self.val_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.val_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.test_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.test_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.emb = torch.nn.Embedding.from_pretrained(torch.eye(4), freeze=True)
        
        self.act = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv1d(4, 50, 9) # 203 -> 195
        self.conv_2 = torch.nn.Conv1d(50, 62, 7) # 
        self.conv_3 = torch.nn.Conv1d(62, 75, 7)
        self.conv_4 = torch.nn.Conv1d(75, 87, 7)
        self.conv_5 = torch.nn.Conv1d(87, 100, 7)
        self.max_pool_02 = torch.nn.MaxPool1d(2)
        self.dropout_02 = torch.nn.Dropout(0.2)
        
        self.feat_ext = torch.nn.Sequential(self.conv_1, self.act, self.dropout_02,
                                            self.conv_2, self.act, self.max_pool_02, self.dropout_02,
                                            self.conv_3, self.act, self.max_pool_02, self.dropout_02,
                                            self.conv_4, self.act, self.max_pool_02, self.dropout_02,
                                            self.conv_5, self.act, self.max_pool_02, self.dropout_02)
        
        self.dropout_05 = torch.nn.Dropout(0.5)
        self.dense_1 = torch.nn.Linear(600,128)
        self.dense_2 = torch.nn.Linear(128,2)
        
        self.feat_learn = torch.nn.Sequential(self.dense_1, self.act, self.dropout_05, 
                                              self.dense_2)

    def forward(self, x):
        x = self.emb(x).permute(0,2,1)
        x = self.feat_ext(x)
        x = x.view(-1, 600)
        x = self.feat_learn(x)
        
        return x

    def training_step(self,batch, index):
        x, y_true = batch
        y_hat = self(x)
        
        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss)

        return loss
        
    def validation_step(self, batch, index):
        x, y_true = batch
        y_hat = self(x)
        
        self.val_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.val_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        
        self.log('val_loss', F.cross_entropy(y_hat, y_true))
        self.log('val_prauc', self.val_prauc, on_step=False, on_epoch=True)
        self.log('val_rocauc', self.val_rocauc, on_step=False, on_epoch=True)
                
    def test_step(self, batch, index):
        x, y_true = batch
        y_hat = self(x)
        
        self.test_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.test_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)

        self.log('test_loss', F.cross_entropy(y_hat, y_true))
        self.log('test_prauc', self.test_prauc, on_step=False, on_epoch=True)
        self.log('test_rocauc', self.test_rocauc, on_step=False, on_epoch=True)
    
    def predict_step(self, batch, index, dataloader_idx=0):
        x, y_true = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)[:,1]
        
        return y_hat.cpu().numpy(), y_true.cpu().numpy()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        return optimizer
    
class h5pyDataset(torch.utils.data.Dataset):
    def __init__(self, fh, ):
        super().__init__()
        self.fh = fh
        
    def __len__(self):
        return len(self.fh['sample'])
    
    def __getitem__(self, index):
        # Transformation is performed when a sample is requested
        x = self.fh['sample'][index].astype(int)
        y = self.fh['label'][index].astype(int)
        
        return [x, y]
        

fh = h5py.File('../../data/TISRover_TITER_samples.h5', 'r')
dataset = h5pyDataset(fh)

main_atg_idxs = np.array(fh['atg_idx'])
alt_atg_idxs = np.array(h5py.File('../../data/DeepGSR_samples.h5', 'r')['atg_idx'])
atg_idx_mask = np.isin(main_atg_idxs, alt_atg_idxs)

val_contigs = [b'2', b'14']
test_contigs = [b'1', b'7', b'13', b'19']

tr_contig_mask = ~np.isin(fh['contig'], val_contigs + test_contigs)
val_contig_mask = np.isin(fh['contig'], val_contigs)
te_contig_mask = np.isin(fh['contig'], test_contigs)

tr_mask = np.logical_and(tr_contig_mask, atg_idx_mask)
val_mask = np.logical_and(val_contig_mask, atg_idx_mask)
te_mask = np.logical_and(te_contig_mask, atg_idx_mask)

print(f"Training set samples: {tr_mask.sum()}")
print(f"validation set samples: {val_mask.sum()}")
print(f"Testing set samples: {te_mask.sum()}")

idxs = np.arange(len(fh['contig']))

batch_size = 256
epochs = 100

train_dataloader = DataLoader(dataset, batch_size, sampler=RandomSampler(idxs[tr_mask]), num_workers=4)
val_dataloader = DataLoader(dataset, batch_size, sampler=SequentialSampler(idxs[val_mask]), num_workers=4)
test_dataloader = DataLoader(dataset, batch_size, sampler=SequentialSampler(idxs[te_mask]), num_workers=4)

trainer = pl.Trainer(accelerator='gpu', devices=1, auto_scale_batch_size=False, 
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])
model = TISRover(lr=0.001)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')