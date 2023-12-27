import os
import sys
import datetime
import click
import tqdm
import numpy as np
import torch
import torchsparse

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchsparse import SparseTensor, nn as spnn
from torchsparse.utils.collate import sparse_collate

from minkunet import MinkUNet

class CustomDataset(Dataset):
    def __init__(self, coords, feats, labels):
        self.coords = coords
        self.feats = feats
        self.labels = labels
        
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.feats[idx], self.labels[idx]
        
@click.command()
@click.argument('current_datetime', type=str, required=True)
@click.argument('loadfrom', type=str, required=True)
@click.argument('iso', type=str, required=True)
@click.argument('learning_rate', type=float, required=True)
@click.argument('epochs', type=int, required=True)
@click.argument('batch_size', type=int, required=True)
@click.argument('checkpoint_path', type=str, required=False, default=None)

def training(current_datetime, loadfrom, iso, learning_rate, epochs, batch_size, checkpoint_path):
    # from click
    ISOTOPE = iso   

    # load data
    coords_train = np.load(loadfrom + ISOTOPE + "_coords_train.npy")
    coords_val = np.load(loadfrom + ISOTOPE + "_coords_val.npy")
    feats_train = np.load(loadfrom + ISOTOPE + "_feats_train.npy")
    feats_val = np.load(loadfrom + ISOTOPE + "_feats_val.npy")
    labels_train = np.load(loadfrom + ISOTOPE + "_labels_train.npy")
    labels_val = np.load(loadfrom + ISOTOPE + "_labels_val.npy")

    # GPU Settings
    device = 'cuda'
    amp_enabled = True

    model = MinkUNet(num_classes=4).to(device)

    # from click
    lr = learning_rate

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler(enabled=amp_enabled)

    # from click
    num_epochs = epochs
    batch_size = batch_size

    # data loaders
    train_set = CustomDataset(coords_train, feats_train, labels_train)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_set = CustomDataset(coords_val, feats_val, labels_val)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # used for loss calculations
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    # loss arrays for plotting
    training_losses = []
    validation_losses = []

    # saving model and model state
    datetime_str = current_datetime
    
    CHECKPOINT_PATH = f"../training/{current_datetime}/checkpoints/"
    if (not os.path.exists(CHECKPOINT_PATH)):
       os.makedirs(CHECKPOINT_PATH)

    MODEL_PATH = f"../training/{datetime_str}/models/"
    LOSS_PATH = f"../training/{datetime_str}/loss_data/"
    
    if (not os.path.exists(MODEL_PATH)) and (not os.path.exists(LOSS_PATH)):
        os.makedirs(MODEL_PATH)
        os.makedirs(LOSS_PATH)
    
    # training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        # batch loading
        for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(train_loader):
            tr_inputs_list = []
            tr_labels_list = []

            # remove empty entries and convert data to sparse tensors
            for i in range(len(batch_coords)):
                # remove empty entries
                current_coords = batch_coords[i]
                current_feats = batch_feats[i]
                current_labels = batch_labels[i]
    
                mask = ~(current_coords == 0).all(axis=1)
    
                current_coords = current_coords[mask]
                current_feats = current_feats[mask]
                current_labels = current_labels[mask]

                # convert to sparse tensor
                current_coords = torch.tensor(current_coords, dtype=torch.int)
                current_feats = torch.tensor(current_feats, dtype=torch.float)
                current_labels = torch.tensor(current_labels, dtype=torch.long)
                
                inputs_sparse = SparseTensor(coords=current_coords, feats=current_feats)
                labels_sparse = SparseTensor(coords=current_coords, feats=current_labels)
                tr_inputs_list.append(inputs_sparse)
                tr_labels_list.append(labels_sparse)
            
            tr_inputs = sparse_collate(tr_inputs_list).to(device=device)
            tr_labels = sparse_collate(tr_labels_list).to(device=device)
            
            with amp.autocast(enabled=amp_enabled):
                tr_outputs = model(tr_inputs)
                tr_labelsloss = tr_labels.feats.squeeze(-1)
                tr_loss = criterion(tr_outputs, tr_labelsloss)
            
            running_loss += tr_loss.item()
        
            optimizer.zero_grad()
            scaler.scale(tr_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        training_losses.append(running_loss / train_steps)
        click.echo()
        click.echo(f"[Epoch {epoch+1}] Running Loss: {running_loss / train_steps}")

    
        checkpoint_filename = os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch_{epoch+1}.pth.tar")

        # validation
        model.eval()
        torchsparse.backends.benchmark = True  # type: ignore
        val_loss = 0.0
        
        with torch.no_grad():
            # batch loading
            for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(val_loader):
                v_inputs_list = []
                v_labels_list = []

                # remove empty entries and convert data to sparse tensors
                for i in range(len(batch_coords)):
                    current_coords = batch_coords[i]
                    current_feats = batch_feats[i]
                    current_labels = batch_labels[i]
        
                    mask = ~(current_coords == 0).all(axis=1)
        
                    current_coords = current_coords[mask]
                    current_feats = current_feats[mask]
                    current_labels = current_labels[mask]

                     # convert to sparse tensor
                    current_coords = torch.tensor(current_coords, dtype=torch.int)
                    current_feats = torch.tensor(current_feats, dtype=torch.float)
                    current_labels = torch.tensor(current_labels, dtype=torch.long)
                    
                    inputs_sparse = SparseTensor(coords=current_coords, feats=current_feats)
                    labels_sparse = SparseTensor(coords=current_coords, feats=current_labels)
                    v_inputs_list.append(inputs_sparse)
                    v_labels_list.append(labels_sparse)
                
                v_inputs = sparse_collate(v_inputs_list).to(device=device)
                v_labels = sparse_collate(v_labels_list).to(device=device)
                
                with amp.autocast(enabled=True):
                    v_outputs = model(v_inputs)
                    v_labelsloss = v_labels.feats.squeeze(-1)
                    v_loss = criterion(v_outputs, v_labelsloss)
                
                val_loss += v_loss.item()
                
        validation_losses.append(val_loss / val_steps)
        click.echo(f"[Epoch {epoch+1}] Validation Loss: {val_loss / val_steps}")
    
    

    model_filename = f"model_epochs{num_epochs}_lr{lr}_train.pth"
    state_filename = f"modelstate_epochs{num_epochs}_lr{lr}_train.pth"
    torch.save(model, MODEL_PATH + model_filename)
    torch.save(model.state_dict(), MODEL_PATH + state_filename)

    # saving loss arrays for plotting
    tr_filename = f"trainloss_{datetime_str}.npy"
    v_filename = f"valloss_{datetime_str}.npy"
    np.save(LOSS_PATH + tr_filename, training_losses)
    np.save(LOSS_PATH + v_filename, validation_losses)
    
    click.echo('Finished Training')

if __name__ == '__main__':
    training()