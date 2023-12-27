import os
import sys
import datetime
import numpy as np
import click

import h5py
import tqdm
import torch
import torchsparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchsparse import SparseTensor, nn as spnn
from torchsparse.utils.collate import sparse_collate
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support

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

def evaluate(current_datetime, loadfrom, iso, learning_rate, epochs, batch_size):
    datetime_str = current_datetime
    click.echo(f"Received datetime: {datetime_str}")

    ISOTOPE = iso
    lr = learning_rate
    num_epochs = epochs
    batch_size = batch_size
    
    coords_test = np.load(loadfrom + ISOTOPE + "_coords_test.npy")
    feats_test = np.load(loadfrom + ISOTOPE + "_feats_test.npy")
    labels_test = np.load(loadfrom + ISOTOPE + "_labels_test.npy")

    runninglen = 0

    test_set = CustomDataset(coords_test, feats_test, labels_test)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = torch.load(f"../training/{current_datetime}/models/model_epochs{num_epochs}_lr{lr}_train.pth")
    loadpath = f"../training/{current_datetime}/models/modelstate_epochs{num_epochs}_lr{lr}_train.pth"

    device = 'cuda'
    
    criterion = nn.CrossEntropyLoss()
    
    model.load_state_dict(torch.load(loadpath))
    model.eval()

    with torch.no_grad():
        
        all_preds = []
        all_labels = []
        all_preds = np.array(all_preds)
        
        total_correct = 0
        for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(test_loader):
            t_inputs_list = []
            t_labels_list = []
    
            for i in range(len(batch_coords)):
                current_coords = batch_coords[i]
                current_feats = batch_feats[i]
                current_labels = batch_labels[i]
                
                mask = ~(current_coords == 0).all(axis=1)
    
                # Apply the mask to the array
                current_coords = current_coords[mask]
                current_feats = current_feats[mask]
                current_labels = current_labels[mask]
                all_labels = np.concatenate((all_labels, current_labels.reshape(-1)))
                
                current_coords = torch.tensor(current_coords, dtype=torch.int)
                current_feats = torch.tensor(current_feats, dtype=torch.float)
                current_labels = torch.tensor(current_labels, dtype=torch.long)
                
                t_inputs_sparse = SparseTensor(coords=current_coords, feats=current_feats)
                t_labels_sparse = SparseTensor(coords=current_coords, feats=current_labels)
                t_inputs_list.append(t_inputs_sparse)
                t_labels_list.append(t_labels_sparse)
    
                runninglen += len(current_coords)
            t_inputs = sparse_collate(t_inputs_list).to(device=device)
            t_labels = sparse_collate(t_labels_list).to(device=device)
    
            
            n_correct = 0
            
            with amp.autocast(enabled=True):
                outputs = model(t_inputs)
                
                labelsloss = t_labels.feats.squeeze(-1)
                loss = criterion(outputs, labelsloss)
                _, predicted = torch.max(outputs, 1)
    
                
                all_preds = np.concatenate((all_preds, predicted.cpu().numpy()))
                n_correct += (predicted == labelsloss).sum().item()
                total_correct += n_correct
        
        acc = 100.0 * total_correct / (len(all_preds))
        click.echo(f'Accuracy of the model: {acc:.3g} %')

    EVAL_PATH = f"../training/{datetime_str}/eval/"
    
    if (not os.path.exists(EVAL_PATH)):
        os.makedirs(EVAL_PATH)
        
    labels_np = np.array(all_labels.astype(int))
    preds_np = np.array(all_preds.astype(int))
    
    labels_filename = "labels.npy"
    preds_filename = "preds.npy"
    np.save(EVAL_PATH + labels_filename, labels_np)
    np.save(EVAL_PATH + preds_filename, preds_np)
    
    click.echo('Finished Evaluation')
    accuracy = accuracy_score(labels_np, preds_np)
    
    average_method = 'macro'  
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average=average_method)
    
    # Create a dictionary to store all metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

    # Writing metrics to a text file
    with open(EVAL_PATH + 'model_performance_metrics.txt', 'w') as file:
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                value_str = ', '.join([f'{v:.3f}' for v in value])
                file.write(f'{key}: [{value_str}]\n')
            else:
                file.write(f'{key}: {value:.3f}\n')

    click.echo('Finished Evaluation and Saved Metrics')

if __name__ == '__main__':
    evaluate()