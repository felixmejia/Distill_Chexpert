import torch
import os
from tqdm import tqdm
from utils import calculate_metrics

from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(epoch, model, optimizer, best_val_auc, history, filepath):
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_auc': best_val_auc, 'history': history}
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer):
    if not os.path.exists(filepath):
        return 0, 0.0, {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_auc = checkpoint['best_val_auc']
    history = checkpoint['history']
    print(f"✅ Checkpoint cargado. Reanudando desde la época {start_epoch}.")
    return start_epoch, best_val_auc, history

def train_one_epoch(model, dataloader, criterion, optimizer, device, description, loss_fn_kd=None):
    """
    Ejecuta una época de entrenamiento y DEVUELVE la pérdida promedio.
    """
    model.train()
    
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=description)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        if loss_fn_kd:
            loss = loss_fn_kd(inputs, outputs, labels)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device, class_names, description):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    return epoch_loss, metrics