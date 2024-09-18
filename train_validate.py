import torch
from sklearn.metrics import r2_score
import numpy as np

def train(model, device, train_loader, optimizer):
    total_loss = 0
    all_targets = []
    all_outputs = []

    for x1, x2, y, e1, e2 in train_loader:
        x1, x2, y, e1, e2 = x1.to(device), x2.to(device), y.to(device), e1.to(device), e2.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2, e1, e2)
        loss = torch.nn.functional.mse_loss(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_targets.append(y.cpu().detach().numpy())  # Collect target data
        all_outputs.append(outputs.cpu().detach().numpy())  # Collect output data
            
    avg_loss = total_loss / len(train_loader)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    overall_r2 = r2_score(all_targets, all_outputs)  # Compute R^2 score across all batches
    return avg_loss, overall_r2

def evaluate(model, device, val_loader, phase='Validation'):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for x1, x2, y, e1, e2 in val_loader:
            x1, x2, y, e1, e2 = x1.to(device), x2.to(device), y.to(device), e1.to(device), e2.to(device)
            outputs = model(x1, x2, e1, e2)
            loss = torch.nn.functional.mse_loss(outputs, y.unsqueeze(1))
            total_loss += loss.item()
            all_targets.append(y.cpu().detach().numpy())  # Collect target data
            all_outputs.append(outputs.cpu().detach().numpy())  # Collect output data
            
    avg_loss = total_loss / len(val_loader)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    overall_r2 = r2_score(all_targets, all_outputs)  # Compute R^2 score across all batches
    return avg_loss, overall_r2
