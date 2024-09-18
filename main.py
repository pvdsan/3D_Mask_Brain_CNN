import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import KFold
from networks.dual_cnn_regressor import DualCNNRegressor
from torch.utils.data import DataLoader, Subset
from MRI_Dataset_GPU3 import MRIDataset
from train_validate import train, evaluate
from tqdm import tqdm
from common_utils import split_dataset, initialize_logger, log_metrics
import os
import torch.multiprocessing as mp

def load_checkpoint(fold = 0, directory="models"):
    """Load a checkpoint file from a specified directory."""
    if directory=="models":
        fold_directory = os.path.join(directory, f"fold_{fold}")
        filepath = os.path.join(fold_directory, "model_checkpoint.pth")
    else:
        filepath = os.path.join(directory, "model_checkpoint.pth")
        
    if os.path.isfile(filepath):
        return torch.load(filepath)
    return None

def save_checkpoint(state, fold, directory="models", filename = "model_checkpoint.pth" ):
    """Save the current state of the model, optimizer, and other parameters."""
    
    if directory=="models":
        fold_directory = os.path.join(directory, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)
        filepath = os.path.join(fold_directory, filename) 
    else:
        filepath = os.path.join(directory,filename)  
    torch.save(state, filepath)

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    batch_size  = 4
    num_epochs = 100
    metric_directory = "training_logs"
    log_file_path = initialize_logger(metric_directory)
    print('Main Function started')
    print('------------------------------------------------------------------------')
    device = torch.device('cuda')
    print(device)
    print("Initializing DataSet")
    dataset = MRIDataset(subject_csv_file='Subject_Data/SMRI_Dataset_Earliest.csv', mask1_csv_file='3D_Mask_Data/Small_Masks.csv', mask2_csv_file='3D_Mask_Data/Large_Masks.csv')
    print(" DataSet Initialized")
    train_dataset, test_dataset = split_dataset(dataset)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize global best model tracking
    global_best_val_loss = float('inf')
    global_best_model_state = None
    
    global_best_checkpoint = load_checkpoint(directory="models/best")
    if global_best_checkpoint:
        global_best_model_state = global_best_checkpoint['model_state_dict']
        global_best_val_loss    = global_best_checkpoint['val_loss']
        global_best_model_epoch = global_best_checkpoint['epoch']
        global_best_model_fold  = global_best_checkpoint['fold'] 
        print(f'The best model so far is from epoch:{global_best_model_epoch} of fold:{global_best_model_fold}')
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Starting Fold {fold}')
        dual_regressor = DualCNNRegressor().to(device)
        dual_regressor = nn.DataParallel(dual_regressor)
            
        optimizer = Adam(dual_regressor.parameters(), lr=0.0001)
        start_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None
        if os.path.isfile(f"models/fold_{fold}/Completed.pth"):
            print(f'Fold{fold} training is complete, moving over')
            continue
        
        checkpoint = load_checkpoint(fold, "models")
        if checkpoint:
            dual_regressor.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_model_state = dual_regressor.state_dict()
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming training from epoch {start_epoch}")
            

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=8)
        
        for epoch in tqdm(range(start_epoch, num_epochs+1), desc="Epochs", position=0):
            train_loss, train_r2 = train(dual_regressor, device, train_loader, optimizer)
            val_loss, val_r2 = evaluate(dual_regressor, device, val_loader)
            log_metrics(log_file_path, [fold, epoch, train_loss, train_r2, val_loss, val_r2])
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving a new best model for fold{fold} in epoch {epoch}")
                best_model_state = dual_regressor.state_dict()
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, fold)
                
            # Update global best model
            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                global_best_model_state = dual_regressor.state_dict()
                print(f"Saving a new best overall model for fold{fold} in epoch {epoch}")
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': global_best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'fold'    : fold
                }, fold, directory='models/best')
                
            if epoch==num_epochs:
                print(f'Training for fold{fold} is complete')     
                save_checkpoint({
                'epoch': epoch,
                }, fold, filename = 'Completed.pth')
        print('----------------------------------------------------------------------------------------')

    print("Training complete. Evaluating on test data...")