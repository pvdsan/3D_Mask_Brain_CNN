from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os
import csv
from datetime import datetime

def initialize_logger(directory):
    """Initialize the CSV logger."""
    os.makedirs(directory, exist_ok=True)
    log_file_path = os.path.join(directory, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_training_log.csv")
    with open(log_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Train R2', 'Validation Loss', 'Validation R2'])
    return log_file_path

def log_metrics(log_file_path, metrics):
    """Log metrics to the CSV file."""
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

def split_dataset(dataset):
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset