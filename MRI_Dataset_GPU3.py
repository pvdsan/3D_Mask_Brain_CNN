import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib

class MRIDataset(Dataset):
    def __init__(self, subject_csv_file, mask1_csv_file, mask2_csv_file, device='cuda'):
        self.device = device
        
        # Load subject data
        self.data_frame = pd.read_csv(subject_csv_file)
        
        # Preload working memory scores and transfer to the device
        self.working_memory_scores = torch.tensor(self.data_frame['tfmri_nb_all_beh_c2b_rate_norm'].values, dtype=torch.float32).to(device)
        
        # Load masks and positional encodings, preloaded to the device
        self.masks_cnn1, self.encodings_cnn1 = self.load_masks(mask1_csv_file, device)
        self.masks_cnn2, self.encodings_cnn2 = self.load_masks(mask2_csv_file, device)
    
    def load_masks(self, mask_file, device):
        mask_data = pd.read_csv(mask_file)
        masks = []
        encodings = []
        for _, row in mask_data.iterrows():
            mask = torch.tensor(nib.load(row['Mask File Path']).get_fdata(), dtype=torch.float32)
            masks.append(mask.to(device))
            encodings.append(torch.tensor(eval(row['Mni_Scaled']), dtype=torch.float32).to(device))
        return masks, torch.stack(encodings)

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['PathToFile']
        image = torch.tensor(nib.load(img_path).get_fdata(), dtype=torch.float32).to(self.device)
        
        cnn1_data = self.apply_masks_and_pad(image, self.masks_cnn1, (25, 25, 25))
        cnn2_data = self.apply_masks_and_pad(image, self.masks_cnn2, (41, 41, 41))
        
        # Retrieve preloaded working memory score for the current index
        working_memory_score = self.working_memory_scores[idx]
        
        return cnn1_data, cnn2_data, working_memory_score, self.encodings_cnn1, self.encodings_cnn2
    
    def apply_masks_and_pad(self, image, masks, target_shape):
        masked_images = []
        for mask in masks:
            masked_image = image * mask
            padded_image = self.truncate_or_pad(masked_image, target_shape)
            masked_images.append(padded_image)
        return torch.stack(masked_images).unsqueeze(1)  # Add channel dimension

    def truncate_or_pad(self, region, target_shape):
        """Crop the region to match the target shape."""
        current_shape = region.shape
        min_dim = [min(ts, cs) for ts, cs in zip(target_shape, current_shape)]
        
        # Calculate start indices for cropping to center the cropped area
        start_indices = [(cs - ts) // 2 for ts, cs in zip(min_dim, current_shape)]
        
        # Perform cropping
        cropped_region = region[
            start_indices[0]:start_indices[0] + min_dim[0],
            start_indices[1]:start_indices[1] + min_dim[1],
            start_indices[2]:start_indices[2] + min_dim[2]
        ]
        
        return cropped_region