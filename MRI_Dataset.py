import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, subject_csv_file, mask1_csv_file, mask2_csv_file):
        # Read the subject data and mask data
        self.data_frame = pd.read_csv(subject_csv_file)
        self.small_mask_data = pd.read_csv(mask1_csv_file)
        self.large_mask_data = pd.read_csv(mask2_csv_file)
        
        # Mask files for cnn1 and cnn2 as per the size clusters
        self.masks_cnn1 = [nib.load(row['Mask File Path']).get_fdata() for _, row in self.small_mask_data.iterrows()]
        self.masks_cnn2 = [nib.load(row['Mask File Path']).get_fdata() for _, row in self.large_mask_data.iterrows()]
        
        # Manual positional encodings
        self.encodings_cnn1 = torch.tensor([eval(row) for row in self.small_mask_data['Mni Coordinates Scaled'].values])
        self.encodings_cnn2 = torch.tensor([eval(row) for row in self.large_mask_data['Mni Coordinates Scaled'].values])
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Load subject data
        img_path = self.data_frame.iloc[idx]['PathToFile']
        image = nib.load(img_path).get_fdata()
        
        cnn1_data = self.get_masked_images(image, self.masks_cnn1, (27,27,27))
        cnn2_data = self.get_masked_images(image, self.masks_cnn2, (56,56,56))
        
        working_memory_score = self.data_frame.iloc[idx]['tfmri_nb_all_beh_c2b_rate_norm']
        working_memory_score = torch.tensor(working_memory_score, dtype=torch.float)
        
        return cnn1_data, cnn2_data, working_memory_score, self.encodings_cnn1, self.encodings_cnn2  # every subject ((174,1,27,27,27), (39,1,56,56,56),  target_score)
    
    def get_masked_images(self, image, mask_cluster, target_shape ):
        masked_images = []
        for mask in mask_cluster:
        
            masked_image = image*mask
            bbox = self.get_bounding_box(mask)
            region_cropped = masked_image[bbox]
            region_processed = self.truncate_or_pad(region_cropped,target_shape) 
            masked_images.append(region_processed)
        
        masked_images = np.array(masked_images, dtype = np.float32)
        masked_images = torch.from_numpy(masked_images).unsqueeze(1)
        
        return masked_images
            
    def get_bounding_box(self, mask):
        """Get the bounding box of the non-zero regions in the mask."""
        non_zero_coords = np.argwhere(mask)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0) + 1  # Add 1 because slice end index is exclusive
        return tuple(slice(min_coord, max_coord) for min_coord, max_coord in zip(min_coords, max_coords))
    
    def truncate_or_pad(self, region, target_shape):
        """Pad the region to match the target shape."""
        region_shape = region.shape
        padded_region = np.zeros(target_shape)
        
        # Calculate the padding for each dimension
        pad_before = [(target_shape[i] - region_shape[i]) // 2 for i in range(3)]
        padded_region[
            pad_before[0]:pad_before[0] + region_shape[0], 
            pad_before[1]:pad_before[1] + region_shape[1], 
            pad_before[2]:pad_before[2] + region_shape[2]] = region

        return padded_region
