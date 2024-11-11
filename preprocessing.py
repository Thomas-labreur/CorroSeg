import numpy as np
import glob
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler


def preprocess(in_folder: str, z_threshold: float = 3) -> Tuple[List[str], np.ndarray]:
    """Fill nan values, removes outliers and scales data. 

    Args:
        in_folder (str): the folder containing unprocessed data
        z_threshold (float): for the outliers processing, a pixel value outside of the mean +/- z_threshold*std interval
            is considered as an outlier pixel. Each patch with an outlier pixel will be removed.
    
    Returns:
        List[str]: list of outier patches names 
        np.ndarray: array with shape (N, 36, 36, 1) where N is the number of inlier data, containing all inlier data. 
    """

    # List of .npy files 
    file_paths = glob.glob(in_folder + '/*.npy') 

    # Initialize list of inlier patches
    inlier_patch_names = []
    inlier_patch_arrays = []
    
    # Firstly, stack every patch in an list of arrays
    patch_names = []
    patch_arrays = []
    for file_path in file_paths:
        
        # Add name to the name list
        name = file_path.split('\\')[-1][:-4]
        patch_names.append(name)
    
        # Add image to the array list by filling nan values with zeros
        img = np.load(file_path)
        img = np.nan_to_num(img, nan=0)
        patch_arrays.append(img)
    

    # Compute mean and std of pixel values
    mean = np.mean(patch_arrays)
    std = np.std(patch_arrays)

    # Only keep inlier patches, which means that all pixel have a z-score lesser than the given threshold.
    inlier_patch_names = []
    inlier_patch_arrays = []
    for name, img in zip(patch_names, patch_arrays) :
        
        # Compute z_scores
        pixel_z_scores = np.abs(img - mean) / std
        if ( pixel_z_scores < z_threshold).all():
            
            # Add name to the list
            inlier_patch_names.append(name)
            
            # Scale the image and add it to the list 
            scaler = RobustScaler()
            img_scaled = scaler.fit_transform(img.reshape(-1,1)).reshape(36,36)
            inlier_patch_arrays.append(img_scaled)
    
    # Turn list into array and reshape
    inlier_patch_array = np.array(inlier_patch_arrays).reshape(-1,36,36,1)

    # Display results and return it
    print(f"{len(file_paths)-len(inlier_patch_names)} files have been removed, {len(inlier_patch_names)} still remain.")
    return inlier_patch_names, inlier_patch_array