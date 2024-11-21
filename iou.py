import numpy as np
import pandas as pd

def IoU(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Compute mean IoU (Intersection over Union) of two outputs.

    Args:
        y_test (pd.DataFrame): dataframe where each line is a patch and columns are pixels
        y_pred (pd.DataFrame): dataframe where each line is a patch and columns are pixels. Must have the same index as y_test.

    Returns:
        float: between 0 and 1, mean IoU over all dataframe lines
    """
    
    # Same dtype
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
    
    # Compute intersection and union
    intersection = (y_test & y_pred).sum(axis=1)
    union = (y_test | y_pred).sum(axis=1)
    
    # Compute IoU. IoU is 0 if union is empty
    iou = np.where(union == 0, 0, intersection / union)
    
    # Return mean IoU
    return iou.mean()