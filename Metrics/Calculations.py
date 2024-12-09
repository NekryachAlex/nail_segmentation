import torch

import torch.nn.functional as F
import numpy as np  
from skimage.filters import threshold_otsu
def threshold(sample_g):
    """

Applies Otsu's thresholding to a grayscale image.


This function computes the Otsu threshold for the input image and returns a binary

image where pixels above the threshold are set to True and those below are set to False.


Parameters:

    sample_g (numpy.ndarray): A 2D array representing a grayscale image.


Returns:

    numpy.ndarray: A binary image (2D array) where pixels above the Otsu threshold are True.

"""
    thresh = threshold_otsu(sample_g)
    sample_ot  = sample_g > thresh
    return sample_ot
def iou(target, prediction):
    """

    Computes the Intersection over Union (IoU) score between the target and predicted binary images.


    This function thresholds the predicted image using Otsu's method and calculates the IoU score

    by finding the intersection and union of the target and predicted binary images.


    Parameters:

        target (numpy.ndarray): A binary image (2D array) representing the ground truth.

        prediction (numpy.ndarray): A predicted binary image (2D array).


    Returns:

        float: The IoU score, which is the ratio of the intersection to the union of the target and prediction.

    """
    prediction = threshold(prediction)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
def check_accuracy(loader, model, device="cuda"):
    """

    Checks the accuracy of a model using the Intersection over Union (IoU) metric.


    This function evaluates the model on the provided data loader, computes the IoU score for each batch,

    and returns the average IoU score across all batches.


    Parameters:

        loader (torch.utils.data.DataLoader): A data loader that provides batches of input data and target labels.

        model (torch.nn.Module): The model to be evaluated.

        device (str): The device to run the model on (default is "cuda").


    Returns:

        float: The average IoU score across all batches in the loader.

    """
    iou_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            threshold = threshold_otsu(((preds).to(device))[0][0].cpu().detach().numpy())
            preds = (preds > threshold).float()
            
            iou_score += (preds * y).sum() / (
                (preds + y).sum() + 1e-8 - (preds * y).sum()
            )

    return iou_score/len(loader)