import torch
import os
from PIL import Image



class NailsDataset(torch.utils.data.Dataset):
  """

    A class to represent a dataset of images and masks.


    This class loads images and their corresponding masks from specified directories,

    applies transformations, and returns them as tensors.


    Attributes:

        img_names (list): A list of paths to the images.

        mask_names (list): A list of paths to the masks.

        trans1 (callable): Transformation applied to images and masks.

        trans2 (callable): Additional transformation applied to images and masks.


    Parameters:

        path_to_imgs (str): Path to the directory containing images.

        path_to_masks (str): Path to the directory containing masks.

        trans1 (callable): Transformation applied to images and masks.

        trans2 (callable): Additional transformation applied to images and masks.

    """
  def __init__(self, path_to_imgs, path_to_masks, trans1, trans2):
    """

        Initializes the NailsDataset.


        Loads the filenames of images and masks from the specified directories and stores

        the transformations for later use.


        Parameters:

            path_to_imgs (str): Path to the directory containing images.

            path_to_masks (str): Path to the directory containing masks.

            trans1 (callable): Transformation applied to images and masks.

            trans2 (callable): Additional transformation applied to images and masks.

    """
    self.img_names = sorted([path_to_imgs + filename for filename in os.listdir(path_to_imgs)])
    self.mask_names = sorted([path_to_masks + filename for filename in os.listdir(path_to_masks)])
    self.trans1 = trans1
    self.trans2 = trans2
  
  def __len__(self):
    """

    Returns the number of images in the dataset.


    Returns:

        int: The number of images in the dataset.

    """
    return len(self.img_names)
  
  def __getitem__(self, index):
    """

    Retrieves an image and its corresponding mask by the given index.


    Loads the image and the corresponding mask, applies transformations, and

    returns them as tensors.


    Parameters:

        index (int): The index of the image and mask.


    Returns:

        tuple: A tuple containing two tensors (image, mask).

            - img_tensor (torch.Tensor): The image tensor.

            - mask_tensor (torch.Tensor): The mask tensor.

    """
    image_path = self.img_names[index]
    mask_path = self.mask_names[index]
    img = Image.open(image_path)
    mask = Image.open(mask_path)
    img_trans = self.trans1(img)
    mask_trans = self.trans1(mask)
    img_np = img_trans.numpy()
    mask_np = mask_trans.numpy()         
    augmented = self.trans2(image=img_np, mask=mask_np)
    img_new = augmented['image']
    mask_new = augmented['mask']
    img_tensor = torch.from_numpy(img_new).float()
    mask_tensor = torch.from_numpy(mask_new).float()
    return img_tensor, mask_tensor
  
