import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
from utils import calculate_center_points
from config import *

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

from feature_matching.generate_points import generate, loading_dino, distance_calculate
from feature_matching.generate_box import generate_boxes
from segmenter.segment import show_points

# Function to draw points on an image
def draw_points_on_image(image, points, color):
    """
    Draws a list of points on an image.

    Parameters:
    image (np.array): The image on which to draw the points.
    points (list of tuples): The list of points to draw.
    color (tuple): The color of the points in BGR format.
    """
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=5, color=color, thickness=-1)
    return image

# Function to save a PyTorch tensor to a text file
def save_tensor_to_txt(tensor, filename):
    """
    Saves a PyTorch tensor to a text file.
    Args:
        tensor (torch.Tensor): The tensor to save.
        filename (str): The path to the text file.
    """
    array = tensor.cpu().numpy()
    np.savetxt(filename, array, fmt='%d')
    print(f"Tensor saved to {filename}")

def main():
    # Hyperparameter setting
    dataset_name = os.path.join('ISIC','10')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = 560 # Must be a multiple of 14

    # Loading the DINO model
    model_dino = loading_dino(device)

    # Define directories
    image_prompt_dir = os.path.join('dataset', dataset_name, 'reference_images')
    mask_path = os.path.join('dataset', dataset_name, 'reference_masks')
    image_dir = os.path.join('dataset', dataset_name, 'target_images')
    save_dir = os.path.join('results', dataset_name, 'initial_prompts')
    initial_image_dir = os.path.join('results', dataset_name, 'initial_images')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(initial_image_dir, exist_ok=True)

    # Get the reference image for prompting
    reference_list = os.listdir(image_prompt_dir)
    reference = reference_list[0]
    
    # Load the reference image and ground truth mask
    image_prompt = Image.open(os.path.join(image_prompt_dir, reference)).resize((image_size, image_size))
    reference_name = os.path.splitext(reference)[0]
    mask_files = os.listdir(mask_path)
    mask_file = next(f for f in mask_files if f.startswith(reference_name))
    gt_mask = Image.open(os.path.join(mask_path, mask_file)).resize((image_size, image_size))
    
    imglist = os.listdir(image_dir)
    dice_list = []

    for name in tqdm(imglist):
        image_path = os.path.join(image_dir, name)
        image = Image.open(image_path).resize((image_size, image_size))

        image_inner = [image_prompt, image]
        features, initial_indices_pos, initial_indices_neg = generate(gt_mask, image_inner, device, model_dino, image_size)

        if len(initial_indices_pos) != 0 and len(initial_indices_neg) != 0:
            print(len(initial_indices_pos), len(initial_indices_neg))
            # 确保索引唯一
            initial_indices_pos = torch.unique(initial_indices_pos).to(device)
            initial_indices_neg = torch.unique(initial_indices_neg).to(device)
            print(len(initial_indices_pos), len(initial_indices_neg))
            # Remove intersections
            intersection = set(initial_indices_pos.tolist()).intersection(set(initial_indices_neg.tolist()))
            if intersection:
                initial_indices_pos = torch.tensor([x for x in initial_indices_pos.cpu().tolist() if x not in intersection]).cuda()
                initial_indices_neg = torch.tensor([x for x in initial_indices_neg.cpu().tolist() if x not in intersection]).cuda()
            print(len(initial_indices_pos), len(initial_indices_neg))
            torch.save(features, os.path.join(save_dir, name + '_features.pt'))
            torch.save(initial_indices_pos, os.path.join(save_dir, name + '_initial_indices_pos.pt'))
            torch.save(initial_indices_neg, os.path.join(save_dir, name + '_initial_indices_neg.pt'))
            
            # 绘制结果
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            # 绘制提示点
            pos_points = calculate_center_points(initial_indices_pos, 560)
            neg_points = calculate_center_points(initial_indices_neg, 560)
            coords = np.array(pos_points + neg_points)
            labels = np.concatenate([
                np.ones(len(initial_indices_pos)),  
                np.zeros(len(initial_indices_neg))
            ])
            show_points(coords, labels, plt.gca())
            
            # 保存结果
            plt.axis('off')
            plt.savefig(os.path.join(initial_image_dir, f'{name}_initial.png'), 
                       bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"No positive or negative indices found for {name}")

if __name__ == "__main__":
    main()
