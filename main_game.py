import os
import sys
import time
import warnings
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# from segment_anything import sam_model_registry, SamPredictor
from segmenter.segment import process_image, loading_seg, seg_main, show_points
from feature_matching.generate_points import generate, loading_dino, distance_calculate
from test_game import test_agent, optimize_nodes
from agents import NodeOptimizationEnv, NodeAgent
from utils import calculate_distances, convert_to_edges, calculate_center_points, refine_mask

# Ignore all warnings
warnings.filterwarnings("ignore")

# DATASET = 'Flare'
DATASET = 'ISIC'  # ISIC | FSS | Kvasir | TEM | Flare | MSMT
CATAGORY = '10'
# CATAGORY = '100'
# DATASET = 'FSS' 
# CATAGORY = 'vineSnake' # vineSnake | bandedGecko

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 560

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset', DATASET, CATAGORY)
REFERENCE_IMAGE_DIR = os.path.join(DATA_DIR, 'reference_images')
MASK_DIR = os.path.join(DATA_DIR, 'reference_masks')
IMAGE_DIR = os.path.join(DATA_DIR, 'target_images')  # Path for test images
RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY)
POINT_MASKS_DIR = os.path.join(RESULTS_DIR, 'masks_point')  # 点提示的结果
FINAL_PROMPTS_DIR = os.path.join(RESULTS_DIR, 'final_prompts_image')  # 保存最终提示的目录

# Ensure the results directories exist
os.makedirs(POINT_MASKS_DIR, exist_ok=True)
os.makedirs(FINAL_PROMPTS_DIR, exist_ok=True)  # 创建新目录

# Load models for segmentation and feature generation
def load_models():
    """
    Load the segmentation model and DINO feature extractor.
    """
    try:
        model_seg = loading_seg('vitl', DEVICE)
        model_dino = loading_dino(DEVICE)
        return model_seg, model_dino
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

# Load single agent system
def load_agents():
    """
    Load node agent from model directory.
    """
    try:
        node_env = NodeOptimizationEnv
        node_agent = NodeAgent(node_env)
        
        # 初始化网络
        dummy_features = torch.randn(1, 1)
        node_agent.initialize_networks(dummy_features)
        
        # 从model目录加载模型
        node_model_path = os.path.join(BASE_DIR, 'model', 'node_final_model.pkl')
        node_agent.policy_net.load_state_dict(torch.load(node_model_path))
        
        # 设置为评估模式
        node_agent.policy_net.eval()
        
        # 禁用随机探索(保留一点点随机性)
        node_agent.epsilon = 0.1
        
        return node_agent
        
    except Exception as e:
        print(f"加载智能体模型失败: {e}")
        sys.exit(1)

# Process a single image
def process_single_image(node_agent, model_dino, model_seg, image_name, reference, mask_dir):
    """
    Use multi-agent system to process a single image for segmentation and optimization.

    Parameters:
    - node_agent: Node agent
    - model_dino: DINO feature extraction model
    - model_seg: SAM model
    - image_name: Name of the image to process
    - reference: Reference image for feature comparison
    - mask_dir: Directory containing ground truth masks
    """
    try:
        # Load input image and reference data
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        reference_image = Image.open(os.path.join(REFERENCE_IMAGE_DIR, reference)).resize((IMAGE_SIZE, IMAGE_SIZE))
        
        mask_name = os.path.splitext(reference)[0]
        for ext in ['.png', '.jpg']:
            mask_path = os.path.join(mask_dir, mask_name + ext)
            if os.path.exists(mask_path):
                gt_mask = Image.open(mask_path).resize((IMAGE_SIZE, IMAGE_SIZE))
                break
        else:
            raise FileNotFoundError(f"No mask file found for {reference} with either .png or .jpg extension")

        # Generate features and initial positive/negative prompts
        image_inner = [reference_image, image]
        start_time = time.time()
        features, pos_indices, neg_indices = generate(gt_mask, image_inner, DEVICE, model_dino, IMAGE_SIZE)
        # unique indices
        pos_indices = torch.unique(pos_indices).to(DEVICE)
        neg_indices = torch.unique(neg_indices).to(DEVICE)
        # Remove intersections
        intersection = set(pos_indices.tolist()).intersection(set(neg_indices.tolist()))
        if intersection:
            pos_indices = torch.tensor([x for x in pos_indices.cpu().tolist() if x not in intersection]).cuda()
            neg_indices = torch.tensor([x for x in neg_indices.cpu().tolist() if x not in intersection]).cuda()
        end_time = time.time()
        print(f"Time to generate initial prompts: {end_time - start_time:.4f} seconds")

        if len(pos_indices) != 0 and len(neg_indices) != 0:
            # 只使用node agent进行优化
            opt_pos_indices, opt_neg_indices = optimize_nodes(
                node_agent, pos_indices, neg_indices, features, max_steps=100, device=DEVICE, image_size=IMAGE_SIZE
            )
            
            pos_points = calculate_center_points(opt_pos_indices, IMAGE_SIZE)
            neg_points = calculate_center_points(opt_neg_indices, IMAGE_SIZE)

            # 只使用点提示进行分割
            mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg)
            mask = refine_mask(mask, threshold=0.5)
            mask_img = Image.fromarray(mask)
            mask_img.save(os.path.join(POINT_MASKS_DIR, f"{image_name}"))

            # 可视化优化后的点
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            coords = np.array(pos_points + neg_points)
            labels = np.concatenate([
                np.ones(len(pos_points)),
                np.zeros(len(neg_points))
            ])
            show_points(coords, labels, plt.gca())
            plt.axis('off')
            plt.savefig(os.path.join(FINAL_PROMPTS_DIR, f'{image_name}_final.png'), 
                       bbox_inches='tight', 
                       pad_inches=0)
            plt.close()

        else:
            print(f"Skipping {image_name}: No positive or negative indices found.")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        import traceback
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    # Load models
    model_seg, model_dino = load_models()

    # Load single agent system
    node_agent = load_agents()

    # Get reference image list
    reference_list = os.listdir(REFERENCE_IMAGE_DIR)
    if not reference_list:
        print("No reference images found.")
        sys.exit(1)

    # Use the first reference image
    reference = reference_list[0]

    # Process all images in the test directory
    img_list = os.listdir(IMAGE_DIR)
    for img_name in tqdm(img_list, desc="Processing images"):
        process_single_image(node_agent, model_dino, model_seg, img_name, reference, MASK_DIR)

    print("Processing complete!")
