import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from medpy.metric.binary import dc
from utils import refine_mask

# Configuration
CONFIG = {
    'image_size': 560,
    'dataset': 'VOC',  # 'FSS' | 'ISIC' | 'Kvasir' | 'TEM'
    'category': 'boat',
    'methods': ['point']  # 只保留point方法
}

# Directory setup
BASE_DIR = os.path.dirname(__file__)
DIRS = {
    'results': os.path.join(BASE_DIR, 'results', CONFIG['dataset'], CONFIG['category']),
    'ground_truth': os.path.join(BASE_DIR, 'dataset', CONFIG['dataset'], CONFIG['category'], 'target_masks'),
    'output': os.path.join(BASE_DIR, 'evaluation', CONFIG['dataset'], CONFIG['category'], datetime.now().strftime("%Y%m%d_%H%M"))
}

# Create required directories
os.makedirs(DIRS['output'], exist_ok=True)

def process_image(image_path):
    """处理单个图像的通用函数"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (CONFIG['image_size'], CONFIG['image_size']))
    img = (img > 127).astype(np.uint8) if np.max(img) > 1 else img
    return refine_mask(img, threshold=0.5)

def calculate_iou(y_true, y_pred):
    """计算IOU（交并比）"""
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / (np.sum(union) + 1e-6)

def evaluate_single_method(pred_dir, ground_truth_dir):
    """评估单个方法的分割结果"""
    result_files = sorted(glob(os.path.join(pred_dir, "*.jpg")))
    if not result_files:
        raise FileNotFoundError(f"No mask images found in {pred_dir}")
    
    results = []
    for result_file in tqdm(result_files, desc="Processing images"):
        filename = os.path.basename(result_file)
        filename_without_ext = os.path.splitext(filename)[0]
        
        for ext in ['.jpg', '.png']:
            gt_file = os.path.join(ground_truth_dir, filename_without_ext + ext)
            if os.path.exists(gt_file):
                break
        else:  # 如果没有找到对应的文件
            continue
        
        result_img = process_image(result_file)
        gt_img = process_image(gt_file)
        
        if result_img is None or gt_img is None:
            continue
        
        results.append({
            'filename': filename,
            'dice': dc(gt_img, result_img),
            'iou': calculate_iou(gt_img, result_img)
        })
    
    return pd.DataFrame(results)

def plot_results(point_results):
    """绘制点提示方法的结果图和表格"""
    fig = plt.figure(figsize=(12, 8))
    
    # 创建网格
    gs = plt.GridSpec(2, 2, height_ratios=[1, 0.5])
    
    # DSC分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot([point_results['dice']], labels=['Points'])
    ax1.set_title('DSC Distribution')
    ax1.set_ylabel('DSC Value')
    ax1.grid(True, alpha=0.3)
    
    # IOU分布
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([point_results['iou']], labels=['Points'])
    ax2.set_title('IOU Distribution')
    ax2.set_ylabel('IOU Value')
    ax2.grid(True, alpha=0.3)
    
    # 添加表格
    summary = pd.DataFrame([{
        'Method': 'Points',
        'Mean DSC': f"{point_results['dice'].mean():.4f} ± {point_results['dice'].std():.4f}",
        'Mean IOU': f"{point_results['iou'].mean():.4f} ± {point_results['iou'].std():.4f}",
        'Min DSC': f"{point_results['dice'].min():.4f}",
        'Max DSC': f"{point_results['dice'].max():.4f}",
        'Min IOU': f"{point_results['iou'].min():.4f}",
        'Max IOU': f"{point_results['iou'].max():.4f}"
    }])
    
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    table = ax_table.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    plt.tight_layout()
    return fig, summary

def evaluate_segmentation():
    """评估分割结果"""
    print("Evaluating Points method...")
    point_dir = os.path.join(DIRS['results'], 'masks_point')
    results = evaluate_single_method(point_dir, DIRS['ground_truth'])
    
    # 生成可视化结果和保存
    fig, summary_df = plot_results(results)
    fig.savefig(os.path.join(DIRS['output'], "point_results.png"), dpi=300, bbox_inches='tight')
    
    # 保存详细结果
    results.to_csv(os.path.join(DIRS['output'], "point_results.csv"), index=False)
    summary_df.to_csv(os.path.join(DIRS['output'], "summary.csv"), index=False)
    
    print("\nResults Summary:")
    print(summary_df.to_string(index=False))
    
    plt.show()

if __name__ == "__main__":
    evaluate_segmentation()