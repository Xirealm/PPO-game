import torch
import numpy as np
from sklearn.cluster import DBSCAN

def get_patch_coordinates(indices, image_size=560, patch_size=14):
    """
    计算patch在物理空间中的中心点坐标
    """
    rows = indices.cpu().numpy() // (image_size // patch_size)
    cols = indices.cpu().numpy() % (image_size // patch_size)
    centers_x = (cols * patch_size + patch_size // 2).astype(int)
    centers_y = (rows * patch_size + patch_size // 2).astype(int)
    return np.column_stack((centers_x, centers_y))

def cluster_features(features, pos_indices, eps=0.3, min_samples=2):
    """多尺度特征聚类策略"""
    all_features = features[1]  # [N, D]
    device = all_features.device
    patches_per_side = int(np.sqrt(all_features.shape[0]))
    
    pos_features = all_features[pos_indices]
    pos_center = pos_features.mean(0)
    similarities = torch.nn.functional.cosine_similarity(all_features, pos_center.unsqueeze(0))
    
    clustered_indices = []
    # 增加更多的相似度阈值档位
    similarity_thresholds = [0.8, 0.7, 0.6, 0.5]
    
    for threshold in similarity_thresholds:
        similar_mask = similarities > threshold
        candidate_indices = torch.where(similar_mask)[0].to(device)
        
        if len(candidate_indices) >= min_samples:
            coords = get_patch_coordinates(candidate_indices)
            pos_coords = get_patch_coordinates(pos_indices)
            # 减小距离限制，允许检测更小的区域
            dist_mask = np.min(np.linalg.norm(coords[:, None] - pos_coords, axis=2), axis=1) < 84
            candidate_indices = candidate_indices[torch.from_numpy(dist_mask).to(device)]
            
            if len(candidate_indices) >= min_samples:
                # 使用更小的eps值进行聚类
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_features[candidate_indices].cpu().numpy())
                for label in np.unique(clustering.labels_):
                    if label != -1:
                        cluster_indices = candidate_indices[clustering.labels_ == label]
                        if len(cluster_indices) >= min_samples:
                            clustered_indices.append(cluster_indices)
    
    # 使用更小的窗口和步长进行细粒度扫描
    window_size, stride = 3, 2
    for i in range(0, patches_per_side - window_size + 1, stride):
        for j in range(0, patches_per_side - window_size + 1, stride):
            window_indices = torch.tensor([(i + wi) * patches_per_side + (j + wj) 
                                         for wi in range(window_size) 
                                         for wj in range(window_size)], device=device)
            # 窗口内的相似度要求
            high_sim_mask = similarities[window_indices] > 0.5
            if high_sim_mask.sum() >= min_samples:
                clustered_indices.append(window_indices[high_sim_mask])
    
    # 移除重复的聚类
    final_clusters = []
    used_indices = set()
    
    for indices in clustered_indices:
        indices_set = set(indices.cpu().numpy())
        new_indices = indices_set - used_indices
        if len(new_indices) >= min_samples:
            final_clusters.append(torch.tensor(list(new_indices), device=device))
            used_indices.update(new_indices)
    
    return final_clusters

def calculate_box_from_indices(indices, image_size=560, patch_size=14):
    """计算边界框"""
    coordinates = get_patch_coordinates(indices)
    padding = patch_size // 2
    
    min_x, min_y = np.maximum(0, np.min(coordinates, axis=0) - padding)
    max_x, max_y = np.minimum(image_size, np.max(coordinates, axis=0) + padding)
    
    # 确保最小尺寸
    for min_val, max_val in [(min_x, max_x), (min_y, max_y)]:
        if max_val - min_val < patch_size * 2:
            center = (max_val + min_val) // 2
            min_val = max(0, center - patch_size)
            max_val = min(image_size, center + patch_size)
    
    return (min_x, min_y, max_x, max_y)

def filter_overlapping_boxes(boxes, iou_threshold=0.7):
    """
    过滤重叠度过高的box
    """
    if not boxes:
        return []
    
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection / (area1 + area2 - intersection)
        return iou
    
    # 按面积从大到小排序
    boxes = sorted(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
    filtered_boxes = []
    
    for box1 in boxes:
        should_keep = True
        for box2 in filtered_boxes:
            if calculate_iou(box1, box2) > iou_threshold:
                should_keep = False
                print(1)
                break
        if should_keep:
            filtered_boxes.append(box1)
    
    return filtered_boxes

def count_patches(box, patch_size=14):
    """计算box包含的patch数量"""
    width = (box[2] - box[0]) // patch_size
    height = (box[3] - box[1]) // patch_size
    return width * height

def merge_boxes(boxes, min_distance=112, patch_size=14):
    """
    修改合并策略，过滤patch数量少的box并合并邻接box
    """
    if not boxes:
        return None
    
    def are_adjacent(box1, box2):
        # 计算两个box的中心点距离
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        return distance < min_distance * 1.5

    # 首先筛选出有效的box
    valid_boxes = [box for box in boxes if count_patches(box) > 4]
    if not valid_boxes:
        return None

    # 处理小box
    merged = False
    for box in boxes:
        if count_patches(box) <= 4:  # 对于小box
            for i, valid_box in enumerate(valid_boxes):
                if are_adjacent(box, valid_box):
                    # 合并小box到邻接的valid box
                    valid_boxes[i] = (
                        min(valid_box[0], box[0]),
                        min(valid_box[1], box[1]),
                        max(valid_box[2], box[2]),
                        max(valid_box[3], box[3])
                    )
                    merged = True
                    break
    
    # 如果只有一个box，直接返回
    if len(valid_boxes) == 1:
        return valid_boxes[0]
    
    # 计算最终的合并box
    final_box = (
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes)
    )
    
    # 确保边界有效
    image_size = 560
    final_box = (
        max(0, final_box[0]),
        max(0, final_box[1]),
        min(image_size, final_box[2]),
        min(image_size, final_box[3])
    )
    
    return final_box

def remove_isolated_boxes(boxes, max_distance=112):
    """
    移除孤立的box（没有邻接box的box）
    
    Args:
        boxes: box列表
        max_distance: 两个box被认为是邻接的最大中心点距离
    
    Returns:
        non_isolated_boxes: 移除孤立box后的列表
    """
    if not boxes or len(boxes) == 1:
        return boxes
        
    def get_box_center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        
    def has_neighbor(box, other_boxes):
        center = get_box_center(box)
        for other_box in other_boxes:
            if box == other_box:
                continue
            other_center = get_box_center(other_box)
            distance = ((center[0] - other_center[0]) ** 2 + 
                       (center[1] - other_center[1]) ** 2) ** 0.5
            if distance < max_distance:
                return True
        return False
    
    non_isolated_boxes = [box for box in boxes if has_neighbor(box, boxes)]
    return non_isolated_boxes

def generate_boxes(features, pos_indices):
    """生成边界框"""
    clusters = cluster_features(features, pos_indices)
    boxes = [calculate_box_from_indices(indices) for indices in clusters]
    
    # 过滤重叠box
    filtered_boxes = filter_overlapping_boxes(boxes)
    
    # 移除孤立的box
    filtered_boxes = remove_isolated_boxes(filtered_boxes)
        
    merged_box = merge_boxes(filtered_boxes)
    return filtered_boxes, merged_box
