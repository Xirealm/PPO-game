import os
import torch
import networkx as nx
import random
from collections import deque
import time
from datetime import timedelta, datetime
from agents import NodeOptimizationEnv, NodeAgent
from utils import calculate_distances, convert_to_edges
import json

# Constants
SIZE = 560
DATASET = 'TEM'
CATAGORY = 'train'
BASE_DIR = os.path.dirname(__file__)

def train_single_agent(node_agent, episodes, output_path, base_dir, file_prefixes, max_steps):
    """训练单智能体系统"""
    rewards = []
    best_reward = -float('inf')
    image_size = SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()

    for episode in range(episodes):
        selected_prefix = random.choice(list(file_prefixes))
        print(f"Episode {episode + 1}/{episodes}, Selected file prefix: {selected_prefix}")
        
        feature_file = os.path.join(base_dir, f"{selected_prefix}_features.pt")
        pos_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_pos.pt")
        neg_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_neg.pt")
        
        if not all(os.path.exists(f) for f in [feature_file, pos_file, neg_file]):
            print(f"Required files not found for prefix {selected_prefix}, skipping this episode.")
            continue
        
        features = torch.load(feature_file, weights_only=True).to(device)
        positive_indices = torch.load(pos_file, weights_only=True).to(device)
        negative_indices = torch.load(neg_file, weights_only=True).to(device)

        # 确保索引唯一性和移除交集
        positive_indices = torch.unique(positive_indices).to(device)
        negative_indices = torch.unique(negative_indices).to(device)
        set1 = set(positive_indices.tolist())
        set2 = set(negative_indices.tolist())
        intersection = set1.intersection(set2)
        if intersection:
            positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
            negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
            
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        print(f"Positive indices: {positive_indices.shape}, Negative indices: {negative_indices.shape}")

        # 计算距离和创建图结构
        feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
            features, positive_indices, negative_indices, image_size, device)

        feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
        physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
        physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
        feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
        physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

        G = nx.MultiGraph()
        G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
        G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')

        G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
        G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
        G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
        G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
        G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

        # 初始化环境
        node_agent.env = NodeOptimizationEnv(G, image_size, max_steps, best_reward=best_reward)
        
        if node_agent.policy_net is None:
            node_agent.initialize_networks(features)
            
        state = node_agent.env.reset()
        done = False
        total_reward = 0

        # 根据最佳奖励调整epsilon
        print(f"Current best reward: {best_reward}")
        if best_reward < 0:
            node_agent.epsilon = node_agent.epsilon_start
        elif best_reward >= 10:
            node_agent.epsilon = node_agent.epsilon_end
        else:
            normalized_reward = (best_reward - 0) / (5 - 0)
            node_agent.epsilon = 1 - normalized_reward

        print(f"Best reward: {best_reward}, Node Epsilon: {node_agent.epsilon}")

        # 训练循环
        while not done:
            state_tensor = node_agent._get_state_features(state)
            action = node_agent.get_action(state)
            next_state, reward, done = node_agent.env.step(action)
            next_state_tensor = node_agent._get_state_features(next_state)
            
            node_agent.memory.append((state_tensor, action, reward, next_state_tensor))
            node_agent.replay()
            node_agent.update_epsilon()
            
            state = next_state
            total_reward += reward
            
            if len(node_agent.memory) % 100 == 0:
                node_agent.update_target_network()

        print(f"Total reward: {total_reward}")
        
        if total_reward > best_reward:
            print(f"New best reward: {total_reward}")
            best_reward = total_reward
            node_agent.env.best_reward = best_reward
            node_agent.best_memory = deque(node_agent.memory, maxlen=node_agent.memory.maxlen)
            node_agent.replay_best()

            best_model_path = os.path.join(output_path, 'best_models')
            os.makedirs(best_model_path, exist_ok=True)
            
            torch.save({
                'policy_net': node_agent.policy_net.state_dict(),
                'target_net': node_agent.target_net.state_dict(),
                'optimizer': node_agent.optimizer.state_dict(),
                'episode': episode,
                'best_reward': best_reward
            }, os.path.join(best_model_path, 'best_model.pt'))
            
            torch.save(node_agent.policy_net.state_dict(), os.path.join(best_model_path, 'node_best_model.pkl'))
            print(f'Updated best model at episode {episode}')

        rewards.append(total_reward)
        save_agent_results(node_agent, episode, episodes, total_reward, output_path, selected_prefix)

        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * (episodes / (episode + 1))
        remaining_time = estimated_total_time - elapsed_time
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_model_path = os.path.join(output_path, 'final_models')
    os.makedirs(final_model_path, exist_ok=True)
    
    torch.save({
        'policy_net': node_agent.policy_net.state_dict(),
        'target_net': node_agent.target_net.state_dict(),
        'optimizer': node_agent.optimizer.state_dict(),
        'episode': episodes,
        'final_reward': total_reward
    }, os.path.join(final_model_path, 'final_model.pt'))
    
    torch.save(node_agent.policy_net.state_dict(), os.path.join(final_model_path, 'node_final_model.pkl'))
    print('Saved final models')
    
    return rewards

def save_agent_results(node_agent, current_episode, total_episodes, reward, output_path, prefix):
    """保存智能体训练结果"""
    node_state = node_agent.env.get_state()
    pos_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'pos']
    neg_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'neg']

    episode_dir = os.path.join(output_path, 'episode_results')
    os.makedirs(episode_dir, exist_ok=True)

    info_path = os.path.join(episode_dir, f'training_log.txt')
    with open(info_path, "a") as f:
        f.write(f"Episode Progress: {current_episode + 1}/{total_episodes}\n")
        f.write(f"Data Prefix: {prefix}\n")
        f.write(f"Node Agent Epsilon: {node_agent.epsilon:.4f}\n")
        f.write(f"Reward: {reward}\n")
        f.write(f"Positive nodes: {len(pos_nodes)}, Negative nodes: {len(neg_nodes)}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")

def main():
    base_dir = os.path.join(BASE_DIR, 'train_data')
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    file_prefixes = set('_'.join(f.split('_')[:3]) for f in files)
    episodes = 1000
    max_steps = 100
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(os.path.dirname(__file__), 'train', current_time)
    os.makedirs(output_path, exist_ok=True)
    
    config = {
        'episodes': episodes,
        'max_steps': max_steps,
        'dataset': DATASET,
        'category': CATAGORY,
        'image_size': SIZE
    }
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    node_env = NodeOptimizationEnv
    node_agent = NodeAgent(node_env)
    
    rewards = train_single_agent(
        node_agent,
        episodes=episodes,
        output_path=output_path,
        base_dir=base_dir,
        file_prefixes=file_prefixes,
        max_steps=max_steps
    )
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Single-Agent Training Rewards Over Episodes')
    plt.savefig(os.path.join(output_path, "rewards_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()
