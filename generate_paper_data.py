
import os
import sys

# Force CPU to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import pandas as pd

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

import config
import truck_routing
import dynamic_step_implementation

# --- 1. Define Detailed Reward Logic ---

def detailed_calculate_step_reward(schedule_result: Dict, time_penalty: float) -> Tuple[float, Dict[str, float]]:
    """
    Calculates step reward and returns breakdown components.
    """
    service_reward = 0.0
    efficiency_reward = 0.0
    cost_penalty = 0.0
    
    # Logic mirrored from dynamic_step_implementation.py but split
    if schedule_result and schedule_result.get('total_service_count', 0) > 0:
        total_served = schedule_result['total_service_count']
        # Service Completion Reward
        service_reward = total_served * 5.0
        
        # Efficiency Reward
        efficiency = schedule_result.get('efficiency_score', 0.0)
        efficiency_reward = efficiency * 10.0
    
    # Cost/Time Penalty
    # dynamic_step uses: step_reward -= time_penalty * 0.1
    cost_penalty = time_penalty * 0.1
    
    total_reward = service_reward + efficiency_reward - cost_penalty
    
    breakdown = {
        "service_reward": service_reward,
        "efficiency_reward": efficiency_reward,
        "cost_penalty": cost_penalty,
        "total_reward": total_reward
    }
    return total_reward, breakdown

# Global storage for episode data
episode_data_logs = [] 

def detailed_dynamic_step(env, actions: List[int]) -> Tuple[Any, List[float], bool, Dict]:
    """
    Monkey-patched version of dynamic_step that logs detailed rewards.
    """
    rewards = [0.0] * env.num_trucks
    breakdowns = [None] * env.num_trucks # Store breakdown per truck
    done = False

    # Save state before action
    state_before = env._get_current_state()

    # Update time step
    env.time_step += 1
    
    # Execute dynamic logic
    for i, truck in enumerate(env.trucks):
        action = actions[i]
        
        # Parse action
        if isinstance(action, dict):
            select_stop = action['select_stop']
        else:
            select_stop = action
        
        old_position = truck['position']
        
        if select_stop == 0:  # Return to depot
            truck['current_location'] = 0
            truck['position'] = env.depot
            truck['current_delivery_load'] = env.initial_delivery_load
            truck['current_return_load'] = 0
            truck['remaining_space'] = env.truck_capacity - truck['current_delivery_load']
            
            # Small penalty for returning without doing anything could be added here, 
            # but we stick to original logic which is 0 reward for just moving back
            rewards[i] = 0.0
            breakdowns[i] = {"service_reward": 0, "efficiency_reward": 0, "cost_penalty": 0, "total_reward": 0}

        elif select_stop <= len(env.lockers_state):  # Move to locker
            new_location_id = select_stop
            target_locker = env.get_locker(new_location_id)
            
            if target_locker and not target_locker['served']:
                truck['current_location'] = new_location_id
                truck['position'] = target_locker['location']
                truck['visited_stops'].append(new_location_id)
                
                # Dynamic Drone Scheduling
                serviceable_lockers = dynamic_step_implementation.get_serviceable_lockers(
                    truck['position'], 
                    env.lockers_state,
                    env.drone_max_range / 2
                )
                
                rl_preferences = {
                    'exploration_enabled': False, # Disable exploration for evaluation
                    'learning_weight': 0.7,
                    'diversity_bonus': 0.3,
                    'risk_tolerance': 0.5,
                    'adaptive_threshold': 0.6,
                    'truck_id': i,
                    'time_step': env.time_step,
                    'episode_progress': env.time_step / env.max_timesteps
                }
                
                schedule_result = env.drone_scheduler.schedule_drones(
                    truck_location=truck['position'],
                    available_lockers=serviceable_lockers,
                    drone_range=env.drone_max_range,
                    rl_preferences=rl_preferences
                )
                
                total_time, time_penalty = dynamic_step_implementation.execute_drone_schedule(env, truck, schedule_result, i)
                
                # Use Detailed Reward Calculation
                step_reward, breakdown = detailed_calculate_step_reward(schedule_result, time_penalty)
                
                rewards[i] = step_reward
                breakdowns[i] = breakdown
                
                truck['is_servicing'] = False
                truck['service_time'] = total_time
                
                # Truck direct service reward
                if target_locker['demand_del'] > 0 or target_locker['demand_ret'] > 0:
                    env.served_delivery += target_locker['demand_del']
                    env.served_return += target_locker['demand_ret']
                    target_locker['demand_del'] = 0
                    target_locker['demand_ret'] = 0
                    target_locker['served'] = True
                    
                    direct_service_reward = 20.0
                    rewards[i] += direct_service_reward
                    
                    # Add to service reward component
                    breakdowns[i]['service_reward'] += direct_service_reward
                    breakdowns[i]['total_reward'] += direct_service_reward
            else:
                 # Visited already served locker or invalid
                 rewards[i] = 0.0
                 breakdowns[i] = {"service_reward": 0, "efficiency_reward": 0, "cost_penalty": 0, "total_reward": 0}
        
        # Update distance
        distance = env._euclidean_distance(old_position, truck['position'])
        truck['total_distance'] += distance
        env.total_truck_distance += distance
        env.episode_truck_distance += distance

    # Check done
    done = dynamic_step_implementation.check_episode_done(env)
    
    # Update demand/uncertainty
    env._update_demand_and_handle_uncertainty()
    
    # Get next state
    next_state, action_mask = env._get_state_with_mask()
    
    # Log the step data
    # Aggregate across trucks for the episode log
    step_log = {
        "step": env.time_step,
        "service_reward": sum(b['service_reward'] for b in breakdowns if b),
        "efficiency_reward": sum(b['efficiency_reward'] for b in breakdowns if b),
        "cost_penalty": sum(b['cost_penalty'] for b in breakdowns if b),
        "total_reward": sum(rewards)
    }
    if len(episode_data_logs) > 0:
        episode_data_logs[-1].append(step_log)

    return next_state, rewards, done, action_mask

# Monkey Patch
truck_routing.dynamic_step = detailed_dynamic_step

# --- 2. Load Model and Run Episodes ---

from quick_test import load_model

def run_data_generation():
    print("Loading model...")
    try:
        mappo, env = load_model("trained_mappo_policy.pth")
    except Exception as e:
        print(f"Could not load model: {e}. Using random agent for demo purposes if model missing.")
        # Fallback or exit, but user implies model exists
        return

    num_episodes = 10
    print(f"Running {num_episodes} episodes for data collection...")

    for ep in range(num_episodes):
        episode_data_logs.append([]) # New list for this episode steps
        
        state, action_mask = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < env.max_timesteps:
            action_masks = env.get_action_masks()
            truck_states = env.get_truck_specific_states()
            
            with torch.no_grad():
                actions, log_probs, values = mappo.act(truck_states, action_masks, env)
            
            # This calls our monkey-patched dynamic_step
            next_state, rewards, done, next_action_mask = env.step(actions)
            
            state = next_state
            step_count += 1
            
        print(f"Episode {ep+1} complete. Steps: {step_count}, Total Reward: {sum(step['total_reward'] for step in episode_data_logs[-1]):.2f}")

    # --- 3. Generate Visualizations ---
    print("Generating visualizations...")
    
    # We pick the "best" episode to plot (highest reward)
    best_ep_idx = np.argmax([sum(step['total_reward'] for step in logs) for logs in episode_data_logs])
    best_logs = episode_data_logs[best_ep_idx]
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(best_logs)
    
    # Plot Reward Breakdown
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['service_reward'], label='Service Reward', color='green', marker='o', markersize=3)
    plt.plot(df['step'], df['efficiency_reward'], label='Efficiency Reward', color='blue', linestyle='--')
    plt.plot(df['step'], df['cost_penalty'], label='Cost Penalty', color='red', linestyle=':')
    plt.plot(df['step'], df['total_reward'], label='Total Reward', color='black', linewidth=2)
    
    plt.title('Reward Function Decomposition (Single Episode)')
    plt.xlabel('Time Step')
    plt.ylabel('Reward Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_decomposition.png')
    print("Saved reward_decomposition.png")
    
    # Plot Accumulated Components
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['service_reward'].cumsum(), label='Cumulative Service', color='green')
    plt.plot(df['step'], df['efficiency_reward'].cumsum(), label='Cumulative Efficiency', color='blue')
    plt.plot(df['step'], df['cost_penalty'].cumsum(), label='Cumulative Cost', color='red')
    plt.plot(df['step'], df['total_reward'].cumsum(), label='Cumulative Total', color='black', linewidth=2)
    plt.title('Cumulative Reward Components')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_rewards.png')
    print("Saved cumulative_rewards.png")

    # --- 4. Save Results to Table ---
    print("\n=== Data for Paper Table (10 Runs) ===")
    print("| Run | Total Reward | Service Reward | Efficiency Reward | Cost Penalty | Steps |")
    print("|-----|--------------|----------------|-------------------|--------------|-------|")
    
    # 准备数据列表
    results_data = []
    
    for i, logs in enumerate(episode_data_logs):
        total = sum(step['total_reward'] for step in logs)
        service = sum(step['service_reward'] for step in logs)
        eff = sum(step['efficiency_reward'] for step in logs)
        cost = sum(step['cost_penalty'] for step in logs)
        steps = len(logs)
        print(f"| {i+1} | {total:.2f} | {service:.2f} | {eff:.2f} | {cost:.2f} | {steps} |")
        
        # 添加到数据列表
        results_data.append({
            'Run': i + 1,
            'Total Reward': total,
            'Service Reward': service,
            'Efficiency Reward': eff,
            'Cost Penalty': cost,
            'Steps': steps
        })
    
    # 保存到CSV文件
    results_df = pd.DataFrame(results_data)
    csv_file = 'simulation_results.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n✅ 结果已保存到: {csv_file}")
    
    # 同时保存到Excel文件（如果可能）
    try:
        excel_file = 'simulation_results.xlsx'
        results_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ 结果已保存到: {excel_file}")
    except ImportError:
        print("⚠️  未安装openpyxl，跳过Excel文件保存（CSV文件已保存）")
    except Exception as e:
        print(f"⚠️  保存Excel文件时出错: {e}（CSV文件已保存）")

if __name__ == "__main__":
    run_data_generation()

