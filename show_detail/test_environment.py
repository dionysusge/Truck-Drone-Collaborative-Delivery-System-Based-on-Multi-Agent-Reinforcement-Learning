#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境测试脚本
作者: Dionysus
联系方式: wechat:gzw1546484791

功能:
- 加载训练好的模型
- 输出环境初始化状态
- 记录每步的环境状态、需求分布、卡车和无人机状态
- 生成详细的测试报告
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# 添加当前目录到路径，以便导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from truck_routing import TruckSchedulingEnv, MAPPO
from config import Config
import config
from detailed_test_output import DetailedTestOutputManager

# 导入 reward breakdown 相关函数
sys.path.insert(0, project_root)

# 导入 generate_paper_data 中的详细奖励计算函数
try:
    from generate_paper_data import detailed_calculate_step_reward, detailed_dynamic_step, episode_data_logs
    # Monkey patch dynamic_step
    import truck_routing
    truck_routing.dynamic_step = detailed_dynamic_step
    REWARD_BREAKDOWN_AVAILABLE = True
except ImportError:
    REWARD_BREAKDOWN_AVAILABLE = False
    episode_data_logs = None
    print("警告: 无法导入 reward breakdown 函数，将使用标准奖励记录")


class EnvironmentTester:
    """
    环境测试器类
    
    功能:
    - 加载训练好的模型
    - 运行测试回合
    - 记录详细状态信息
    - 生成测试报告
    """
    
    def __init__(self, model_path: str = None, output_dir: str = "test_results"):
        """
        初始化测试器
        
        Args:
            model_path: 训练好的模型路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.test_data = []
        self.environment_states = []
        self.demand_history = []
        self.truck_states = []
        self.drone_states = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化详细输出管理器
        self.detailed_output = DetailedTestOutputManager(output_dir)
        
        # 存储 reward breakdown 数据
        self.episode_reward_breakdown = []
        
        # 初始化环境
        self.env = TruckSchedulingEnv(verbose=True)
        
        # 获取环境参数
        self.num_trucks = self.env.num_trucks
        self.num_lockers = self.env.num_lockers
        
        # 获取单个卡车的状态维度
        truck_states = self.env.get_truck_specific_states()
        self.state_dim = len(truck_states[0]) if truck_states else len(self.env._get_current_state())
        
        # 设置动作维度（字典格式，与MAPPO期望的格式匹配）
        self.action_dim = {
            "select_stop": self.num_lockers + 1,  # 0:仓库, 1-n:快递柜
            "service_area": self.num_lockers  # 每个快递柜一个二进制选择
        }
        
        # 记录初始需求状态
        self.initial_total_demand = 0
        
        # 初始化模型
        self.mappo = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("警告: 未提供有效的模型路径，将使用随机策略进行测试")
    
    def load_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.mappo = MAPPO(
                num_trucks=self.num_trucks,
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
            
            # 兼容新版本PyTorch的weights_only参数
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                # 旧版本PyTorch不支持weights_only参数
                checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查checkpoint的结构
            if isinstance(checkpoint, dict):
                # 检查不同的键名格式
                if 'policy_net_state_dict' in checkpoint and 'value_net_state_dict' in checkpoint:
                    self.mappo.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    self.mappo.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                elif 'policy_net' in checkpoint and 'value_net' in checkpoint:
                    self.mappo.policy_net.load_state_dict(checkpoint['policy_net'])
                    self.mappo.value_net.load_state_dict(checkpoint['value_net'])
                else:
                    # 如果checkpoint直接是state_dict
                    print("警告: 模型文件格式不匹配，可用键:", list(checkpoint.keys()))
                    self.mappo = None
                    return
            else:
                print("警告: 模型文件格式不正确")
                self.mappo = None
                return
            
            print(f"成功加载模型: {model_path}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.mappo = None
    
    def save_environment_initialization(self):
        """
        保存环境初始化状态到文件
        """
        init_state = {
            'timestamp': datetime.now().isoformat(),
            'environment_config': {
                'num_trucks': self.num_trucks,
                'num_lockers': self.num_lockers,
                'truck_capacity': self.env.truck_capacity,
                'max_timesteps': self.env.max_timesteps
            },
            'initial_truck_positions': [
                {
                    'truck_id': i,
                    'position': self.env.trucks[i]['position'],
                    'capacity': self.env.trucks[i]['capacity'],
                    'current_delivery_load': self.env.trucks[i]['current_delivery_load'],
                    'current_return_load': self.env.trucks[i]['current_return_load']
                }
                for i in range(self.num_trucks)
            ],
            'locker_positions': [
                {
                    'locker_id': i,
                    'position': self.env.lockers_state[i]['location'],
                    'demand_del': self.env.lockers_state[i]['demand_del'],
                    'demand_ret': self.env.lockers_state[i]['demand_ret']
                }
                for i in range(self.num_lockers)
            ],
            'initial_demand_distribution': {
                'total_demand_del': sum(locker['demand_del'] for locker in self.env.lockers_state),
                'total_demand_ret': sum(locker['demand_ret'] for locker in self.env.lockers_state),
                'average_demand_del': np.mean([locker['demand_del'] for locker in self.env.lockers_state]),
                'average_demand_ret': np.mean([locker['demand_ret'] for locker in self.env.lockers_state]),
                'demand_variance_del': np.var([locker['demand_del'] for locker in self.env.lockers_state]),
                'demand_variance_ret': np.var([locker['demand_ret'] for locker in self.env.lockers_state]),
                'high_demand_lockers_del': [
                    i for i, locker in enumerate(self.env.lockers_state) 
                    if locker['demand_del'] > np.mean([l['demand_del'] for l in self.env.lockers_state])
                ],
                'high_demand_lockers_ret': [
                    i for i, locker in enumerate(self.env.lockers_state) 
                    if locker['demand_ret'] > np.mean([l['demand_ret'] for l in self.env.lockers_state])
                ]
            }
        }
        
        # 保存到JSON文件
        init_file = os.path.join(self.output_dir, 'environment_initialization.json')
        with open(init_file, 'w', encoding='utf-8') as f:
            json.dump(init_state, f, indent=2, ensure_ascii=False)
        
        print(f"环境初始化状态已保存到: {init_file}")
        return init_state
    
    def record_step_state(self, step: int, actions: List[int], rewards: List[float], 
                         done: bool, info: Dict = None):
        """
        记录每步的详细状态
        
        Args:
            step: 当前步数
            actions: 各卡车的动作
            rewards: 各卡车的奖励
            done: 是否结束
            info: 额外信息
        """
        # 记录 reward breakdown（如果可用）
        if REWARD_BREAKDOWN_AVAILABLE and episode_data_logs is not None:
            # 从全局的 episode_data_logs 中获取 breakdown
            # detailed_dynamic_step 会在 env.step() 执行后立即添加 step log
            # 所以我们可以直接取最新的 log
            try:
                if len(episode_data_logs) > 0:
                    latest_logs = episode_data_logs[-1]
                    if len(latest_logs) > 0:
                        # 取最新的 step log（因为 detailed_dynamic_step 在 step 执行后立即添加）
                        latest_step = latest_logs[-1]
                        # 检查是否是当前 step 的 log（通过比较 step 数量或直接使用最新的）
                        # 由于 detailed_dynamic_step 在 step 执行后立即添加，最新的 log 应该就是当前 step 的
                        step_breakdown = {
                            'step': step,
                            'service_reward': latest_step.get('service_reward', 0.0),
                            'efficiency_reward': latest_step.get('efficiency_reward', 0.0),
                            'cost_penalty': latest_step.get('cost_penalty', 0.0),
                            'total_reward': latest_step.get('total_reward', sum(rewards))
                        }
                        # 只有当这个 step 还没有被记录时才添加（避免重复）
                        if len(self.episode_reward_breakdown) == 0 or self.episode_reward_breakdown[-1]['step'] != step:
                            self.episode_reward_breakdown.append(step_breakdown)
            except Exception as e:
                # 如果获取 breakdown 失败，继续使用标准奖励记录
                pass
        # 记录环境状态
        env_state = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'total_served_demand': sum(
                locker.get('served_demand', 0) for locker in self.env.lockers_state
            ),
            'total_remaining_demand': sum(
                locker.get('demand_del', 0) + locker.get('demand_ret', 0) for locker in self.env.lockers_state
            ),
            'completion_rate': self.env._calculate_completion_rate(),
            'path_efficiency': self.env._calculate_path_efficiency(),
            'capacity_utilization': self.env._calculate_capacity_utilization(),
            'done': done
        }
        self.environment_states.append(env_state)
        
        # 记录需求分布
        demand_state = {
            'step': step,
            'locker_demands': [locker.get('demand_del', 0) + locker.get('demand_ret', 0) for locker in self.env.lockers_state],
            'locker_served': [locker.get('served_demand', 0) for locker in self.env.lockers_state],
            'demand_hotspots': self.env._identify_demand_hotspots(),
            'unserved_demand_total': self.env._calculate_total_unserved_demand()
        }
        self.demand_history.append(demand_state)
        
        # 记录卡车状态
        truck_state = {
            'step': step,
            'trucks': []
        }
        
        for i, truck in enumerate(self.env.trucks):
            truck_info = {
                'truck_id': i,
                'position': truck['position'],
                'capacity': truck['capacity'],
                'current_delivery_load': truck.get('current_delivery_load', 0),
                'current_return_load': truck.get('current_return_load', 0),
                'current_load': truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0),
                'action': actions[i] if i < len(actions) else -1,
                'reward': rewards[i] if i < len(rewards) else 0.0,
                'total_distance': truck.get('total_distance', 0),
                'served_lockers': truck.get('served_lockers', []),
                'efficiency_score': self._calculate_truck_efficiency(i)
            }
            truck_state['trucks'].append(truck_info)
        
        self.truck_states.append(truck_state)
        
        # 记录详细操作信息
        for i, truck in enumerate(self.env.trucks):
            if i < len(actions) and i < len(rewards):
                # 计算时间花费（基于移动距离和服务时间）
                time_cost = self._calculate_time_cost(actions[i], truck)
                
                # 记录详细操作
                self.detailed_output.record_truck_step_operation(
                    step=step,
                    truck_id=i,
                    action=actions[i],
                    truck_state=truck,
                    env_state={'lockers_state': self.env.lockers_state},
                    reward=rewards[i],
                    time_cost=time_cost
                )
                
                # 记录无人机服务详情（如果有）
                if hasattr(self.env, 'drone_scheduler') and self.env.drone_scheduler:
                    drone_services = self._get_drone_services_for_truck(i, actions[i])
                    if drone_services:
                        self.detailed_output.record_drone_service_details(step, i, drone_services)
        
        # 记录无人机状态（如果有）
        if hasattr(self.env, 'drone_scheduler') and self.env.drone_scheduler:
            drone_state = {
                'step': step,
                'active_schedules': len(self.env.drone_scheduler.active_schedules),
                'completed_tasks': len(self.env.drone_scheduler.completed_tasks),
                'total_scheduled_time': sum(
                    schedule.total_time 
                    for schedule in self.env.drone_scheduler.active_schedules
                )
            }
            self.drone_states.append(drone_state)
    
    def _calculate_truck_efficiency(self, truck_id: int) -> float:
        """
        计算卡车效率分数
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            效率分数
        """
        truck = self.env.trucks[truck_id]
        
        if truck.get('total_distance', 0) == 0:
            return 0.0
        
        # 计算效率：服务的需求量 / 行驶距离
        served_demand = sum(
            self.env.lockers[locker_id].get('served_demand', 0)
            for locker_id in truck.get('served_lockers', [])
        )
        
        efficiency = served_demand / max(truck.get('total_distance', 1), 1)
        return efficiency
    
    def _calculate_time_cost(self, action: Dict[str, Any], truck_state: Dict[str, Any]) -> float:
        """
        计算动作的时间花费
        
        Args:
            action: 执行的动作
            truck_state: 卡车状态
            
        Returns:
            time_cost: 时间花费
        """
        time_cost = 0.0
        
        if isinstance(action, dict):
            select_stop = action.get('select_stop', 0)
            service_area = action.get('service_area', [])
            
            # 计算移动时间
            current_pos = truck_state.get('position', (0, 0))
            if select_stop == 0:
                target_pos = (0, 0)  # 仓库
            else:
                # 获取快递柜位置
                if hasattr(self.env, 'lockers_state') and select_stop - 1 < len(self.env.lockers_state):
                    target_pos = self.env.lockers_state[select_stop - 1].get('location', (0, 0))
                else:
                    target_pos = (0, 0)
            
            # 计算移动距离和时间
            distance = np.sqrt((current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2)
            truck_speed = getattr(Config, 'TRUCK_SPEED', 20)
            move_time = distance / truck_speed
            
            # 计算服务时间
            service_time = 0
            if select_stop > 0:  # 不是返回仓库
                truck_service_time = getattr(Config, 'TRUCK_SERVICE_TIME', 60)
                service_time = truck_service_time
                
                # 计算无人机服务时间（与调度器逻辑保持一致）
                drone_service_time = 0
                for i, serve in enumerate(service_area):
                    if serve == 1 and i < len(self.env.lockers_state):
                        locker = self.env.lockers_state[i]
                        locker_pos = locker.get('location', (0, 0))
                        
                        # 计算实际飞行距离
                        flight_distance = np.sqrt(
                            (target_pos[0] - locker_pos[0])**2 + 
                            (target_pos[1] - locker_pos[1])**2
                        )
                        
                        # 计算服务需求
                        service_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                        
                        drone_speed = getattr(Config, 'DRONE_SPEED', 1.0)
                        drone_service_per_demand = getattr(Config, 'DRONE_SERVICE_TIME', 2)
                        
                        # 估算无人机服务时间：飞行时间 + 服务时间
                        flight_time = flight_distance / drone_speed * 2  # 往返时间
                        service_time_for_locker = service_demand * drone_service_per_demand
                        drone_service_time += flight_time + service_time_for_locker
                
                service_time += drone_service_time
            
            time_cost = move_time + service_time
        
        return time_cost
    
    def _get_drone_services_for_truck(self, truck_id: int, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取卡车的无人机服务详情
        
        Args:
            truck_id: 卡车ID
            action: 执行的动作
            
        Returns:
            drone_services: 无人机服务列表
        """
        drone_services = []
        
        if isinstance(action, dict):
            service_area = action.get('service_area', [])
            truck_pos = self.env.trucks[truck_id].get('position', (0, 0))
            
            for i, serve in enumerate(service_area):
                if serve == 1 and i < len(self.env.lockers_state):
                    locker = self.env.lockers_state[i]
                    locker_pos = locker.get('location', (0, 0))
                    
                    # 计算飞行距离
                    flight_distance = np.sqrt(
                        (truck_pos[0] - locker_pos[0])**2 + 
                        (truck_pos[1] - locker_pos[1])**2
                    )
                    
                    # 计算服务需求
                    service_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                    
                    # 计算服务时间（与调度器逻辑保持一致）
                    drone_speed = getattr(Config, 'DRONE_SPEED', 1.0)  # 使用正确的默认速度
                    drone_service_time_per_item = getattr(Config, 'DRONE_SERVICE_TIME', 2)
                    
                    flight_time = flight_distance / drone_speed * 2  # 往返
                    service_time = service_demand * drone_service_time_per_item  # 服务时间 = 需求数量 × 单项服务时间
                    total_time = flight_time + service_time
                    
                    drone_service = {
                        'drone_id': i % getattr(Config, 'DRONE_NUM', 3),  # 简单分配
                        'service_type': 'delivery_and_return',
                        'target_locker': i + 1,
                        'locker_position': locker_pos,
                        'service_demand': service_demand,
                        'flight_distance': flight_distance,
                        'service_time': service_time,
                        'total_time': total_time,
                        'efficiency': service_demand / max(total_time, 1)
                    }
                    drone_services.append(drone_service)
        
        return drone_services
    
    def run_test_episode(self, max_steps: int = None) -> Dict[str, Any]:
        """
        运行一个测试回合
        
        Args:
            max_steps: 最大步数
            
        Returns:
            测试结果
        """
        if max_steps is None:
            max_steps = self.env.max_steps
        
        # 重置环境
        states = self.env.reset()
        
        # 重置 reward breakdown 记录
        self.episode_reward_breakdown = []
        
        # 如果使用 reward breakdown，初始化全局的 episode_data_logs
        if REWARD_BREAKDOWN_AVAILABLE and episode_data_logs is not None:
            episode_data_logs.append([])  # 为这个 episode 创建新的日志列表
        
        # 记录初始总需求
        self.initial_total_demand = sum(
            locker['demand_del'] + locker['demand_ret'] 
            for locker in self.env.lockers_state
        )
        
        # 保存初始化状态
        init_state = self.save_environment_initialization()
        
        total_rewards = [0.0] * self.num_trucks
        episode_length = 0
        
        print(f"开始测试回合，最大步数: {max_steps}")
        
        for step in range(max_steps):
            # 获取动作
            if self.mappo:
                # 使用训练好的模型
                action_masks = self.env.get_action_masks()
                # 获取单个卡车的状态
                truck_states = self.env.get_truck_specific_states()
                actions, _, _ = self.mappo.act(truck_states, action_masks, self.env)
            else:
                # 使用随机策略
                actions = self._get_random_actions()
            
            # 执行动作
            next_states, rewards, done, info = self.env.step(actions)
            
            # 记录状态
            self.record_step_state(step, actions, rewards, done, info)
            
            # 更新累计奖励
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward
            
            episode_length = step + 1
            
            if done:
                print(f"回合在第 {step + 1} 步结束")
                break
            
            states = next_states
        
        # 计算测试结果
        test_result = {
            'episode_length': episode_length,
            'total_rewards': total_rewards,
            'average_reward': np.mean(total_rewards),
            'final_completion_rate': self.env._calculate_completion_rate(),
            'final_path_efficiency': self.env._calculate_path_efficiency(),
            'final_capacity_utilization': self.env._calculate_capacity_utilization(),
            'total_served_demand': self._calculate_total_served_demand(),
            'total_distance': sum(
                truck.get('total_distance', 0) for truck in self.env.trucks
            )
        }
        
        # 生成详细测试报告
        if self.detailed_output:
            detailed_report = self.detailed_output.generate_text_report()
            print("\n=== 详细测试报告 ===")
            print(detailed_report)
            
            # 保存详细报告到文件
            report_file = os.path.join(self.output_dir, 'detailed_test_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(detailed_report)
            print(f"\n详细测试报告已保存到: {report_file}")
            
            # 同时保存JSON格式的详细数据
            json_report = self.detailed_output.generate_detailed_report()
            json_file = os.path.join(self.output_dir, 'detailed_test_data.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"详细测试数据已保存到: {json_file}")
        
        print(f"测试完成！")
        print(f"回合长度: {episode_length}")
        print(f"平均奖励: {test_result['average_reward']:.2f}")
        print(f"完成率: {test_result['final_completion_rate']:.2%}")
        print(f"路径效率: {test_result['final_path_efficiency']:.2f}")
        
        return test_result
    
    def _calculate_total_served_demand(self) -> int:
        """
        计算总的已服务需求
        
        Returns:
            已服务的总需求量
        """
        # 计算当前剩余需求
        current_total_demand = sum(
            locker['demand_del'] + locker['demand_ret'] 
            for locker in self.env.lockers_state
        )
        
        # 已服务需求 = 初始需求 - 剩余需求
        served_demand = self.initial_total_demand - current_total_demand
        return max(0, served_demand)  # 确保非负
    
    def _get_random_actions(self) -> List[Dict[str, Any]]:
        """
        生成随机动作
        
        Returns:
            随机动作列表，每个动作包含select_stop和service_area
        """
        actions = []
        action_masks = self.env.get_action_masks()
        
        for truck_id in range(self.num_trucks):
            if truck_id < len(action_masks):
                mask = action_masks[truck_id]
                
                # 选择停靠点
                stop_mask = mask['stop_mask']
                valid_stops = [i for i, valid in enumerate(stop_mask) if valid > 0]
                
                if valid_stops:
                    select_stop = np.random.choice(valid_stops)
                else:
                    select_stop = 0  # 返回仓库
                
                # 选择服务区域
                service_mask = mask['service_mask']
                service_area = []
                for i, valid in enumerate(service_mask):
                    if valid > 0:
                        # 随机决定是否服务该区域
                        service_area.append(1 if np.random.random() > 0.5 else 0)
                    else:
                        service_area.append(0)
                
                action = {
                    "select_stop": select_stop,
                    "service_area": service_area
                }
            else:
                # 默认动作
                action = {
                    "select_stop": 0,
                    "service_area": [0] * self.env.num_lockers
                }
            
            actions.append(action)
        
        return actions
    
    def save_test_results(self, test_result: Dict[str, Any]):
        """
        保存测试结果到文件
        
        Args:
            test_result: 测试结果
        """
        # 保存详细状态数据
        detailed_data = {
            'test_summary': test_result,
            'environment_states': self.environment_states,
            'demand_history': self.demand_history,
            'truck_states': self.truck_states,
            'drone_states': self.drone_states
        }
        
        # 添加 reward breakdown 数据（如果可用）
        if self.episode_reward_breakdown:
            detailed_data['reward_breakdown'] = self.episode_reward_breakdown
            
            # 计算并保存 summary table 格式的数据
            total_reward = sum(step['total_reward'] for step in self.episode_reward_breakdown)
            service_reward = sum(step['service_reward'] for step in self.episode_reward_breakdown)
            efficiency_reward = sum(step['efficiency_reward'] for step in self.episode_reward_breakdown)
            cost_penalty = sum(step['cost_penalty'] for step in self.episode_reward_breakdown)
            steps = len(self.episode_reward_breakdown)
            
            summary_table_data = {
                'Run': 1,
                'Total Reward': total_reward,
                'Service Reward': service_reward,
                'Efficiency Reward': efficiency_reward,
                'Cost Penalty': cost_penalty,
                'Steps': steps
            }
            detailed_data['summary_table'] = summary_table_data
            
            # 保存为CSV格式（与 generate_paper_data.py 格式一致）
            summary_df = pd.DataFrame([summary_table_data])
            summary_csv_file = os.path.join(self.output_dir, 'simulation_results.csv')
            summary_df.to_csv(summary_csv_file, index=False, encoding='utf-8')
            print(f"✅ Reward breakdown 结果已保存到: {summary_csv_file}")
            
            # 同时保存到Excel文件（如果可能）
            try:
                summary_excel_file = os.path.join(self.output_dir, 'simulation_results.xlsx')
                summary_df.to_excel(summary_excel_file, index=False, engine='openpyxl')
                print(f"✅ Reward breakdown 结果已保存到: {summary_excel_file}")
            except ImportError:
                print("⚠️  未安装openpyxl，跳过Excel文件保存（CSV文件已保存）")
            except Exception as e:
                print(f"⚠️  保存Excel文件时出错: {e}（CSV文件已保存）")
        
        # 保存到JSON文件
        results_file = os.path.join(self.output_dir, 'detailed_test_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        # 保存CSV格式的数据用于分析
        self._save_csv_data()
        
        print(f"详细测试结果已保存到: {results_file}")
    
    def _json_serializer(self, obj):
        """
        JSON序列化器，处理numpy类型
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            可序列化的对象
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    
    def _save_csv_data(self):
        """
        保存CSV格式的数据
        """
        # 环境状态CSV
        if self.environment_states:
            env_df = pd.DataFrame(self.environment_states)
            env_df.to_csv(
                os.path.join(self.output_dir, 'environment_states.csv'),
                index=False, encoding='utf-8'
            )
        
        # 卡车状态CSV
        if self.truck_states:
            truck_data = []
            for state in self.truck_states:
                for truck in state['trucks']:
                    truck_data.append({
                        'step': state['step'],
                        **truck
                    })
            
            truck_df = pd.DataFrame(truck_data)
            truck_df.to_csv(
                os.path.join(self.output_dir, 'truck_states.csv'),
                index=False, encoding='utf-8'
            )
        
        # 需求历史CSV
        if self.demand_history:
            demand_data = []
            for state in self.demand_history:
                for i, demand in enumerate(state['locker_demands']):
                    demand_data.append({
                        'step': state['step'],
                        'locker_id': i,
                        'demand': demand,
                        'served': state['locker_served'][i]
                    })
            
            demand_df = pd.DataFrame(demand_data)
            demand_df.to_csv(
                os.path.join(self.output_dir, 'demand_history.csv'),
                index=False, encoding='utf-8'
            )
    
    def generate_visualization(self):
        """
        生成可视化图表
        """
        if not self.environment_states:
            print("没有数据可供可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Environment Test Results Visualization', fontsize=16)
        
        # 1. 完成率变化
        steps = [state['step'] for state in self.environment_states]
        completion_rates = [state['completion_rate'] for state in self.environment_states]
        
        axes[0, 0].plot(steps, completion_rates, 'b-', linewidth=2)
        axes[0, 0].set_title('Completion Rate Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Completion Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 路径效率变化
        path_efficiency = [state['path_efficiency'] for state in self.environment_states]
        
        axes[0, 1].plot(steps, path_efficiency, 'g-', linewidth=2)
        axes[0, 1].set_title('Path Efficiency Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Path Efficiency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 卡车奖励分布
        if self.truck_states:
            truck_rewards = []
            for state in self.truck_states:
                for truck in state['trucks']:
                    truck_rewards.append(truck['reward'])
            
            axes[1, 0].hist(truck_rewards, bins=30, alpha=0.7, color='orange')
            axes[1, 0].set_title('Truck Reward Distribution')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 容量利用率
        capacity_util = [state['capacity_utilization'] for state in self.environment_states]
        
        axes[1, 1].plot(steps, capacity_util, 'r-', linewidth=2)
        axes[1, 1].set_title('Capacity Utilization Over Time')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Capacity Utilization')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(self.output_dir, 'test_visualization.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {chart_file}")


def main():
    """
    主函数
    """
    print("=== 环境测试脚本 ===")
    
    # 模型路径配置
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_model.pth")  # 模型路径
    output_dir = "test_results"  # 输出目录
    max_steps = 200  # 最大测试步数
    
    # 创建测试器
    tester = EnvironmentTester(model_path=model_path, output_dir=output_dir)
    
    # 运行测试
    test_result = tester.run_test_episode(max_steps=max_steps)
    
    # 保存结果
    tester.save_test_results(test_result)
    
    # 生成可视化
    tester.generate_visualization()
    
    print("\n=== 测试完成 ===")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()