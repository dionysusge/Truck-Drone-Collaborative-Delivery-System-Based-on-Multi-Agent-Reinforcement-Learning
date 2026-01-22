"""
动态step方法实现
作者: Dionysus
联系方式: wechat:gzw1546484791

实现新的动态调度逻辑：
1. 卡车决定停靠点
2. 在300s内动态调度无人机
3. 等待所有无人机返回（软时间窗）
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from config import Config


def get_serviceable_lockers(truck_position: Tuple[float, float], lockers_state: List[Dict], max_range: float = 100.0) -> List[Dict]:
    """
    获取卡车当前位置可服务的快递柜列表
    
    参数:
        truck_position: 卡车当前位置 (x, y)
        lockers_state: 所有快递柜状态列表
        max_range: 最大服务范围
        
    返回:
        可服务的快递柜列表
    """
    serviceable = []
    
    for locker in lockers_state:
        if locker['served']:
            continue
            
        # 计算距离
        distance = np.sqrt(
            (truck_position[0] - locker['location'][0])**2 + 
            (truck_position[1] - locker['location'][1])**2
        )
        
        # 检查是否在服务范围内且有需求
        if distance <= max_range and (locker['demand_del'] > 0 or locker['demand_ret'] > 0):
            locker_info = locker.copy()
            locker_info['distance'] = distance
            serviceable.append(locker_info)
    
    return serviceable


def execute_drone_schedule(env, truck: Dict, schedule_result: Dict, truck_id: int) -> Tuple[float, float]:
    """
    执行无人机调度并等待完成
    
    参数:
        env: 环境对象
        truck: 卡车信息
        schedule_result: 调度结果
        truck_id: 卡车ID
        
    返回:
        (总时间, 时间窗惩罚)
    """
    total_time = 0.0
    penalty = 0.0
    
    if not schedule_result or not schedule_result.get('success', False):
        return total_time, penalty
    
    # 获取调度信息
    deployment_details = schedule_result.get('deployment_details', [])
    max_completion_time = schedule_result.get('max_completion_time', 0.0)
    total_time = max_completion_time
    
    # 执行所有调度任务
    for drone_detail in deployment_details:
        tasks = drone_detail.get('tasks', [])
        
        for task in tasks:
            # 更新快递柜状态
            locker = env.get_locker(task['locker_id'])
            if locker:
                # 获取任务中的需求信息
                delivery_demand = task['delivery_demand']
                return_demand = task['return_demand']
                
                # 确保不超过快递柜的实际需求
                actual_delivered = min(delivery_demand, locker['demand_del'])
                actual_returned = min(return_demand, locker['demand_ret'])
                
                # 更新需求
                locker['demand_del'] -= actual_delivered
                locker['demand_ret'] -= actual_returned
                
                # 更新统计
                env.served_delivery += actual_delivered
                env.served_return += actual_returned
                
                # 如果完全满足了该快递柜的需求
                if locker['demand_del'] == 0 and locker['demand_ret'] == 0:
                    locker['served'] = True
    
    # 计算软时间窗惩罚
    if total_time > 300:  # 超过300秒时间窗
        penalty = (total_time - 300) * 0.1  # 每秒超时惩罚0.1
    
    return total_time, penalty


def calculate_step_reward(schedule_result: Dict, time_penalty: float) -> Tuple[float, Dict[str, float]]:
    """
    计算单步奖励并返回分解
    
    参数:
        schedule_result: 调度结果
        time_penalty: 时间窗惩罚
        
    返回:
        (总奖励, 奖励分解字典)
    """
    step_reward = 0.0
    breakdown = {
        "service_reward": 0.0,
        "efficiency_reward": 0.0,
        "cost_penalty": 0.0,
        "total_reward": 0.0
    }
    
    if schedule_result and schedule_result.get('total_service_count', 0) > 0:
        # 每个服务的需求给予基础奖励
        total_served = schedule_result['total_service_count']
        service_comp = total_served * 5
        step_reward += service_comp
        breakdown['service_reward'] += service_comp
        
        # 效率奖励（基于服务密度）
        efficiency = schedule_result.get('efficiency_score', 0.0)
        efficiency_comp = efficiency * 10
        step_reward += efficiency_comp
        breakdown['efficiency_reward'] += efficiency_comp
    
    # 轻微的时间惩罚，鼓励快速完成
    penalty_comp = time_penalty * 0.1
    step_reward -= penalty_comp
    breakdown['cost_penalty'] += penalty_comp
    
    breakdown['total_reward'] = step_reward
    
    return step_reward, breakdown


def check_episode_done(env) -> bool:
    """
    检查回合是否结束
    
    参数:
        env: 环境对象
        
    返回:
        是否结束
    """
    # 检查所有需求是否完成（优先条件）
    all_demands_satisfied = True
    for locker in env.lockers_state:
        # 检查每个快递柜的配送和退货需求是否都为0
        if locker.get('demand_del', 0) > 0 or locker.get('demand_ret', 0) > 0:
            all_demands_satisfied = False
            break
    
    # 如果所有需求都满足，立即结束episode
    if all_demands_satisfied:
        return True
    
    # 检查时间限制（备用条件）
    time_limit_reached = env.time_step >= env.max_timesteps
    
    return time_limit_reached


def dynamic_step(env, actions: List[int]) -> Tuple[Any, List[float], bool, Dict]:
    """
    新的动态step方法实现
    
    参数:
        env: 环境对象
        actions: 动作列表
        
    返回:
        (状态, 奖励, 完成标志, 动作掩码)
    """
    rewards = [0.0] * env.num_trucks
    
    # 初始化本步的奖励分解列表
    step_breakdowns = []
    for _ in range(env.num_trucks):
        step_breakdowns.append({
            "service_reward": 0.0,
            "efficiency_reward": 0.0,
            "cost_penalty": 0.0,
            "total_reward": 0.0
        })
    
    done = False

    # 保存动作前状态
    state_before = env._get_current_state()

    # 更新时间步
    env.time_step += 1
    
    # 执行新的动态调度逻辑
    for i, truck in enumerate(env.trucks):
        action = actions[i]
        
        # 解析动作：停靠点选择
        if isinstance(action, dict):
            select_stop = action['select_stop']
            service_area = action.get('service_area', [])
        else:
            # 兼容旧格式
            select_stop = action
            service_area = []
        
        # 移动卡车到新位置
        old_location_id = truck['current_location']
        old_position = truck['position']
        
        if select_stop == 0:  # 返回仓库
            new_location_id = 0
            new_position = env.depot
            truck['current_location'] = new_location_id
            truck['position'] = new_position
            
            # 重置卡车状态，模拟补货/卸货
            truck['current_delivery_load'] = env.initial_delivery_load
            truck['current_return_load'] = 0
            truck['remaining_space'] = env.truck_capacity - truck['current_delivery_load']
            
        elif select_stop <= len(env.lockers_state):  # 移动到快递柜
            new_location_id = select_stop
            target_locker = env.get_locker(new_location_id)
            
            if target_locker and not target_locker['served']:
                new_position = target_locker['location']
                truck['current_location'] = new_location_id
                truck['position'] = new_position
                truck['visited_stops'].append(new_location_id)
                
                # 开始动态无人机调度
                serviceable_lockers = get_serviceable_lockers(
                    truck['position'], 
                    env.lockers_state,
                    env.drone_max_range / 2  # 单程最大距离
                )
                
                # 构建强化学习偏好参数
                rl_preferences = {
                    'exploration_enabled': True,
                    'learning_weight': 0.7,  # 70%的学习权重
                    'diversity_bonus': 0.3,  # 30%的多样性奖励
                    'risk_tolerance': 0.5,   # 中等风险容忍度
                    'adaptive_threshold': 0.6,  # 自适应阈值
                    'truck_id': i,  # 卡车ID用于个性化学习
                    'time_step': env.time_step,  # 当前时间步
                    'episode_progress': env.time_step / env.max_timesteps  # 训练进度
                }
                
                # 调用动态调度器（传递RL偏好）
                schedule_result = env.drone_scheduler.schedule_drones(
                    truck_location=truck['position'],
                    available_lockers=serviceable_lockers,
                    drone_range=env.drone_max_range,
                    rl_preferences=rl_preferences
                )
                
                # 执行调度并等待完成
                total_time, time_penalty = execute_drone_schedule(env, truck, schedule_result, i)
                
                # 计算奖励（包含时间窗惩罚）
                step_reward, breakdown = calculate_step_reward(schedule_result, time_penalty)
                rewards[i] = step_reward
                step_breakdowns[i] = breakdown # 保存该卡车的奖励分解
                
                # 更新卡车状态
                truck['is_servicing'] = False
                truck['service_time'] = total_time
                
                # 立即服务当前快递柜（卡车自身服务）
                if target_locker['demand_del'] > 0 or target_locker['demand_ret'] > 0:
                    env.served_delivery += target_locker['demand_del']
                    env.served_return += target_locker['demand_ret']
                    target_locker['demand_del'] = 0
                    target_locker['demand_ret'] = 0
                    target_locker['served'] = True
                    rewards[i] += 20  # 卡车直接服务奖励
                    step_breakdowns[i]['service_reward'] += 20
                    step_breakdowns[i]['total_reward'] += 20
        
        # 更新行驶距离
        distance = env._euclidean_distance(old_position, truck['position'])
        truck['total_distance'] += distance
        env.total_truck_distance += distance
        env.episode_truck_distance += distance
    
    # 将本次步骤的奖励分解保存到环境中，供外部读取
    env.last_reward_breakdown = step_breakdowns
    
    # 检查是否完成
    done = check_episode_done(env)
    
    # 更新需求和不确定性
    env._update_demand_and_handle_uncertainty()
    
    # 获取新状态和动作掩码
    next_state, action_mask = env._get_state_with_mask()
    
    return next_state, rewards, done, action_mask
