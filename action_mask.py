"""
增强的动作掩码机制
作者: Dionysus

实现多智能体协调的动作掩码，避免冲突和非法动作
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math


class ActionMaskManager:
    """
    动作掩码管理器
    
    负责生成和管理多智能体环境中的动作掩码，
    确保动作的有效性和智能体间的协调
    """
    
    def __init__(self, num_trucks: int, num_lockers: int, truck_capacity: int,
                 depot_location: Tuple[float, float], max_distance: float = 100.0):
        """
        初始化动作掩码管理器
        
        Args:
            num_trucks: 卡车数量
            num_lockers: 快递柜数量
            truck_capacity: 卡车容量
            depot_location: 仓库位置
            max_distance: 最大行驶距离
        """
        self.num_trucks = num_trucks
        self.num_lockers = num_lockers
        self.truck_capacity = truck_capacity
        self.depot_location = depot_location
        self.max_distance = max_distance
        
        # 动作空间配置
        self.stop_action_dim = num_lockers + 1  # 0:仓库, 1-n:快递柜
        self.service_action_dim = num_lockers    # 每个快递柜的服务选择
        
        # 协调机制配置
        self.coordination_enabled = True
        self.conflict_resolution = "global_optimal"  # "global_optimal", "priority", "distance", "capacity"
        
    def get_action_masks(self, env_state: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """
        获取所有卡车的动作掩码
        
        Args:
            env_state: 环境状态
            
        Returns:
            每个卡车的动作掩码列表
        """
        action_masks = []
        
        # 获取全局约束信息
        global_constraints = self._get_global_constraints(env_state)
        
        for truck_id in range(self.num_trucks):
            if truck_id < len(env_state['trucks']):
                truck_mask = self._get_truck_action_mask(
                    truck_id, env_state, global_constraints
                )
            else:
                # 不存在的卡车使用空掩码
                truck_mask = self._get_empty_mask()
            
            action_masks.append(truck_mask)
        
        # 应用协调约束
        if self.coordination_enabled:
            action_masks = self._apply_coordination_constraints(
                action_masks, env_state
            )
        
        return action_masks
    
    def _get_truck_action_mask(self, truck_id: int, env_state: Dict[str, Any],
                              global_constraints: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        获取单个卡车的动作掩码
        
        Args:
            truck_id: 卡车ID
            env_state: 环境状态
            global_constraints: 全局约束
            
        Returns:
            卡车的动作掩码
        """
        truck = env_state['trucks'][truck_id]
        
        # 初始化掩码
        stop_mask = torch.zeros(self.stop_action_dim)
        service_mask = torch.zeros(self.service_action_dim)
        
        # 如果卡车已返回，所有动作都无效
        if truck.get('returned', False):
            return {
                'stop_mask': stop_mask,
                'service_mask': service_mask
            }
        
        # 检查返回仓库的有效性
        if self._can_return_to_depot(truck, env_state):
            stop_mask[0] = 1.0
        
        # 检查每个快递柜的可达性和服务能力
        for locker_id in range(1, self.num_lockers + 1):
            locker = self._get_locker_by_id(env_state['lockers'], locker_id)
            
            if locker is None:
                continue
                
            # 检查快递柜是否可以作为停靠点
            if self._can_visit_locker(truck, locker, env_state, global_constraints):
                stop_mask[locker_id] = 1.0
                
                # 检查是否可以服务该快递柜
                if self._can_service_locker(truck, locker, env_state):
                    service_mask[locker_id - 1] = 1.0
        
        # 安全检查：确保至少有一个动作可用
        total_available = stop_mask.sum() + service_mask.sum()
        if total_available == 0:
            # 如果没有可用动作，强制允许返回仓库
            if truck['current_location'] != 0:
                stop_mask[0] = 1.0
            else:
                # 如果已在仓库，允许访问第一个快递柜
                service_mask[0] = 1.0
        
        return {
            'stop_mask': stop_mask,
            'service_mask': service_mask
        }
    
    def _can_return_to_depot(self, truck: Dict[str, Any], env_state: Dict[str, Any]) -> bool:
        """
        检查卡车是否可以返回仓库
        
        Args:
            truck: 卡车状态
            env_state: 环境状态
            
        Returns:
            是否可以返回仓库
        """
        # 如果已经在仓库，不能再返回仓库
        if truck['current_location'] == 0:
            return False
        
        # 检查是否有未服务的快递柜且卡车有能力服务
        unserved_lockers = [locker for locker in env_state['lockers'] 
                           if not locker.get('served', False) and 
                           (locker.get('demand_del', 0) > 0 or locker.get('demand_ret', 0) > 0)]
        
        # 如果还有未服务的快递柜，限制返回仓库
        if unserved_lockers:
            # 检查卡车是否已经满载或无法服务任何快递柜
            current_delivery_load = truck.get('current_delivery_load', 0)
            current_return_load = truck.get('current_return_load', 0)
            remaining_delivery_capacity = self.truck_capacity - current_delivery_load
            remaining_return_capacity = self.truck_capacity - current_return_load
            
            # 如果卡车还有容量且有可服务的快递柜，不允许返回仓库
            can_service_any = False
            for locker in unserved_lockers:
                delivery_demand = locker.get('demand_del', 0)
                return_demand = locker.get('demand_ret', 0)
                if (delivery_demand <= remaining_delivery_capacity and 
                    return_demand <= remaining_return_capacity):
                    can_service_any = True
                    break
            
            # 如果还能服务快递柜，不允许返回仓库
            if can_service_any:
                return False
        
        # 其他情况允许返回仓库
        return True
    
    def _can_visit_locker(self, truck: Dict[str, Any], locker: Dict[str, Any],
                         env_state: Dict[str, Any], global_constraints: Dict[str, Any]) -> bool:
        """
        检查卡车是否可以访问快递柜
        
        Args:
            truck: 卡车状态
            locker: 快递柜状态
            env_state: 环境状态
            global_constraints: 全局约束
            
        Returns:
            是否可以访问
        """
        # 已服务的快递柜不能再访问
        if locker.get('served', False):
            return False
        
        # 放宽距离约束：大幅增加最大距离限制
        current_pos = self._get_truck_position(truck, env_state)
        locker_pos = tuple(locker['location'])
        distance = self._euclidean_distance(current_pos, locker_pos)
        
        # 将最大距离限制增加到500，基本不限制距离
        if distance > 500.0:
            return False
        
        # 检查是否有需求
        total_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
        if total_demand <= 0:
            return False
        
        # 检查时间约束
        if not self._check_time_constraints(truck, locker, env_state):
            return False
        
        return True
    
    def _can_service_locker(self, truck: Dict[str, Any], locker: Dict[str, Any],
                           env_state: Dict[str, Any]) -> bool:
        """
        检查卡车是否可以服务快递柜
        
        Args:
            truck: 卡车状态
            locker: 快递柜状态
            env_state: 环境状态
            
        Returns:
            是否可以服务
        """
        # 检查容量约束
        delivery_demand = locker.get('demand_del', 0)
        return_demand = locker.get('demand_ret', 0)
        
        # 检查配送容量
        current_delivery_load = truck.get('current_delivery_load', 0)
        remaining_delivery_capacity = self.truck_capacity - current_delivery_load
        
        if delivery_demand > remaining_delivery_capacity:
            return False
        
        # 检查退货容量
        current_return_load = truck.get('current_return_load', 0)
        remaining_return_capacity = self.truck_capacity - current_return_load
        
        if return_demand > remaining_return_capacity:
            return False
        
        return True
    
    def _check_time_constraints(self, truck: Dict[str, Any], locker: Dict[str, Any],
                               env_state: Dict[str, Any]) -> bool:
        """
        检查时间约束
        
        Args:
            truck: 卡车状态
            locker: 快递柜状态
            env_state: 环境状态
            
        Returns:
            是否满足时间约束
        """
        current_time = env_state.get('time_step', 0)
        max_time = env_state.get('max_timesteps', 100)
        
        # 放宽时间约束：只要还有足够时间到达快递柜即可
        # 不强制要求必须能返回仓库
        remaining_time = max_time - current_time
        
        # 如果剩余时间太少（少于5步），则不允许新的访问
        if remaining_time < 5:
            return False
        
        # 简化时间检查：只要不是最后几步都允许
        return True
    
    def _apply_coordination_constraints(self, action_masks: List[Dict[str, torch.Tensor]],
                                       env_state: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """
        应用协调约束，避免冲突
        
        Args:
            action_masks: 原始动作掩码
            env_state: 环境状态
            
        Returns:
            协调后的动作掩码
        """
        if self.conflict_resolution == "priority":
            return self._apply_priority_coordination(action_masks, env_state)
        elif self.conflict_resolution == "global_optimal":
            return self._apply_global_optimal_coordination(action_masks, env_state)
        elif self.conflict_resolution == "distance":
            return self._apply_distance_coordination(action_masks, env_state)
        elif self.conflict_resolution == "capacity":
            return self._apply_capacity_coordination(action_masks, env_state)
        else:
            return action_masks
    
    def _apply_priority_coordination(self, action_masks: List[Dict[str, torch.Tensor]],
                                    env_state: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """
        基于优先级的协调机制（改进版：允许多卡车协作）
        
        Args:
            action_masks: 原始动作掩码
            env_state: 环境状态
            
        Returns:
            协调后的动作掩码
        """
        coordinated_masks = [mask.copy() for mask in action_masks]
        
        # 计算每辆卡车的总体负载和活跃度
        truck_loads = []
        truck_activities = []
        for truck_id, truck in enumerate(env_state['trucks']):
            current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
            truck_loads.append(current_load)
            # 活跃度基于访问过的站点数量
            activity = len(truck.get('visited_stops', []))
            truck_activities.append(activity)
        
        # 负载均衡：如果某辆卡车负载过重或过于活跃，降低其优先级
        avg_load = sum(truck_loads) / len(truck_loads) if truck_loads else 0
        avg_activity = sum(truck_activities) / len(truck_activities) if truck_activities else 0
        
        # 为每个快递柜分配卡车，但允许负载均衡
        for locker_id in range(1, self.num_lockers + 1):
            candidate_trucks = []
            
            # 找到所有可以访问该快递柜的卡车
            for truck_id, mask in enumerate(coordinated_masks):
                if truck_id < len(env_state['trucks']) and mask['stop_mask'][locker_id] > 0:
                    truck = env_state['trucks'][truck_id]
                    base_priority = self._calculate_truck_priority(truck, locker_id, env_state)
                    
                    # 负载均衡调整：负载过重的卡车优先级降低
                    load_penalty = 0
                    if truck_loads[truck_id] > avg_load * 1.5:
                        load_penalty = 0.3
                    elif truck_loads[truck_id] > avg_load * 1.2:
                        load_penalty = 0.1
                    
                    # 活跃度均衡：过于活跃的卡车优先级降低
                    activity_penalty = 0
                    if truck_activities[truck_id] > avg_activity * 2:
                        activity_penalty = 0.4
                    elif truck_activities[truck_id] > avg_activity * 1.5:
                        activity_penalty = 0.2
                    
                    # 调整后的优先级
                    adjusted_priority = base_priority - load_penalty - activity_penalty
                    candidate_trucks.append((truck_id, adjusted_priority))
            
            # 按优先级排序
            candidate_trucks.sort(key=lambda x: x[1], reverse=True)
            
            # 严格的一对一分配：只允许优先级最高的卡车访问
            if len(candidate_trucks) > 1:
                # 屏蔽除了优先级最高的卡车之外的所有卡车
                for truck_id, _ in candidate_trucks[1:]:
                    coordinated_masks[truck_id]['stop_mask'][locker_id] = 0.0
                    if locker_id <= self.service_action_dim:
                        coordinated_masks[truck_id]['service_mask'][locker_id - 1] = 0.0
            # 如果只有一辆卡车候选，允许访问
        
        return coordinated_masks
    
    def _calculate_truck_priority(self, truck: Dict[str, Any], locker_id: int,
                                 env_state: Dict[str, Any]) -> float:
        """
        计算卡车对特定快递柜的优先级
        
        Args:
            truck: 卡车状态
            locker_id: 快递柜ID
            env_state: 环境状态
            
        Returns:
            优先级分数（越高越优先）
        """
        import random
        
        locker = self._get_locker_by_id(env_state['lockers'], locker_id)
        if locker is None:
            return 0.0
        
        # 距离因子（距离越近优先级越高，但降低影响权重）
        current_pos = self._get_truck_position(truck, env_state)
        locker_pos = tuple(locker['location'])
        distance = self._euclidean_distance(current_pos, locker_pos)
        distance_factor = 1.0 / (distance + 1e-6)
        
        # 容量匹配因子
        delivery_demand = locker.get('demand_del', 0)
        return_demand = locker.get('demand_ret', 0)
        total_demand = delivery_demand + return_demand
        
        current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
        remaining_capacity = 2 * self.truck_capacity - current_load
        capacity_factor = min(1.0, remaining_capacity / (total_demand + 1e-6))
        
        # 负载均衡因子（负载越轻优先级越高）
        load_factor = 1.0 - (current_load / (2 * self.truck_capacity))
        
        # 工作量因子（访问过的站点越少优先级越高，鼓励负载均衡）
        visited_stops = len(truck.get('visited_stops', []))
        workload_factor = 1.0 / (1.0 + visited_stops * 0.1)
        
        # 随机因子（增加一些随机性，避免总是选择同一辆卡车）
        random_factor = random.uniform(0.8, 1.2)
        
        # 综合优先级（调整权重，更注重负载均衡）
        priority = (distance_factor * 0.2 + 
                   capacity_factor * 0.3 + 
                   load_factor * 0.25 +
                   workload_factor * 0.25) * random_factor
        
        return priority
    
    def _get_global_constraints(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取全局约束信息
        
        Args:
            env_state: 环境状态
            
        Returns:
            全局约束字典
        """
        return {
            'time_step': env_state.get('time_step', 0),
            'max_timesteps': env_state.get('max_timesteps', 100),
            'total_demand': sum(
                locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                for locker in env_state['lockers']
            ),
            'served_count': sum(
                1 for locker in env_state['lockers'] if locker.get('served', False)
            )
        }
    
    def _get_empty_mask(self) -> Dict[str, torch.Tensor]:
        """
        获取空的动作掩码
        
        Returns:
            空的动作掩码
        """
        return {
            'stop_mask': torch.zeros(self.stop_action_dim),
            'service_mask': torch.zeros(self.service_action_dim)
        }
    
    def _get_locker_by_id(self, lockers: List[Dict[str, Any]], locker_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取快递柜
        
        Args:
            lockers: 快递柜列表
            locker_id: 快递柜ID
            
        Returns:
            快递柜状态或None
        """
        for locker in lockers:
            if locker.get('id') == locker_id:
                return locker
        return None
    
    def _get_truck_position(self, truck: Dict[str, Any], env_state: Dict[str, Any]) -> Tuple[float, float]:
        """
        获取卡车当前位置
        
        Args:
            truck: 卡车状态
            env_state: 环境状态
            
        Returns:
            卡车位置坐标
        """
        current_location = truck.get('current_location', 0)
        
        if current_location == 0:
            return self.depot_location
        else:
            locker = self._get_locker_by_id(env_state['lockers'], current_location)
            if locker:
                return tuple(locker['location'])
            else:
                return self.depot_location
    
    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        计算欧几里得距离
        
        Args:
            p1: 点1坐标
            p2: 点2坐标
            
        Returns:
            欧几里得距离
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def validate_action(self, truck_id: int, action: Dict[str, Any],
                       action_mask: Dict[str, torch.Tensor]) -> bool:
        """
        验证动作的有效性
        
        Args:
            truck_id: 卡车ID
            action: 动作
            action_mask: 动作掩码
            
        Returns:
            动作是否有效
        """
        # 检查停靠点选择
        select_stop = action.get('select_stop', 0)
        if select_stop >= len(action_mask['stop_mask']) or action_mask['stop_mask'][select_stop] == 0:
            return False
        
        # 检查服务区域选择
        service_area = action.get('service_area', [])
        if isinstance(service_area, list):
            for i, service in enumerate(service_area):
                if i >= len(action_mask['service_mask']) and service > 0:
                    return False
                if service > 0 and action_mask['service_mask'][i] == 0:
                    return False
        
        return True
    
    def get_action_space_info(self) -> Dict[str, int]:
        """
        获取动作空间信息
        
        Returns:
            动作空间维度信息
        """
        return {
            'stop_action_dim': self.stop_action_dim,
            'service_action_dim': self.service_action_dim,
        }
    
    def _apply_global_optimal_coordination(self, action_masks: List[Dict[str, torch.Tensor]],
                                         env_state: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """
        应用增强的全局最优协调策略，集成多维度成本计算和智能分配
        
        Args:
            action_masks: 各卡车的动作掩码列表
            env_state: 环境状态
            
        Returns:
            协调后的动作掩码列表
        """
        trucks = env_state.get('trucks', [])
        lockers = env_state.get('lockers', [])
        
        if len(trucks) <= 1:
            return action_masks
        
        # 获取全局协调信息
        global_coord_info = env_state.get('global_coordination_info', {})
        truck_performance = env_state.get('truck_performance_metrics', {})
        load_balancing_weights = env_state.get('load_balancing_weights', {})
        
        # 构建增强的成本矩阵
        cost_matrix = []
        truck_available_lockers = []
        
        # 计算全局负载平衡因子
        avg_load = self._calculate_average_truck_load(trucks)
        
        for truck_idx, truck in enumerate(trucks):
            available_for_truck = []
            truck_costs = []
            truck_pos = self._get_truck_position(truck, env_state)
            
            # 获取卡车性能指标
            truck_perf = truck_performance.get(truck_idx, {})
            truck_efficiency = truck_perf.get('efficiency_score', 0.5)
            truck_load_util = truck_perf.get('load_utilization', 0.0)
            predicted_completion = truck_perf.get('predicted_completion_time', 0.0)
            
            # 获取负载平衡权重
            load_weight = load_balancing_weights.get(truck_idx, 1.0)
            
            for locker_idx, locker in enumerate(lockers):
                locker_id = locker['id']
                
                # 检查基本可访问性
                can_visit = (truck_idx < len(action_masks) and 
                           locker_id < len(action_masks[truck_idx]['stop_mask']) and
                           action_masks[truck_idx]['stop_mask'][locker_id] > 0.5)
                
                has_demand = (locker.get('demand_del', 0) > 0 or locker.get('demand_ret', 0) > 0) and not locker.get('served', False)
                
                if can_visit and has_demand:
                    available_for_truck.append(locker_idx)
                    
                    # 多维度成本计算
                    locker_pos = locker['location']
                    distance = self._euclidean_distance(truck_pos, locker_pos)
                    
                    # 1. 基础距离成本
                    distance_cost = distance
                    
                    # 2. 负载匹配成本
                    total_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                    current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
                    capacity_factor = abs(current_load - total_demand) / max(self.truck_capacity, 1)
                    
                    # 3. 负载平衡成本（考虑全局平衡）
                    load_deviation = abs(truck_load_util - avg_load)
                    load_balance_cost = load_deviation * load_weight * 15
                    
                    # 4. 效率成本（低效率卡车分配更近的任务）
                    efficiency_cost = (1.0 - truck_efficiency) * distance * 0.5
                    
                    # 5. 时间成本（考虑预测完成时间）
                    time_cost = predicted_completion * 0.1
                    
                    # 6. 协调冲突成本
                    conflict_cost = self._calculate_coordination_conflict_cost(
                        truck_idx, locker_idx, trucks, lockers, global_coord_info
                    )
                    
                    # 7. 需求优先级成本（紧急需求优先）
                    urgency_cost = self._calculate_urgency_cost(locker, env_state)
                    
                    # 8. 区域覆盖成本（鼓励分散覆盖）
                    coverage_cost = self._calculate_coverage_cost(
                        truck_idx, locker_idx, trucks, lockers
                    )
                    
                    # 综合成本计算
                    total_cost = (
                        distance_cost * 1.0 +
                        capacity_factor * 12 +
                        load_balance_cost * 1.2 +
                        efficiency_cost * 0.8 +
                        time_cost * 0.6 +
                        conflict_cost * 2.0 +
                        urgency_cost * 1.5 +
                        coverage_cost * 0.7
                    )
                    
                    truck_costs.append(total_cost)
                else:
                    truck_costs.append(float('inf'))
            
            cost_matrix.append(truck_costs)
            truck_available_lockers.append(available_for_truck)
        
        if not any(truck_available_lockers):
            return action_masks
        
        # 使用增强的匈牙利算法进行最优分配
        assignments = self._enhanced_hungarian_assignment(cost_matrix, env_state)
        
        # 根据分配结果更新动作掩码
        coordinated_masks = [mask.copy() for mask in action_masks]
        
        # 创建分配映射并应用软约束
        truck_assignments = {}
        for truck_idx, locker_idx in enumerate(assignments):
            if locker_idx is not None and truck_idx < len(trucks) and locker_idx < len(lockers):
                locker_id = lockers[locker_idx]['id']
                truck_assignments[truck_idx] = locker_id
        
        # 应用分配结果到动作掩码
        self._apply_assignment_to_masks(
            coordinated_masks, truck_assignments, trucks, lockers, env_state
        )
        
        return coordinated_masks
    
    def _hungarian_assignment(self, cost_matrix: List[List[float]]) -> List[Optional[int]]:
        """
        简化的匈牙利算法实现，用于最优分配
        
        Args:
            cost_matrix: 成本矩阵，cost_matrix[i][j]表示卡车i到快递柜j的成本
            
        Returns:
            分配结果，assignments[i]表示卡车i分配到的快递柜索引，None表示未分配
        """
        if not cost_matrix or not cost_matrix[0]:
            return []
        
        num_trucks = len(cost_matrix)
        num_lockers = len(cost_matrix[0])
        
        # 简化版本：贪心分配，每次选择成本最小的未分配组合
        assignments = [None] * num_trucks
        assigned_lockers = set()
        
        # 创建所有可能的(卡车, 快递柜, 成本)组合
        all_combinations = []
        for truck_idx in range(num_trucks):
            for locker_idx in range(num_lockers):
                cost = cost_matrix[truck_idx][locker_idx]
                all_combinations.append((truck_idx, locker_idx, cost))
        
        # 按成本排序
        all_combinations.sort(key=lambda x: x[2])
        
        # 贪心分配
        for truck_idx, locker_idx, cost in all_combinations:
            if assignments[truck_idx] is None and locker_idx not in assigned_lockers:
                assignments[truck_idx] = locker_idx
                assigned_lockers.add(locker_idx)
        
        return assignments
    
    def _calculate_average_truck_load(self, trucks: List[Dict[str, Any]]) -> float:
        """
        计算卡车平均负载率
        
        Args:
            trucks: 卡车列表
            
        Returns:
            float: 平均负载率
        """
        if not trucks or self.truck_capacity == 0:
            return 0.0
        
        total_load_ratio = 0.0
        for truck in trucks:
            current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
            load_ratio = current_load / self.truck_capacity
            total_load_ratio += load_ratio
        
        return total_load_ratio / len(trucks)
    
    def _calculate_coordination_conflict_cost(self, truck_idx: int, locker_idx: int,
                                            trucks: List[Dict[str, Any]], lockers: List[Dict[str, Any]],
                                            global_coord_info: Dict[str, Any]) -> float:
        """
        计算协调冲突成本
        
        Args:
            truck_idx: 卡车索引
            locker_idx: 快递柜索引
            trucks: 卡车列表
            lockers: 快递柜列表
            global_coord_info: 全局协调信息
            
        Returns:
            float: 冲突成本
        """
        conflicts = global_coord_info.get('conflicts', [])
        conflict_cost = 0.0
        
        # 检查目标冲突
        for conflict in conflicts:
            if conflict.get('type') == 'target_conflict':
                if (truck_idx in conflict.get('truck_ids', []) and 
                    conflict.get('locker_id') == lockers[locker_idx]['id']):
                    conflict_cost += conflict.get('severity', 0) * 20
            
            elif conflict.get('type') == 'proximity_conflict':
                if truck_idx in conflict.get('truck_ids', []):
                    conflict_cost += conflict.get('severity', 0) * 10
        
        return conflict_cost
    
    def _calculate_urgency_cost(self, locker: Dict[str, Any], env_state: Dict[str, Any]) -> float:
        """
        计算需求紧急性成本
        
        Args:
            locker: 快递柜信息
            env_state: 环境状态
            
        Returns:
            float: 紧急性成本（负值表示高优先级）
        """
        # 基于需求量和等待时间计算紧急性
        delivery_demand = locker.get('demand_del', 0)
        return_demand = locker.get('demand_ret', 0)
        total_demand = delivery_demand + return_demand
        
        # 等待时间（简化：基于时间步）
        current_time = env_state.get('time_step', 0)
        last_service_time = locker.get('last_service_time', 0)
        waiting_time = current_time - last_service_time
        
        # 紧急性分数：需求越多、等待越久，成本越低（优先级越高）
        urgency_score = total_demand * 2 + waiting_time * 0.1
        
        # 返回负成本（高紧急性 = 低成本）
        return -urgency_score
    
    def _calculate_coverage_cost(self, truck_idx: int, locker_idx: int,
                               trucks: List[Dict[str, Any]], lockers: List[Dict[str, Any]]) -> float:
        """
        计算区域覆盖成本，鼓励卡车分散到不同区域
        
        Args:
            truck_idx: 卡车索引
            locker_idx: 快递柜索引
            trucks: 卡车列表
            lockers: 快递柜列表
            
        Returns:
            float: 覆盖成本
        """
        target_locker = lockers[locker_idx]
        target_pos = target_locker['location']
        
        # 计算其他卡车到目标位置的距离
        coverage_penalty = 0.0
        for other_idx, other_truck in enumerate(trucks):
            if other_idx != truck_idx and other_truck['current_location'] != 0:
                # 获取其他卡车的位置
                other_pos = self._get_truck_position(other_truck, {'lockers': lockers})
                distance = self._euclidean_distance(other_pos, target_pos)
                
                # 如果其他卡车很近，增加覆盖成本
                if distance < 30.0:  # 30单位范围内
                    coverage_penalty += (30.0 - distance) / 30.0 * 5
        
        return coverage_penalty
    
    def _enhanced_hungarian_assignment(self, cost_matrix: List[List[float]], 
                                     env_state: Dict[str, Any]) -> List[Optional[int]]:
        """
        增强的匈牙利算法实现，考虑多轮优化和约束
        
        Args:
            cost_matrix: 成本矩阵
            env_state: 环境状态
            
        Returns:
            分配结果
        """
        if not cost_matrix or not cost_matrix[0]:
            return []
        
        num_trucks = len(cost_matrix)
        num_lockers = len(cost_matrix[0])
        
        # 第一轮：基础贪心分配
        assignments = [None] * num_trucks
        assigned_lockers = set()
        
        # 创建所有可能的(卡车, 快递柜, 成本)组合
        all_combinations = []
        for truck_idx in range(num_trucks):
            for locker_idx in range(num_lockers):
                cost = cost_matrix[truck_idx][locker_idx]
                if cost != float('inf'):
                    all_combinations.append((truck_idx, locker_idx, cost))
        
        # 按成本排序
        all_combinations.sort(key=lambda x: x[2])
        
        # 贪心分配
        for truck_idx, locker_idx, cost in all_combinations:
            if assignments[truck_idx] is None and locker_idx not in assigned_lockers:
                assignments[truck_idx] = locker_idx
                assigned_lockers.add(locker_idx)
        
        # 第二轮：局部优化（2-opt交换）
        assignments = self._local_optimization(assignments, cost_matrix)
        
        return assignments
    
    def _local_optimization(self, assignments: List[Optional[int]], 
                          cost_matrix: List[List[float]]) -> List[Optional[int]]:
        """
        局部优化：尝试2-opt交换改善分配
        
        Args:
            assignments: 当前分配
            cost_matrix: 成本矩阵
            
        Returns:
            优化后的分配
        """
        improved = True
        max_iterations = 10
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(assignments)):
                for j in range(i + 1, len(assignments)):
                    if assignments[i] is not None and assignments[j] is not None:
                        # 计算当前成本
                        current_cost = (cost_matrix[i][assignments[i]] + 
                                      cost_matrix[j][assignments[j]])
                        
                        # 计算交换后成本
                        swap_cost = (cost_matrix[i][assignments[j]] + 
                                   cost_matrix[j][assignments[i]])
                        
                        # 如果交换能改善，则执行交换
                        if swap_cost < current_cost:
                            assignments[i], assignments[j] = assignments[j], assignments[i]
                            improved = True
        
        return assignments
    
    def _apply_assignment_to_masks(self, coordinated_masks: List[Dict[str, torch.Tensor]],
                                 truck_assignments: Dict[int, int],
                                 trucks: List[Dict[str, Any]], lockers: List[Dict[str, Any]],
                                 env_state: Dict[str, Any]) -> None:
        """
        将分配结果应用到动作掩码，支持软约束和优先级
        
        Args:
            coordinated_masks: 协调后的动作掩码列表
            truck_assignments: 卡车分配映射
            trucks: 卡车列表
            lockers: 快递柜列表
            env_state: 环境状态
        """
        # 获取协调强度设置
        coordination_strength = env_state.get('coordination_strength', 0.8)  # 0.0-1.0
        
        # 对于所有有需求的快递柜，应用分配约束
        for locker in lockers:
            locker_id = locker['id']
            has_demand = (locker.get('demand_del', 0) > 0 or locker.get('demand_ret', 0) > 0) and not locker.get('served', False)
            
            if has_demand:
                # 找到被分配给这个快递柜的卡车
                assigned_truck = None
                for truck_idx, assigned_locker_id in truck_assignments.items():
                    if assigned_locker_id == locker_id:
                        assigned_truck = truck_idx
                        break
                
                # 应用软约束：降低非分配卡车的访问概率，而不是完全屏蔽
                for truck_idx in range(len(trucks)):
                    if truck_idx < len(coordinated_masks):
                        if truck_idx == assigned_truck:
                            # 分配的卡车：保持或增强访问权限
                            if locker_id < len(coordinated_masks[truck_idx]['stop_mask']):
                                coordinated_masks[truck_idx]['stop_mask'][locker_id] = 1.0
                            if locker_id <= self.service_action_dim:
                                coordinated_masks[truck_idx]['service_mask'][locker_id - 1] = 1.0
                        else:
                            # 非分配的卡车：根据协调强度降低访问权限
                            penalty_factor = coordination_strength
                            if locker_id < len(coordinated_masks[truck_idx]['stop_mask']):
                                current_value = coordinated_masks[truck_idx]['stop_mask'][locker_id]
                                coordinated_masks[truck_idx]['stop_mask'][locker_id] = current_value * (1.0 - penalty_factor)
                            if locker_id <= self.service_action_dim:
                                current_value = coordinated_masks[truck_idx]['service_mask'][locker_id - 1]
                                coordinated_masks[truck_idx]['service_mask'][locker_id - 1] = current_value * (1.0 - penalty_factor)