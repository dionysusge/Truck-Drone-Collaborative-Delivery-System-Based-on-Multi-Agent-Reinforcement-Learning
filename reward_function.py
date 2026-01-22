#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化整合的奖励函数模块
作者: Dionysus
联系方式: wechat:gzw1546484791

该模块实现了简化的奖励结构，将复杂的多组件奖励整合为5个核心类别，
突出重点，提高训练效率和可解释性。
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from config import Config


class RewardFunction:
    """
    简化整合的奖励函数
    
    核心设计原则:
    1. 突出主要业务目标：服务完成
    2. 平衡效率与成本
    3. 简化复杂度，提高可解释性
    4. 保持训练稳定性
    """
    
    def __init__(self, max_timesteps: int = 100):
        """
        初始化奖励函数
        
        参数:
        - max_timesteps: 最大时间步数
        """
        self.max_timesteps = max_timesteps
        
        # 优化的奖励权重结构 - 强化路径效率和步数优化
        self.weights = {
            # === 1. 服务完成奖励 (降低权重，减少对完成率的依赖) ===
            'service_completion': 0,          # 降低服务完成奖励
            'demand_satisfaction': 0.001,         # 降低需求满足权重
            'early_completion_bonus': 5.0,       # 增加快速完成奖励，鼓励用更少步数完成
            
            # === 2. 运营效率奖励 (大幅提高路径效率权重) ===
            'operational_efficiency': 60.0,      # 大幅提高路径效率权重
            'demand_weighted_efficiency': 50.0,  # 大幅提高需求量加权路径效率专项奖励
            'resource_utilization': 15.0,        # 提高资源利用权重
            'coordination_bonus': 100.0,          # 提高协调奖励
            
            # === 3. 成本控制惩罚 (强化步数效率，增强长距离移动惩罚) ===
            'travel_cost_penalty': 12.0,         # 增强长距离移动惩罚
            'time_cost_penalty': 3.0,            # 增强时间惩罚
            'step_penalty': 25.0,                # 大幅增加步数效率奖励权重
            
            # === 4. 约束违规惩罚 (保持适度惩罚) ===
            'constraint_violation': 3.0,         # 保持适度惩罚
            'invalid_action_penalty': 2.0,       # 保持适度惩罚无效动作
            'unserved_demand_penalty': 1.0,      # 保持未服务惩罚
            
            # === 5. 策略优化奖励 (大幅提高权重，鼓励智能路径规划) ===
            'strategic_decision': 35.0,          # 大幅提高智能路径规划奖励
            'coverage_optimization': 30.0,       # 大幅提高覆盖优化奖励
        }
        
        # 统计信息
        self.episode_stats = {
            'total_reward': 0.0,
            'service_rewards': 0.0,
            'efficiency_rewards': 0.0,
            'cost_penalties': 0.0,
            'violation_penalties': 0.0,
            'strategy_rewards': 0.0,
            'reward_count': 0
        }
        
    def calculate_reward(self, 
                        action: Dict[str, Any], 
                        state_before: Dict[str, Any], 
                        state_after: Dict[str, Any], 
                        done: bool,
                        truck_id: int,
                        timestep: int) -> Tuple[float, Dict[str, float]]:
        """
        计算综合奖励 - 简化版本
        
        参数:
        - action: 执行的动作
        - state_before: 动作前状态
        - state_after: 动作后状态
        - done: 是否结束
        - truck_id: 卡车ID
        - timestep: 当前时间步
        
        返回:
        - total_reward: 总奖励
        - reward_breakdown: 奖励分解
        """
        reward_breakdown = {}
        
        # 1. 服务完成奖励
        service_completion = self._calculate_service_completion_reward(
            state_before, state_after, action, timestep, done, truck_id
        )
        reward_breakdown['service_completion'] = service_completion
        
        # 2. 运营效率奖励 (整合效率相关)
        efficiency_reward = self._calculate_operational_efficiency_reward(
            action, state_before, state_after, timestep
        )
        reward_breakdown['operational_efficiency'] = efficiency_reward
        
        # 3. 成本控制惩罚 (整合成本相关)
        cost_penalty = self._calculate_cost_control_penalty(
            action, state_before, state_after, timestep
        )
        reward_breakdown['cost_control'] = cost_penalty
        
        # 4. 约束违规惩罚 (整合违规相关)
        violation_penalty = self._calculate_constraint_violation_penalty(
            action, state_before, state_after, done
        )
        reward_breakdown['constraint_violation'] = violation_penalty
        
        # 5. 策略优化奖励 (整合智能决策相关)
        strategy_reward = self._calculate_strategic_optimization_reward(
            action, state_before, state_after
        )
        reward_breakdown['strategic_optimization'] = strategy_reward

        # 计算总奖励
        total_reward = sum(reward_breakdown.values())

        # 更新统计信息
        self._update_stats(total_reward, reward_breakdown)

        return total_reward, reward_breakdown
    
    def _calculate_service_completion_reward(self, 
                                           state_before: Dict[str, Any], 
                                           state_after: Dict[str, Any],
                                           action: Dict[str, Any],
                                           timestep: int,
                                           done: bool,
                                           truck_id: int = 0) -> float:
        """
        计算服务完成奖励 (整合原有的服务相关奖励)
        
        整合内容:
        - 无人机配送完成奖励
        - 需求点满足奖励
        - 提前完成奖励
        """
        reward = 0.0
        
        # 基础服务完成奖励 (只奖励新完成的服务)
        if action.get('action_type') == 'service':
            # 检查是否有新的服务完成
            served_before = sum(1 for locker in state_before.get('lockers', []) 
                              if locker.get('served', False))
            served_after = sum(1 for locker in state_after.get('lockers', []) 
                             if locker.get('served', False))
            
            new_services = served_after - served_before
            if new_services > 0:
                reward += self.weights['service_completion'] * new_services
                
                # 需求满足奖励 (只计算新服务的需求，避免累积)
                new_demand_satisfied = 0
                for locker in state_after.get('lockers', []):
                    # 只计算新服务的快递柜需求
                    if locker.get('served', False):
                        # 检查这个快递柜是否是新服务的
                        was_served_before = False
                        for old_locker in state_before.get('lockers', []):
                            if (old_locker.get('location') == locker.get('location') and 
                                old_locker.get('served', False)):
                                was_served_before = True
                                break
                        
                        if not was_served_before:
                            new_demand_satisfied += locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                
                # 按新服务的平均需求给奖励，而不是累积总需求
                if new_services > 0:
                    avg_demand_per_service = new_demand_satisfied / new_services
                    reward += self.weights['demand_satisfaction'] * avg_demand_per_service * 0.1  # 降低10倍
        
        # 提前完成奖励 (只给truck_0计算，避免多卡车重复计算)
        if done and timestep < self.max_timesteps * 0.8 and truck_id == 0:
            completion_ratio = 1.0 - (timestep / self.max_timesteps)
            reward += self.weights['early_completion_bonus'] * completion_ratio * 0.05  # 降低20倍
        
        return reward
    
    def _calculate_operational_efficiency_reward(self, 
                                               action: Dict[str, Any], 
                                               state_before: Dict[str, Any], 
                                               state_after: Dict[str, Any],
                                               timestep: int) -> float:
        """
        计算运营效率奖励 (重点奖励路径效率和快速完成)
        
        重点优化内容:
        - 路径效率 (大幅提高权重)
        - 服务密度奖励
        - 距离效率奖励
        - 快速决策奖励
        """
        reward = 0.0
        
        # 1. 需求量加权路径效率奖励 (全新的核心奖励机制)
        if action.get('action_type') == 'move':
            travel_distance = action.get('travel_distance', 0)
            truck_pos = action.get('truck_position', (0, 0))
            nearby_unserved = 0
            total_nearby_demand = 0
            demand_weighted_score = 0.0
            
            for locker in state_after.get('lockers', []):
                if not locker.get('served', False):
                    locker_pos = locker.get('location', (0, 0))
                    distance = self._euclidean_distance(truck_pos, locker_pos)
                    if distance <= 30.0:  # 无人机服务范围
                        nearby_unserved += 1
                        locker_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                        total_nearby_demand += locker_demand
                        
                        # 需求量加权评分：高需求快递柜获得更高权重
                        if locker_demand > 0:
                            demand_weight = min(locker_demand / 3.0, 2.0)  # 需求量权重，最高2倍
                            demand_weighted_score += demand_weight
            
            if nearby_unserved > 0 and travel_distance > 0:
                # 传统服务密度效率
                density_efficiency = nearby_unserved / max(travel_distance, 0.1)
                reward += self.weights['operational_efficiency'] * density_efficiency * 0.08
                
                # 需求密度效率
                demand_efficiency = total_nearby_demand / max(travel_distance, 0.1)
                reward += self.weights['operational_efficiency'] * demand_efficiency * 0.04
                
                # 核心：需求量加权路径效率奖励
                demand_weighted_efficiency = demand_weighted_score / max(travel_distance, 0.1)
                reward += self.weights['demand_weighted_efficiency'] * demand_weighted_efficiency
                
                # 智能距离惩罚：根据目标价值调整惩罚
                if travel_distance > 15.0:
                    # 如果目标区域需求量高，减少距离惩罚
                    demand_factor = min(total_nearby_demand / 10.0, 1.0)  # 需求因子
                    base_penalty = (travel_distance - 15.0) * 0.15
                    adjusted_penalty = base_penalty * (1.0 - demand_factor * 0.5)  # 高需求区域减少50%惩罚
                    reward -= adjusted_penalty
        
        # 2. 大幅增强的服务效率奖励
        if action.get('action_type') == 'service':
            services_completed = action.get('services_completed', 0)
            if services_completed > 0:
                # 多服务奖励：一次性服务多个快递柜的大幅奖励
                multi_service_bonus = (services_completed - 1) * 3.0  # 大幅提高倍数
                reward += multi_service_bonus
                
                # 站点停留奖励：鼓励在同一位置完成更多服务
                if services_completed >= 3:  # 一次服务3个或以上快递柜
                    station_efficiency_bonus = services_completed * 2.0  # 站点效率奖励
                    reward += station_efficiency_bonus
        
        # 2. 资源利用效率
        trucks = state_after.get('trucks', [])
        if trucks:
            total_capacity = sum(truck.get('capacity', 100) for truck in trucks)
            total_load = sum(truck.get('current_load', 0) for truck in trucks)
            utilization = total_load / max(total_capacity, 1)
            
            # 适中的利用率最优 (60%-90%)
            if 0.6 <= utilization <= 0.9:
                reward += self.weights['resource_utilization'] * utilization
            elif utilization > 0.9:
                reward += self.weights['resource_utilization'] * 0.5  # 过载惩罚
        
        # 3. 智能协作奖励 (与需求量加权效率协同)
        if action.get('action_type') == 'service':
            # 检查无人机协调情况
            active_drones = action.get('active_drones', 0)
            services_completed = action.get('services_completed', 0)
            
            if active_drones > 0:
                # 需求量加权的基础协调奖励
                total_service_demand = 0
                for service in action.get('completed_services', []):
                    total_service_demand += service.get('demand', 1)
                
                demand_factor = min(total_service_demand / 5.0, 2.0)  # 需求因子，最高2倍
                base_coordination_score = min(active_drones / 3.0, 1.0) * demand_factor
                reward += self.weights['coordination_bonus'] * base_coordination_score
                
                # 需求量加权多无人机协作奖励
                if active_drones >= 2:
                    multi_drone_bonus = (active_drones - 1) * 4.0 * demand_factor
                    reward += multi_drone_bonus
                
                # 高效协作奖励：考虑需求密度的协同效应
                if active_drones >= 2 and services_completed >= 2:
                    # 计算平均服务需求量
                    avg_service_demand = total_service_demand / max(services_completed, 1)
                    demand_multiplier = min(avg_service_demand / 2.0, 1.8)
                    synergy_bonus = active_drones * services_completed * 1.8 * demand_multiplier
                    reward += synergy_bonus
        
        return reward
    
    def _calculate_cost_control_penalty(self, 
                                      action: Dict[str, Any], 
                                      state_before: Dict[str, Any], 
                                      state_after: Dict[str, Any],
                                      timestep: int) -> float:
        """
        计算成本控制惩罚 (大幅增强对长距离移动的惩罚)
        
        优化重点:
        - 大幅增强卡车长距离移动惩罚
        - 鼓励卡车在同一区域停留更久
        - 渐进式距离惩罚机制
        - 降低时间和步数惩罚
        """
        penalty = 0.0
        
        # 1. 增强的行驶成本惩罚 - 渐进式惩罚机制
        travel_distance = action.get('travel_distance', 0)
        if travel_distance > 0:
            # 基础距离惩罚
            base_penalty = self.weights['travel_cost_penalty'] * travel_distance
            
            # 渐进式惩罚：距离越长，惩罚越重
            if travel_distance > 15.0:  # 超过15单位的移动
                extra_penalty = (travel_distance - 15.0) * 2.0  # 额外惩罚
                penalty -= base_penalty + extra_penalty
            elif travel_distance > 10.0:  # 超过10单位的移动
                extra_penalty = (travel_distance - 10.0) * 1.5  # 中等额外惩罚
                penalty -= base_penalty + extra_penalty
            else:
                penalty -= base_penalty
        
        # 2. 无人机使用成本（保持原有逻辑）
        drone_distance = action.get('drone_total_distance', 0)
        if drone_distance > 0:
            # 无人机成本相对较低，鼓励使用无人机而非卡车移动
            penalty -= self.weights['travel_cost_penalty'] * drone_distance * 0.5
        
        # 3. 降低时间成本惩罚（鼓励在同一位置停留）
        penalty -= self.weights['time_cost_penalty'] * timestep * 0.5
        
        # 4. 降低步数惩罚（不过分催促快速完成）
        penalty -= self.weights['step_penalty'] * 0.5
        
        return penalty
    
    def _calculate_constraint_violation_penalty(self, 
                                              action: Dict[str, Any], 
                                              state_before: Dict[str, Any], 
                                              state_after: Dict[str, Any],
                                              done: bool) -> float:
        """
        计算约束违规惩罚 (整合违规相关惩罚)
        
        整合内容:
        - 容量违规
        - 无效动作
        - 未服务需求
        - 不必要的返回仓库
        """
        penalty = 0.0
        
        # 1. 容量违规检查
        trucks = state_after.get('trucks', [])
        for truck in trucks:
            current_load = truck.get('current_load', 0)
            capacity = truck.get('capacity', 100)
            if current_load > capacity:
                penalty -= self.weights['constraint_violation'] * (current_load - capacity)
        
        # 2. 无效动作惩罚
        if action.get('is_invalid', False):
            penalty -= self.weights['invalid_action_penalty']
        
        # 3. 未服务需求惩罚 (终止时)
        if done:
            unserved_count = sum(1 for locker in state_after.get('lockers', []) 
                               if not locker.get('served', False))
            if unserved_count > 0:
                penalty -= self.weights['unserved_demand_penalty'] * unserved_count
        
        # 4. 不必要返回仓库惩罚
        if action.get('action_type') == 'return_depot':
            # 检查是否还有未服务的需求
            unserved_count = sum(1 for locker in state_before.get('lockers', []) 
                               if not locker.get('served', False))
            if unserved_count > 0:
                penalty -= self.weights['constraint_violation'] * 0.5
        
        return penalty
    
    def _calculate_strategic_optimization_reward(self, 
                                               action: Dict[str, Any], 
                                               state_before: Dict[str, Any], 
                                               state_after: Dict[str, Any]) -> float:
        """
        计算策略优化奖励 (整合智能决策相关奖励)
        
        整合内容:
        - 战略位置选择
        - 覆盖优化
        - 密集区域优先
        - 热点服务
        """
        reward = 0.0
        
        # 1. 战略决策奖励
        if action.get('action_type') == 'move':
            truck_pos = action.get('truck_position', (0, 0))
            
            # 计算位置的战略价值
            strategic_value = self._calculate_position_strategic_value(truck_pos, state_after)
            reward += self.weights['strategic_decision'] * strategic_value
        
        # 2. 覆盖优化奖励
        if action.get('action_type') == 'service':
            # 计算服务覆盖效率
            coverage_efficiency = self._calculate_coverage_efficiency(action, state_after)
            reward += self.weights['coverage_optimization'] * coverage_efficiency
        
        return reward
    
    def _calculate_position_strategic_value(self, 
                                          position: Tuple[float, float], 
                                          state: Dict[str, Any]) -> float:
        """
        计算位置的战略价值
        
        考虑因素:
        - 周围未服务快递柜数量
        - 需求密度
        - 覆盖效率
        """
        value = 0.0
        nearby_unserved = 0
        total_demand = 0
        
        for locker in state.get('lockers', []):
            if not locker.get('served', False):
                locker_pos = locker.get('location', (0, 0))
                distance = self._euclidean_distance(position, locker_pos)
                
                if distance <= 30.0:  # 无人机服务范围
                    nearby_unserved += 1
                    total_demand += locker.get('demand_del', 0) + locker.get('demand_ret', 0)
        
        if nearby_unserved > 0:
            # 综合考虑数量和需求
            value = (nearby_unserved * 0.6 + total_demand * 0.4) / 10.0
        
        return min(value, 1.0)  # 限制在[0,1]范围内
    
    def _calculate_coverage_efficiency(self, 
                                     action: Dict[str, Any], 
                                     state: Dict[str, Any]) -> float:
        """
        计算覆盖效率
        
        考虑因素:
        - 一次服务覆盖的快递柜数量
        - 服务的需求总量
        """
        efficiency = 0.0
        
        served_count = action.get('served_lockers_count', 0)
        total_demand_served = action.get('total_demand_served', 0)
        
        if served_count > 0:
            # 基础覆盖效率
            efficiency = served_count / 5.0  # 假设最多一次服务5个快递柜
            
            # 需求量加权
            if total_demand_served > 0:
                efficiency += total_demand_served / 50.0  # 需求量归一化
        
        return min(efficiency, 1.0)  # 限制在[0,1]范围内
    
    def _euclidean_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _update_stats(self, total_reward: float, reward_breakdown: Dict[str, float]):
        """更新统计信息"""
        self.episode_stats['total_reward'] += total_reward
        self.episode_stats['reward_count'] += 1
        
        # 分类统计
        if 'service_completion' in reward_breakdown:
            self.episode_stats['service_rewards'] += reward_breakdown['service_completion']
        
        if 'operational_efficiency' in reward_breakdown:
            self.episode_stats['efficiency_rewards'] += reward_breakdown['operational_efficiency']
        
        if 'cost_control' in reward_breakdown:
            self.episode_stats['cost_penalties'] += reward_breakdown['cost_control']
        
        if 'constraint_violation' in reward_breakdown:
            self.episode_stats['violation_penalties'] += reward_breakdown['constraint_violation']
        
        if 'strategic_optimization' in reward_breakdown:
            self.episode_stats['strategy_rewards'] += reward_breakdown['strategic_optimization']
    
    def reset_episode_stats(self):
        """重置回合统计"""
        self.episode_stats = {
            'total_reward': 0.0,
            'service_rewards': 0.0,
            'efficiency_rewards': 0.0,
            'cost_penalties': 0.0,
            'violation_penalties': 0.0,
            'strategy_rewards': 0.0,
            'reward_count': 0
        }
    
    def get_episode_stats(self) -> Dict[str, float]:
        """获取回合统计信息"""
        return self.episode_stats.copy()


class AdaptiveRewardScheduler:
    """
    简化的自适应奖励调度器
    
    只调整核心权重，减少复杂度
    """
    
    def __init__(self, reward_function: RewardFunction):
        """初始化调度器"""
        self.reward_function = reward_function
        self.initial_weights = reward_function.weights.copy()
        self.weight_history = []
        self.performance_history = []
        self.adjustment_factor = 0.1
        
    def update_weights(self, episode_performance: Dict[str, float]):
        """
        更新权重 - 简化版本
        
        参数:
        - episode_performance: 回合性能指标
        """
        self.performance_history.append(episode_performance)
        
        # 保持最近100个回合的历史
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # 每10个回合调整一次权重
        if len(self.performance_history) % 10 == 0:
            self._adjust_core_weights()
    
    def _adjust_core_weights(self):
        """调整核心权重"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = self.performance_history[-10:]
        avg_completion_rate = np.mean([p.get('completion_rate', 0) for p in recent_performance])
        avg_efficiency = np.mean([p.get('efficiency', 0) for p in recent_performance])
        avg_total_reward = np.mean([p.get('total_reward', 0) for p in recent_performance])
        
        # 根据性能调整权重
        if avg_completion_rate < 0.7:
            # 完成率低，增加服务奖励
            self.reward_function.weights['service_completion'] *= (1 + self.adjustment_factor)
            self.reward_function.weights['demand_satisfaction'] *= (1 + self.adjustment_factor)
        
        if avg_efficiency < 0.5:
            # 效率低，增加效率奖励
            self.reward_function.weights['operational_efficiency'] *= (1 + self.adjustment_factor)
            self.reward_function.weights['strategic_decision'] *= (1 + self.adjustment_factor)
        
        if avg_total_reward < 0:
            # 总奖励为负，减少惩罚
            self.reward_function.weights['travel_cost_penalty'] *= (1 - self.adjustment_factor)
            self.reward_function.weights['step_penalty'] *= (1 - self.adjustment_factor)
        
        # 限制权重范围
        self._clamp_weights()
    
    def _clamp_weights(self):
        """限制权重范围"""
        for key, initial_value in self.initial_weights.items():
            current_value = self.reward_function.weights[key]
            # 限制在初始值的0.5-2.0倍之间
            min_value = initial_value * 0.5
            max_value = initial_value * 2.0
            self.reward_function.weights[key] = np.clip(current_value, min_value, max_value)
    
    def get_weight_changes(self) -> Dict[str, float]:
        """获取权重变化"""
        changes = {}
        for key in self.initial_weights:
            initial = self.initial_weights[key]
            current = self.reward_function.weights[key]
            changes[key] = (current - initial) / initial if initial != 0 else 0
        return changes
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        获取权重稳定性指标
        
        Returns:
            Dict[str, float]: 包含权重变化稳定性的指标
        """
        if len(self.weight_history) < 2:
            return {
                'weight_variance': 0.0,
                'weight_stability': 1.0,
                'adjustment_frequency': 0.0
            }
        
        # 计算权重变化方差
        weight_changes = []
        for i in range(1, len(self.weight_history)):
            prev_weights = self.weight_history[i-1]
            curr_weights = self.weight_history[i]
            
            total_change = 0.0
            for key in prev_weights:
                if key in curr_weights:
                    total_change += abs(curr_weights[key] - prev_weights[key])
            weight_changes.append(total_change)
        
        weight_variance = np.var(weight_changes) if weight_changes else 0.0
        weight_stability = 1.0 / (1.0 + weight_variance)  # 稳定性指标
        adjustment_frequency = len([c for c in weight_changes if c > 0.1]) / len(weight_changes)
        
        return {
            'weight_variance': float(weight_variance),
            'weight_stability': float(weight_stability),
            'adjustment_frequency': float(adjustment_frequency)
        }

    def reset_scheduler(self):
        """重置调度器"""
        self.reward_function.weights = self.initial_weights.copy()
        self.weight_history.clear()
        self.performance_history.clear()
        print("自适应奖励调度器已重置")