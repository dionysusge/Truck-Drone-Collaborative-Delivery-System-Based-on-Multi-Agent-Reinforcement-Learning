"""
智能卡车补货决策模块

该模块实现了智能补货策略，包括：
1. 补货时机判断
2. 补货路径优化
3. 补货与配送任务平衡
4. 动态补货策略调整
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ReplenishmentTrigger(Enum):
    """补货触发条件类型"""
    CAPACITY_THRESHOLD = "capacity_threshold"  # 容量阈值触发
    DEMAND_PREDICTION = "demand_prediction"    # 需求预测触发
    DISTANCE_OPTIMIZATION = "distance_optimization"  # 距离优化触发
    TIME_WINDOW = "time_window"               # 时间窗触发
    EMERGENCY = "emergency"                   # 紧急补货


class ReplenishmentStrategy(Enum):
    """补货策略类型"""
    CONSERVATIVE = "conservative"  # 保守策略：早补货，避免风险
    AGGRESSIVE = "aggressive"      # 激进策略：晚补货，最大化效率
    BALANCED = "balanced"          # 平衡策略：综合考虑风险和效率
    ADAPTIVE = "adaptive"          # 自适应策略：根据环境动态调整


@dataclass
class ReplenishmentDecision:
    """补货决策结果"""
    should_replenish: bool
    trigger_reason: ReplenishmentTrigger
    urgency_level: float  # 0-1，紧急程度
    recommended_route: List[int]  # 推荐的补货路径
    expected_benefit: float  # 预期收益
    risk_assessment: float  # 风险评估
    confidence: float  # 决策置信度


@dataclass
class TruckState:
    truck_id: int
    current_location: int
    current_delivery_load: int
    current_return_load: int
    remaining_capacity: int
    total_distance: float
    visited_stops: List[int]
    returned: bool


@dataclass
class LockerDemand:
    """快递柜需求信息"""
    locker_id: int
    location: Tuple[float, float]
    delivery_demand: float
    return_demand: float
    served: bool
    priority: float
    time_window_start: float
    time_window_end: float


class ReplenishmentOptimizer:
    """补货优化器"""
    
    def __init__(self, 
                 truck_capacity: int = 300,
                 depot_location: Tuple[float, float] = (0, 0),
                 strategy: ReplenishmentStrategy = ReplenishmentStrategy.BALANCED):
        """
        初始化补货优化器
        
        Args:
            truck_capacity: 卡车容量
            depot_location: 仓库位置
            strategy: 补货策略
        """
        self.truck_capacity = truck_capacity
        self.depot_location = depot_location
        self.strategy = strategy
        
        # 策略参数
        self.strategy_params = self._initialize_strategy_params()
        
        # 历史数据
        self.replenishment_history = []
        self.performance_metrics = {
            'total_replenishments': 0,
            'successful_replenishments': 0,
            'average_efficiency': 0.0,
            'total_distance_saved': 0.0
        }
    
    def _initialize_strategy_params(self) -> Dict[str, float]:
        """初始化策略参数"""
        if self.strategy == ReplenishmentStrategy.CONSERVATIVE:
            return {
                'capacity_threshold': 0.3,  # 30%容量时触发补货
                'risk_tolerance': 0.2,
                'efficiency_weight': 0.3,
                'safety_margin': 0.2
            }
        elif self.strategy == ReplenishmentStrategy.AGGRESSIVE:
            return {
                'capacity_threshold': 0.1,  # 10%容量时触发补货
                'risk_tolerance': 0.8,
                'efficiency_weight': 0.8,
                'safety_margin': 0.05
            }
        elif self.strategy == ReplenishmentStrategy.BALANCED:
            return {
                'capacity_threshold': 0.2,  # 20%容量时触发补货
                'risk_tolerance': 0.5,
                'efficiency_weight': 0.5,
                'safety_margin': 0.1
            }
        else:  # ADAPTIVE
            return {
                'capacity_threshold': 0.2,
                'risk_tolerance': 0.5,
                'efficiency_weight': 0.5,
                'safety_margin': 0.1
            }
    
    def should_replenish(self, 
                        truck_state: TruckState,
                        remaining_lockers: List[LockerDemand],
                        current_time: int) -> ReplenishmentDecision:
        """
        判断是否需要补货
        
        Args:
            truck_state: 卡车当前状态
            remaining_lockers: 剩余未服务的快递柜
            current_time: 当前时间步
            
        Returns:
            ReplenishmentDecision: 补货决策结果
        """
        # 检查各种触发条件
        triggers = self._check_replenishment_triggers(truck_state, remaining_lockers, current_time)
        
        if not triggers:
            return ReplenishmentDecision(
                should_replenish=False,
                trigger_reason=None,
                urgency_level=0.0,
                recommended_route=[],
                expected_benefit=0.0,
                risk_assessment=0.0,
                confidence=1.0
            )
        
        # 选择最紧急的触发条件
        primary_trigger = max(triggers, key=lambda x: x[1])
        trigger_type, urgency = primary_trigger
        
        # 计算补货路径和收益
        route, benefit, risk = self._optimize_replenishment_route(
            truck_state, remaining_lockers, current_time
        )
        
        # 计算决策置信度
        confidence = self._calculate_confidence(truck_state, remaining_lockers, urgency, benefit, risk)
        
        return ReplenishmentDecision(
            should_replenish=True,
            trigger_reason=trigger_type,
            urgency_level=urgency,
            recommended_route=route,
            expected_benefit=benefit,
            risk_assessment=risk,
            confidence=confidence
        )
    
    def _check_replenishment_triggers(self, 
                                    truck_state: TruckState,
                                    remaining_lockers: List[LockerDemand],
                                    current_time: int) -> List[Tuple[ReplenishmentTrigger, float]]:
        """检查所有补货触发条件"""
        triggers = []
        
        # 1. 容量阈值触发
        delivery_ratio = truck_state.current_delivery_load / self.truck_capacity
        return_ratio = truck_state.current_return_load / self.truck_capacity
        
        if delivery_ratio < self.strategy_params['capacity_threshold']:
            urgency = 1.0 - delivery_ratio / self.strategy_params['capacity_threshold']
            triggers.append((ReplenishmentTrigger.CAPACITY_THRESHOLD, urgency))
        
        # 2. 需求预测触发
        total_remaining_demand = sum(l.delivery_demand + l.return_demand for l in remaining_lockers)
        if total_remaining_demand > truck_state.current_delivery_load + truck_state.remaining_capacity:
            shortage_ratio = (total_remaining_demand - truck_state.current_delivery_load - truck_state.remaining_capacity) / total_remaining_demand
            triggers.append((ReplenishmentTrigger.DEMAND_PREDICTION, shortage_ratio))
        
        # 3. 距离优化触发
        distance_benefit = self._calculate_distance_benefit(truck_state, remaining_lockers)
        if distance_benefit > 20.0:  # 如果补货能节省超过20单位距离
            urgency = min(1.0, distance_benefit / 50.0)
            triggers.append((ReplenishmentTrigger.DISTANCE_OPTIMIZATION, urgency))
        
        # 4. 时间窗触发
        time_pressure = self._calculate_time_pressure(remaining_lockers, current_time)
        if time_pressure > 0.7:
            triggers.append((ReplenishmentTrigger.TIME_WINDOW, time_pressure))
        
        # 5. 紧急补货触发
        if self._is_emergency_situation(truck_state, remaining_lockers):
            triggers.append((ReplenishmentTrigger.EMERGENCY, 1.0))
        
        return triggers
    
    def _calculate_distance_benefit(self, truck_state: TruckState, remaining_lockers: List[LockerDemand]) -> float:
        """计算补货带来的距离收益"""
        if not remaining_lockers:
            return 0.0
        
        current_location = self._get_location(truck_state.current_location, remaining_lockers)
        
        # 计算不补货的总距离
        no_replenish_distance = self._calculate_route_distance(
            current_location, [l.location for l in remaining_lockers], self.depot_location
        )
        
        # 计算补货后的总距离
        depot_distance = self._euclidean_distance(current_location, self.depot_location)
        replenish_distance = depot_distance + self._calculate_route_distance(
            self.depot_location, [l.location for l in remaining_lockers], self.depot_location
        )
        
        return max(0.0, no_replenish_distance - replenish_distance)
    
    def _calculate_time_pressure(self, remaining_lockers: List[LockerDemand], current_time: int) -> float:
        """计算时间压力"""
        if not remaining_lockers:
            return 0.0
        
        pressures = []
        for locker in remaining_lockers:
            if locker.time_window_end > current_time:
                time_remaining = locker.time_window_end - current_time
                pressure = 1.0 - (time_remaining / (locker.time_window_end - locker.time_window_start))
                pressures.append(max(0.0, pressure))
        
        return np.mean(pressures) if pressures else 0.0
    
    def _is_emergency_situation(self, truck_state: TruckState, remaining_lockers: List[LockerDemand]) -> bool:
        """判断是否为紧急情况"""
        # 如果卡车完全没有配送货物且还有高优先级需求
        if truck_state.current_delivery_load == 0:
            high_priority_lockers = [l for l in remaining_lockers if l.priority > 0.8 and l.delivery_demand > 0]
            return len(high_priority_lockers) > 0
        
        # 如果卡车没有剩余容量且还有大量取件需求
        if truck_state.remaining_capacity == 0:
            total_return_demand = sum(l.return_demand for l in remaining_lockers)
            return total_return_demand > self.truck_capacity * 0.3
        
        return False
    
    def _optimize_replenishment_route(self, 
                                    truck_state: TruckState,
                                    remaining_lockers: List[LockerDemand],
                                    current_time: int) -> Tuple[List[int], float, float]:
        """优化补货路径"""
        if not remaining_lockers:
            return [], 0.0, 0.0
        
        current_location = self._get_location(truck_state.current_location, remaining_lockers)
        
        # 简单的贪心算法优化路径
        unvisited = remaining_lockers.copy()
        route = [0]  # 从仓库开始
        total_benefit = 0.0
        total_risk = 0.0
        
        current_pos = self.depot_location
        
        while unvisited:
            # 选择下一个最优快递柜
            best_locker = None
            best_score = -float('inf')
            
            for locker in unvisited:
                distance = self._euclidean_distance(current_pos, locker.location)
                
                # 计算综合评分
                demand_score = (locker.delivery_demand + locker.return_demand) / 20.0
                priority_score = locker.priority
                distance_score = 1.0 / (1.0 + distance / 10.0)
                time_score = self._calculate_time_score(locker, current_time)
                
                score = (demand_score * 0.3 + priority_score * 0.3 + 
                        distance_score * 0.2 + time_score * 0.2)
                
                if score > best_score:
                    best_score = score
                    best_locker = locker
            
            if best_locker:
                route.append(best_locker.locker_id)
                total_benefit += best_score * 10.0
                total_risk += self._calculate_locker_risk(best_locker, current_time)
                current_pos = best_locker.location
                unvisited.remove(best_locker)
        
        route.append(0)  # 返回仓库
        
        return route, total_benefit, total_risk / len(remaining_lockers) if remaining_lockers else 0.0
    
    def _calculate_time_score(self, locker: LockerDemand, current_time: int) -> float:
        """计算时间评分"""
        if current_time < locker.time_window_start:
            return 0.5  # 太早
        elif current_time > locker.time_window_end:
            return 0.0  # 太晚
        else:
            # 在时间窗内，越接近开始时间评分越高
            window_progress = (current_time - locker.time_window_start) / (locker.time_window_end - locker.time_window_start)
            return 1.0 - window_progress * 0.5
    
    def _calculate_locker_risk(self, locker: LockerDemand, current_time: int) -> float:
        """计算快递柜风险"""
        time_risk = 0.0
        if current_time > locker.time_window_end:
            time_risk = 0.8
        elif current_time > locker.time_window_start + (locker.time_window_end - locker.time_window_start) * 0.8:
            time_risk = 0.4
        
        demand_risk = min(0.5, (locker.delivery_demand + locker.return_demand) / 30.0)
        
        return time_risk + demand_risk
    
    def _calculate_confidence(self, 
                            truck_state: TruckState,
                            remaining_lockers: List[LockerDemand],
                            urgency: float,
                            benefit: float,
                            risk: float) -> float:
        """计算决策置信度"""
        # 基于历史成功率
        historical_confidence = self.performance_metrics['successful_replenishments'] / max(1, self.performance_metrics['total_replenishments'])
        
        # 基于当前情况
        situation_confidence = min(1.0, urgency + benefit / 50.0 - risk)
        
        # 综合置信度
        return (historical_confidence * 0.3 + situation_confidence * 0.7)
    
    def _get_location(self, location_id: int, remaining_lockers: List[LockerDemand]) -> Tuple[float, float]:
        """获取位置坐标"""
        if location_id == 0:
            return self.depot_location
        
        for locker in remaining_lockers:
            if locker.locker_id == location_id:
                return locker.location
        
        return self.depot_location
    
    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def _calculate_route_distance(self, 
                                start: Tuple[float, float],
                                waypoints: List[Tuple[float, float]],
                                end: Tuple[float, float]) -> float:
        """计算路径总距离"""
        if not waypoints:
            return self._euclidean_distance(start, end)
        
        total_distance = self._euclidean_distance(start, waypoints[0])
        
        for i in range(len(waypoints) - 1):
            total_distance += self._euclidean_distance(waypoints[i], waypoints[i + 1])
        
        total_distance += self._euclidean_distance(waypoints[-1], end)
        
        return total_distance
    
    def update_performance(self, decision: ReplenishmentDecision, actual_benefit: float, success: bool):
        """更新性能指标"""
        self.performance_metrics['total_replenishments'] += 1
        
        if success:
            self.performance_metrics['successful_replenishments'] += 1
            self.performance_metrics['total_distance_saved'] += actual_benefit
        
        # 更新平均效率
        self.performance_metrics['average_efficiency'] = (
            self.performance_metrics['total_distance_saved'] / 
            max(1, self.performance_metrics['total_replenishments'])
        )
        
        # 记录历史
        self.replenishment_history.append({
            'decision': decision,
            'actual_benefit': actual_benefit,
            'success': success
        })
        
        # 自适应策略调整
        if self.strategy == ReplenishmentStrategy.ADAPTIVE:
            self._adapt_strategy()
    
    def _adapt_strategy(self):
        """自适应策略调整"""
        if len(self.replenishment_history) < 10:
            return
        
        recent_history = self.replenishment_history[-10:]
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        
        # 根据成功率调整策略参数
        if success_rate < 0.6:
            # 成功率低，采用更保守的策略
            self.strategy_params['capacity_threshold'] = min(0.4, self.strategy_params['capacity_threshold'] + 0.05)
            self.strategy_params['safety_margin'] = min(0.3, self.strategy_params['safety_margin'] + 0.02)
        elif success_rate > 0.8:
            # 成功率高，可以采用更激进的策略
            self.strategy_params['capacity_threshold'] = max(0.1, self.strategy_params['capacity_threshold'] - 0.02)
            self.strategy_params['safety_margin'] = max(0.05, self.strategy_params['safety_margin'] - 0.01)
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        return {
            'strategy': self.strategy.value,
            'parameters': self.strategy_params.copy(),
            'performance': self.performance_metrics.copy(),
            'recent_decisions': len(self.replenishment_history),
            'success_rate': (
                self.performance_metrics['successful_replenishments'] / 
                max(1, self.performance_metrics['total_replenishments'])
            )
        }