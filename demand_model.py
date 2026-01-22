"""
随机需求模型和不确定性处理
作者: Dionysus

实现需求的随机性建模、预测机制和补救策略
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class DemandPattern(Enum):
    """需求模式枚举"""
    UNIFORM = "uniform"          # 均匀分布
    NORMAL = "normal"            # 正态分布
    POISSON = "poisson"          # 泊松分布
    SEASONAL = "seasonal"        # 季节性模式
    PEAK_HOURS = "peak_hours"    # 高峰时段模式


@dataclass
class DemandParameters:
    """需求参数配置"""
    pattern: DemandPattern
    base_rate: float             # 基础需求率
    variance: float              # 方差
    seasonal_amplitude: float    # 季节性振幅
    peak_hours: List[int]        # 高峰时段
    uncertainty_factor: float    # 不确定性因子


class RandomDemandGenerator:
    """
    随机需求生成器
    
    根据不同的需求模式生成随机需求
    """
    
    def __init__(self, num_lockers: int, time_horizon: int = 100):
        """
        初始化随机需求生成器
        
        Args:
            num_lockers: 快递柜数量
            time_horizon: 时间范围
        """
        self.num_lockers = num_lockers
        self.time_horizon = time_horizon
        
        # 使用时间戳确保每次初始化都有不同的随机性
        import time
        seed = int(time.time() * 1000) % 2147483647 + random.randint(0, 1000)
        random.seed(seed)
        np.random.seed(seed)
        
        # 为每个快递柜配置需求参数
        self.locker_demand_params = self._initialize_demand_parameters()
        
        # 历史需求数据
        self.demand_history = []
        self.max_history_length = 50
        
    def _initialize_demand_parameters(self) -> Dict[int, Dict[str, DemandParameters]]:
        """
        初始化每个快递柜的需求参数
        
        Returns:
            快递柜需求参数字典
        """
        params = {}
        
        for locker_id in range(1, self.num_lockers + 1):
            # 使用泊松分布模式，与config.py中的1-4泊松分布保持一致
            delivery_pattern = DemandPattern.POISSON
            return_pattern = DemandPattern.POISSON
            
            params[locker_id] = {
                'delivery': DemandParameters(
                    pattern=delivery_pattern,
                    base_rate=random.uniform(1.0, 4.0),    # 与config.py中的lambda范围保持一致
                    variance=random.uniform(0.5, 1.5),     # 泊松分布的方差等于均值，这里设置较小的额外方差
                    seasonal_amplitude=random.uniform(0.1, 0.3),
                    peak_hours=[8, 12, 18, 20],
                    uncertainty_factor=random.uniform(0.1, 0.2)
                ),
                'return': DemandParameters(
                    pattern=return_pattern,
                    base_rate=random.uniform(1.0, 4.0),    # 与config.py中的lambda范围保持一致
                    variance=random.uniform(0.5, 1.5),     # 泊松分布的方差等于均值，这里设置较小的额外方差
                    seasonal_amplitude=random.uniform(0.1, 0.3),
                    peak_hours=[10, 14, 19],
                    uncertainty_factor=random.uniform(0.1, 0.2)
                )
            }
        
        return params
    
    def generate_demand(self, locker_id: int, demand_type: str, 
                       current_time: int) -> Tuple[float, float]:
        """
        生成指定快递柜的需求
        
        Args:
            locker_id: 快递柜ID
            demand_type: 需求类型 ('delivery' 或 'return')
            current_time: 当前时间
            
        Returns:
            (期望需求, 实际需求)
        """
        if locker_id not in self.locker_demand_params:
            return 0.0, 0.0
        
        params = self.locker_demand_params[locker_id][demand_type]
        
        # 计算基础需求率
        base_demand = self._calculate_base_demand(params, current_time)
        
        # 添加随机性
        actual_demand = self._add_randomness(base_demand, params)
        
        # 确保需求为非负整数
        actual_demand = max(0, int(round(actual_demand)))
        
        return base_demand, float(actual_demand)
    
    def _calculate_base_demand(self, params: DemandParameters, current_time: int) -> float:
        """
        计算基础需求率
        
        Args:
            params: 需求参数
            current_time: 当前时间
            
        Returns:
            基础需求率
        """
        base = params.base_rate
        
        # 添加季节性变化
        if params.pattern == DemandPattern.SEASONAL:
            seasonal_factor = 1 + params.seasonal_amplitude * math.sin(
                2 * math.pi * current_time / self.time_horizon
            )
            base *= seasonal_factor
        
        # 添加高峰时段效应
        if params.pattern == DemandPattern.PEAK_HOURS:
            hour_of_day = current_time % 24
            if hour_of_day in params.peak_hours:
                base *= (1 + params.seasonal_amplitude)
        
        return base
    
    def _add_randomness(self, base_demand: float, params: DemandParameters) -> float:
        """
        添加随机性到基础需求
        
        Args:
            base_demand: 基础需求
            params: 需求参数
            
        Returns:
            带随机性的需求
        """
        if params.pattern == DemandPattern.UNIFORM:
            noise = random.uniform(-params.variance, params.variance)
        elif params.pattern == DemandPattern.NORMAL:
            noise = random.gauss(0, params.variance)
        elif params.pattern == DemandPattern.POISSON:
            # 泊松分布的方差等于均值
            return np.random.poisson(base_demand)
        else:
            noise = random.gauss(0, params.variance)
        
        return base_demand + noise
    
    def update_demand_history(self, locker_demands: Dict[int, Dict[str, float]]):
        """
        更新需求历史
        
        Args:
            locker_demands: 快递柜需求数据
        """
        self.demand_history.append(locker_demands.copy())
        
        # 保持历史长度限制
        if len(self.demand_history) > self.max_history_length:
            self.demand_history.pop(0)


class DemandPredictor:
    """
    需求预测器
    
    基于历史数据预测未来需求
    """
    
    def __init__(self, num_lockers: int, prediction_horizon: int = 10):
        """
        初始化需求预测器
        
        Args:
            num_lockers: 快递柜数量
            prediction_horizon: 预测时间范围
        """
        self.num_lockers = num_lockers
        self.prediction_horizon = prediction_horizon
        
        # 简单的移动平均预测模型
        self.window_size = 5
        
    def predict_demand(self, demand_history: List[Dict[int, Dict[str, float]]], 
                      current_time: int) -> Dict[int, Dict[str, List[float]]]:
        """
        预测未来需求
        
        Args:
            demand_history: 历史需求数据
            current_time: 当前时间
            
        Returns:
            预测的需求数据
        """
        predictions = {}
        
        for locker_id in range(1, self.num_lockers + 1):
            predictions[locker_id] = {
                'delivery': self._predict_locker_demand(
                    demand_history, locker_id, 'delivery'
                ),
                'return': self._predict_locker_demand(
                    demand_history, locker_id, 'return'
                )
            }
        
        return predictions
    
    def _predict_locker_demand(self, demand_history: List[Dict[int, Dict[str, float]]], 
                              locker_id: int, demand_type: str) -> List[float]:
        """
        预测单个快递柜的需求
        
        Args:
            demand_history: 历史需求数据
            locker_id: 快递柜ID
            demand_type: 需求类型
            
        Returns:
            预测的需求序列
        """
        if not demand_history:
            return [1.0] * self.prediction_horizon
        
        # 提取历史需求序列
        historical_demands = []
        for record in demand_history[-self.window_size:]:
            if locker_id in record and demand_type in record[locker_id]:
                historical_demands.append(record[locker_id][demand_type])
        
        if not historical_demands:
            return [1.0] * self.prediction_horizon
        
        # 简单移动平均预测
        avg_demand = sum(historical_demands) / len(historical_demands)
        
        # 计算趋势
        if len(historical_demands) >= 2:
            trend = (historical_demands[-1] - historical_demands[0]) / len(historical_demands)
        else:
            trend = 0.0
        
        # 生成预测序列
        predictions = []
        for i in range(self.prediction_horizon):
            predicted_demand = avg_demand + trend * i
            predictions.append(max(0.0, predicted_demand))
        
        return predictions
    
    def get_demand_uncertainty(self, demand_history: List[Dict[int, Dict[str, float]]], 
                              locker_id: int, demand_type: str) -> float:
        """
        计算需求的不确定性
        
        Args:
            demand_history: 历史需求数据
            locker_id: 快递柜ID
            demand_type: 需求类型
            
        Returns:
            需求不确定性（标准差）
        """
        if not demand_history:
            return 1.0
        
        # 提取历史需求序列
        historical_demands = []
        for record in demand_history[-self.window_size:]:
            if locker_id in record and demand_type in record[locker_id]:
                historical_demands.append(record[locker_id][demand_type])
        
        if len(historical_demands) < 2:
            return 1.0
        
        # 计算标准差
        mean_demand = sum(historical_demands) / len(historical_demands)
        variance = sum((d - mean_demand) ** 2 for d in historical_demands) / len(historical_demands)
        
        return math.sqrt(variance)


class ContingencyStrategy:
    """
    应急策略管理器
    
    处理需求超出预期时的应急措施
    """
    
    def __init__(self, num_trucks: int, truck_capacity: int):
        """
        初始化应急策略管理器
        
        Args:
            num_trucks: 卡车数量
            truck_capacity: 卡车容量
        """
        self.num_trucks = num_trucks
        self.truck_capacity = truck_capacity
        
        # 应急策略配置
        self.emergency_threshold = 0.8  # 容量使用率阈值
        self.reallocation_enabled = True
        self.priority_service_enabled = True
        
    def detect_capacity_shortage(self, trucks: List[Dict[str, Any]], 
                                lockers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检测容量短缺情况
        
        Args:
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            
        Returns:
            容量短缺分析结果
        """
        analysis = {
            'shortage_detected': False,
            'critical_lockers': [],
            'overloaded_trucks': [],
            'total_unmet_demand': 0,
            'recommended_actions': []
        }
        
        # 计算总需求和总容量
        total_delivery_demand = sum(
            locker.get('demand_del', 0) for locker in lockers if not locker.get('served', False)
        )
        total_return_demand = sum(
            locker.get('demand_ret', 0) for locker in lockers if not locker.get('served', False)
        )
        
        total_delivery_capacity = sum(
            self.truck_capacity - truck.get('current_delivery_load', 0) 
            for truck in trucks if not truck.get('returned', False)
        )
        total_return_capacity = sum(
            self.truck_capacity - truck.get('current_return_load', 0) 
            for truck in trucks if not truck.get('returned', False)
        )
        
        # 检测短缺
        delivery_shortage = max(0, total_delivery_demand - total_delivery_capacity)
        return_shortage = max(0, total_return_demand - total_return_capacity)
        
        if delivery_shortage > 0 or return_shortage > 0:
            analysis['shortage_detected'] = True
            analysis['total_unmet_demand'] = delivery_shortage + return_shortage
            
            # 识别关键快递柜
            analysis['critical_lockers'] = self._identify_critical_lockers(lockers)
            
            # 识别过载卡车
            analysis['overloaded_trucks'] = self._identify_overloaded_trucks(trucks)
            
            # 生成建议行动
            analysis['recommended_actions'] = self._generate_recommendations(
                delivery_shortage, return_shortage, trucks, lockers
            )
        
        return analysis
    
    def _identify_critical_lockers(self, lockers: List[Dict[str, Any]]) -> List[int]:
        """
        识别关键快递柜（需求高且未服务）
        
        Args:
            lockers: 快递柜状态列表
            
        Returns:
            关键快递柜ID列表
        """
        critical_lockers = []
        
        for locker in lockers:
            if locker.get('served', False):
                continue
                
            total_demand = locker.get('demand_del', 0) + locker.get('demand_ret', 0)
            if total_demand >= 5:  # 高需求阈值
                critical_lockers.append(locker['id'])
        
        return critical_lockers
    
    def _identify_overloaded_trucks(self, trucks: List[Dict[str, Any]]) -> List[int]:
        """
        识别过载的卡车
        
        Args:
            trucks: 卡车状态列表
            
        Returns:
            过载卡车ID列表
        """
        overloaded_trucks = []
        
        for i, truck in enumerate(trucks):
            if truck.get('returned', False):
                continue
                
            delivery_load = truck.get('current_delivery_load', 0)
            return_load = truck.get('current_return_load', 0)
            
            delivery_utilization = delivery_load / self.truck_capacity
            return_utilization = return_load / self.truck_capacity
            
            if delivery_utilization > self.emergency_threshold or return_utilization > self.emergency_threshold:
                overloaded_trucks.append(i)
        
        return overloaded_trucks
    
    def _generate_recommendations(self, delivery_shortage: float, return_shortage: float,
                                 trucks: List[Dict[str, Any]], 
                                 lockers: List[Dict[str, Any]]) -> List[str]:
        """
        生成应急建议
        
        Args:
            delivery_shortage: 配送容量短缺
            return_shortage: 退货容量短缺
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            
        Returns:
            建议行动列表
        """
        recommendations = []
        
        if delivery_shortage > 0:
            recommendations.append(f"配送容量短缺 {delivery_shortage:.1f} 单位，建议增加配送车辆或优化路径")
        
        if return_shortage > 0:
            recommendations.append(f"退货容量短缺 {return_shortage:.1f} 单位，建议优先处理退货任务")
        
        # 检查是否有卡车可以重新分配
        available_trucks = [
            i for i, truck in enumerate(trucks) 
            if not truck.get('returned', False) and truck.get('current_location', 0) == 0
        ]
        
        if available_trucks:
            recommendations.append(f"可重新分配 {len(available_trucks)} 辆卡车处理紧急需求")
        
        # 建议优先级服务
        if self.priority_service_enabled:
            recommendations.append("启用优先级服务，优先处理高需求快递柜")
        
        return recommendations
    
    def apply_emergency_measures(self, trucks: List[Dict[str, Any]], 
                                lockers: List[Dict[str, Any]], 
                                shortage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用应急措施
        
        Args:
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            shortage_analysis: 容量短缺分析结果
            
        Returns:
            应急措施执行结果
        """
        measures_applied = {
            'reallocation_performed': False,
            'priority_adjustment': False,
            'emergency_routing': False,
            'affected_trucks': [],
            'affected_lockers': []
        }
        
        if not shortage_analysis['shortage_detected']:
            return measures_applied
        
        # 重新分配任务
        if self.reallocation_enabled and shortage_analysis['overloaded_trucks']:
            measures_applied.update(
                self._reallocate_tasks(trucks, lockers, shortage_analysis['overloaded_trucks'])
            )
        
        # 调整优先级
        if self.priority_service_enabled and shortage_analysis['critical_lockers']:
            measures_applied.update(
                self._adjust_priorities(lockers, shortage_analysis['critical_lockers'])
            )
        
        return measures_applied
    
    def _reallocate_tasks(self, trucks: List[Dict[str, Any]], 
                         lockers: List[Dict[str, Any]], 
                         overloaded_trucks: List[int]) -> Dict[str, Any]:
        """
        重新分配任务
        
        Args:
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            overloaded_trucks: 过载卡车列表
            
        Returns:
            重新分配结果
        """
        result = {
            'reallocation_performed': True,
            'affected_trucks': overloaded_trucks
        }
        
        # 简单的重新分配策略：减少过载卡车的负担
        for truck_id in overloaded_trucks:
            truck = trucks[truck_id]
            
            # 减少部分负载（模拟重新分配给其他卡车）
            if truck.get('current_delivery_load', 0) > self.truck_capacity * 0.8:
                reduction = truck['current_delivery_load'] * 0.2
                truck['current_delivery_load'] = max(0, truck['current_delivery_load'] - reduction)
            
            if truck.get('current_return_load', 0) > self.truck_capacity * 0.8:
                reduction = truck['current_return_load'] * 0.2
                truck['current_return_load'] = max(0, truck['current_return_load'] - reduction)
        
        return result
    
    def _adjust_priorities(self, lockers: List[Dict[str, Any]], 
                          critical_lockers: List[int]) -> Dict[str, Any]:
        """
        调整快递柜优先级
        
        Args:
            lockers: 快递柜状态列表
            critical_lockers: 关键快递柜列表
            
        Returns:
            优先级调整结果
        """
        result = {
            'priority_adjustment': True,
            'affected_lockers': critical_lockers
        }
        
        # 为关键快递柜添加优先级标记
        for locker in lockers:
            if locker['id'] in critical_lockers:
                locker['priority'] = locker.get('priority', 1.0) * 1.5
        
        return result


class UncertaintyHandler:
    """
    不确定性处理器
    
    综合管理需求不确定性和应急响应
    """
    
    def __init__(self, num_trucks: int, num_lockers: int, truck_capacity: int):
        """
        初始化不确定性处理器
        
        Args:
            num_trucks: 卡车数量
            num_lockers: 快递柜数量
            truck_capacity: 卡车容量
        """
        self.demand_generator = RandomDemandGenerator(num_lockers)
        self.demand_predictor = DemandPredictor(num_lockers)
        self.contingency_strategy = ContingencyStrategy(num_trucks, truck_capacity)
        
        # 不确定性管理配置
        self.confidence_threshold = 0.7
        self.adaptation_rate = 0.05  # 降低适应率，减少环境变化
        
    def update_demand_model(self, current_time: int, 
                           observed_demands: Dict[int, Dict[str, float]]):
        """
        更新需求模型
        
        Args:
            current_time: 当前时间
            observed_demands: 观察到的需求数据
        """
        # 更新需求历史
        self.demand_generator.update_demand_history(observed_demands)
        
        # 可以在这里添加模型参数的在线学习
        
    def get_robust_demand_estimate(self, locker_id: int, demand_type: str, 
                                  current_time: int) -> Dict[str, float]:
        """
        获取鲁棒的需求估计
        
        Args:
            locker_id: 快递柜ID
            demand_type: 需求类型
            current_time: 当前时间
            
        Returns:
            需求估计结果
        """
        # 生成基础需求
        expected_demand, actual_demand = self.demand_generator.generate_demand(
            locker_id, demand_type, current_time
        )
        
        # 计算不确定性
        uncertainty = self.demand_predictor.get_demand_uncertainty(
            self.demand_generator.demand_history, locker_id, demand_type
        )
        
        # 计算置信区间
        confidence_interval = 1.96 * uncertainty  # 95%置信区间
        
        return {
            'expected': expected_demand,
            'actual': actual_demand,
            'uncertainty': uncertainty,
            'lower_bound': max(0, actual_demand - confidence_interval),
            'upper_bound': actual_demand + confidence_interval,
            'confidence': min(1.0, 1.0 / (1.0 + uncertainty))
        }
    
    def handle_demand_shock(self, trucks: List[Dict[str, Any]], 
                           lockers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理需求冲击
        
        Args:
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            
        Returns:
            处理结果
        """
        # 检测容量短缺
        shortage_analysis = self.contingency_strategy.detect_capacity_shortage(trucks, lockers)
        
        # 应用应急措施
        emergency_measures = self.contingency_strategy.apply_emergency_measures(
            trucks, lockers, shortage_analysis
        )
        
        return {
            'shortage_analysis': shortage_analysis,
            'emergency_measures': emergency_measures,
            'system_status': 'critical' if shortage_analysis['shortage_detected'] else 'normal'
        }