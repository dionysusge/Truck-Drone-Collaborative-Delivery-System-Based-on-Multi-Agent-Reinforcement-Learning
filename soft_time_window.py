"""
软时间窗约束处理模块

功能: 实现软时间窗模型，提供渐进式惩罚机制和时间窗违规处理策略
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ViolationType(Enum):
    """时间窗违规类型"""
    EARLY_ARRIVAL = "early_arrival"  # 早到
    LATE_ARRIVAL = "late_arrival"    # 迟到
    NO_VIOLATION = "no_violation"    # 无违规


class PenaltyFunction(Enum):
    """惩罚函数类型"""
    LINEAR = "linear"           # 线性惩罚
    QUADRATIC = "quadratic"     # 二次惩罚
    EXPONENTIAL = "exponential" # 指数惩罚
    PIECEWISE = "piecewise"     # 分段惩罚


@dataclass
class TimeWindow:
    """时间窗定义"""
    early_start: float      # 最早开始时间
    preferred_start: float  # 偏好开始时间
    preferred_end: float    # 偏好结束时间
    late_end: float        # 最晚结束时间
    service_time: float    # 服务时间
    priority: float = 1.0  # 优先级权重


@dataclass
class ViolationInfo:
    """违规信息"""
    violation_type: ViolationType
    violation_amount: float  # 违规时间量
    penalty_score: float     # 惩罚分数
    location_id: int         # 位置ID
    arrival_time: float      # 实际到达时间


class SoftTimeWindowManager:
    """软时间窗管理器"""
    
    def __init__(self, 
                 penalty_function: PenaltyFunction = PenaltyFunction.QUADRATIC,
                 early_penalty_weight: float = 0.5,
                 late_penalty_weight: float = 2.0,
                 max_penalty: float = 100.0):
        """
        初始化软时间窗管理器
        
        参数:
        - penalty_function: 惩罚函数类型
        - early_penalty_weight: 早到惩罚权重
        - late_penalty_weight: 迟到惩罚权重
        - max_penalty: 最大惩罚值
        """
        self.penalty_function = penalty_function
        self.early_penalty_weight = early_penalty_weight
        self.late_penalty_weight = late_penalty_weight
        self.max_penalty = max_penalty
        
        # 时间窗存储
        self.time_windows: Dict[int, TimeWindow] = {}
        
        # 违规历史记录
        self.violation_history: List[ViolationInfo] = []
        
        # 自适应参数
        self.adaptive_weights = True
        self.violation_counts = {"early": 0, "late": 0}
        
    def set_time_window(self, location_id: int, time_window: TimeWindow):
        """设置位置的时间窗"""
        self.time_windows[location_id] = time_window
        
    def get_time_window(self, location_id: int) -> Optional[TimeWindow]:
        """获取位置的时间窗"""
        return self.time_windows.get(location_id)
        
    def calculate_violation_penalty(self, 
                                  location_id: int, 
                                  arrival_time: float) -> Tuple[float, ViolationInfo]:
        """
        计算时间窗违规惩罚
        
        参数:
        - location_id: 位置ID
        - arrival_time: 到达时间
        
        返回:
        - penalty: 惩罚分数
        - violation_info: 违规信息
        """
        time_window = self.time_windows.get(location_id)
        if not time_window:
            # 如果没有时间窗约束，返回零惩罚
            return 0.0, ViolationInfo(
                ViolationType.NO_VIOLATION, 0.0, 0.0, location_id, arrival_time
            )
        
        # 判断违规类型和程度
        violation_type = ViolationType.NO_VIOLATION
        violation_amount = 0.0
        
        if arrival_time < time_window.preferred_start:
            # 早到违规
            violation_type = ViolationType.EARLY_ARRIVAL
            violation_amount = time_window.preferred_start - arrival_time
        elif arrival_time > time_window.preferred_end:
            # 迟到违规
            violation_type = ViolationType.LATE_ARRIVAL
            violation_amount = arrival_time - time_window.preferred_end
        
        # 计算惩罚分数
        penalty = self._calculate_penalty_score(
            violation_type, violation_amount, time_window.priority
        )
        
        # 创建违规信息
        violation_info = ViolationInfo(
            violation_type, violation_amount, penalty, location_id, arrival_time
        )
        
        # 记录违规历史
        self.violation_history.append(violation_info)
        
        # 更新违规计数
        if violation_type == ViolationType.EARLY_ARRIVAL:
            self.violation_counts["early"] += 1
        elif violation_type == ViolationType.LATE_ARRIVAL:
            self.violation_counts["late"] += 1
        
        return penalty, violation_info
    
    def _calculate_penalty_score(self, 
                               violation_type: ViolationType, 
                               violation_amount: float, 
                               priority: float) -> float:
        """计算惩罚分数"""
        if violation_type == ViolationType.NO_VIOLATION:
            return 0.0
        
        # 选择权重
        if violation_type == ViolationType.EARLY_ARRIVAL:
            weight = self.early_penalty_weight
        else:
            weight = self.late_penalty_weight
        
        # 自适应权重调整
        if self.adaptive_weights:
            weight = self._get_adaptive_weight(violation_type)
        
        # 根据惩罚函数类型计算基础惩罚
        base_penalty = self._apply_penalty_function(violation_amount)
        
        # 应用权重和优先级
        penalty = base_penalty * weight * priority
        
        # 限制最大惩罚
        return min(penalty, self.max_penalty)
    
    def _apply_penalty_function(self, violation_amount: float) -> float:
        """应用惩罚函数"""
        if self.penalty_function == PenaltyFunction.LINEAR:
            return violation_amount
        
        elif self.penalty_function == PenaltyFunction.QUADRATIC:
            return violation_amount ** 2
        
        elif self.penalty_function == PenaltyFunction.EXPONENTIAL:
            return math.exp(violation_amount) - 1
        
        elif self.penalty_function == PenaltyFunction.PIECEWISE:
            # 分段惩罚：轻微违规线性，严重违规二次
            if violation_amount <= 5.0:
                return violation_amount
            else:
                return 5.0 + (violation_amount - 5.0) ** 2
        
        return violation_amount
    
    def _get_adaptive_weight(self, violation_type: ViolationType) -> float:
        """获取自适应权重"""
        total_violations = sum(self.violation_counts.values())
        if total_violations == 0:
            return self.early_penalty_weight if violation_type == ViolationType.EARLY_ARRIVAL else self.late_penalty_weight
        
        # 根据违规历史调整权重
        if violation_type == ViolationType.EARLY_ARRIVAL:
            early_ratio = self.violation_counts["early"] / total_violations
            # 如果早到违规过多，增加早到惩罚权重
            return self.early_penalty_weight * (1 + early_ratio)
        else:
            late_ratio = self.violation_counts["late"] / total_violations
            # 如果迟到违规过多，增加迟到惩罚权重
            return self.late_penalty_weight * (1 + late_ratio)
    
    def get_feasible_time_range(self, location_id: int) -> Tuple[float, float]:
        """
        获取位置的可行时间范围
        
        返回:
        - (earliest_time, latest_time): 最早和最晚可接受时间
        """
        time_window = self.time_windows.get(location_id)
        if not time_window:
            return 0.0, float('inf')
        
        return time_window.early_start, time_window.late_end
    
    def get_preferred_time_range(self, location_id: int) -> Tuple[float, float]:
        """
        获取位置的偏好时间范围
        
        返回:
        - (preferred_start, preferred_end): 偏好开始和结束时间
        """
        time_window = self.time_windows.get(location_id)
        if not time_window:
            return 0.0, float('inf')
        
        return time_window.preferred_start, time_window.preferred_end
    
    def is_feasible_arrival(self, location_id: int, arrival_time: float) -> bool:
        """检查到达时间是否可行"""
        earliest, latest = self.get_feasible_time_range(location_id)
        return earliest <= arrival_time <= latest
    
    def get_violation_statistics(self) -> Dict[str, float]:
        """获取违规统计信息"""
        if not self.violation_history:
            return {
                "total_violations": 0,
                "early_violations": 0,
                "late_violations": 0,
                "avg_penalty": 0.0,
                "max_penalty": 0.0,
                "early_ratio": 0.0,
                "late_ratio": 0.0
            }
        
        early_violations = [v for v in self.violation_history if v.violation_type == ViolationType.EARLY_ARRIVAL]
        late_violations = [v for v in self.violation_history if v.violation_type == ViolationType.LATE_ARRIVAL]
        
        total_violations = len(early_violations) + len(late_violations)
        
        return {
            "total_violations": total_violations,
            "early_violations": len(early_violations),
            "late_violations": len(late_violations),
            "avg_penalty": np.mean([v.penalty_score for v in self.violation_history]),
            "max_penalty": max([v.penalty_score for v in self.violation_history]),
            "early_ratio": len(early_violations) / total_violations if total_violations > 0 else 0.0,
            "late_ratio": len(late_violations) / total_violations if total_violations > 0 else 0.0
        }
    
    def reset_violation_history(self):
        """重置违规历史"""
        self.violation_history.clear()
        self.violation_counts = {"early": 0, "late": 0}
    
    def update_penalty_weights(self, early_weight: float, late_weight: float):
        """更新惩罚权重"""
        self.early_penalty_weight = early_weight
        self.late_penalty_weight = late_weight
    
    def get_time_window_slack(self, location_id: int) -> Dict[str, float]:
        """
        获取时间窗的松弛度信息
        
        返回:
        - early_slack: 早到容忍时间
        - late_slack: 迟到容忍时间
        - total_window: 总时间窗长度
        """
        time_window = self.time_windows.get(location_id)
        if not time_window:
            return {"early_slack": 0.0, "late_slack": 0.0, "total_window": 0.0}
        
        early_slack = time_window.preferred_start - time_window.early_start
        late_slack = time_window.late_end - time_window.preferred_end
        total_window = time_window.late_end - time_window.early_start
        
        return {
            "early_slack": early_slack,
            "late_slack": late_slack,
            "total_window": total_window
        }


class TimeWindowOptimizer:
    """时间窗优化器"""
    
    def __init__(self, soft_tw_manager: SoftTimeWindowManager):
        """
        初始化时间窗优化器
        
        参数:
        - soft_tw_manager: 软时间窗管理器
        """
        self.soft_tw_manager = soft_tw_manager
    
    def optimize_arrival_time(self, 
                            location_id: int, 
                            earliest_possible: float, 
                            latest_possible: float) -> float:
        """
        优化到达时间以最小化时间窗惩罚
        
        参数:
        - location_id: 位置ID
        - earliest_possible: 最早可能到达时间
        - latest_possible: 最晚可能到达时间
        
        返回:
        - optimal_arrival_time: 最优到达时间
        """
        time_window = self.soft_tw_manager.get_time_window(location_id)
        if not time_window:
            # 如果没有时间窗约束，选择最早到达时间
            return earliest_possible
        
        # 获取偏好时间范围
        preferred_start = time_window.preferred_start
        preferred_end = time_window.preferred_end
        
        # 如果可行时间范围与偏好时间范围有交集，选择交集中的时间
        if latest_possible >= preferred_start and earliest_possible <= preferred_end:
            # 选择交集中的最优时间
            return max(earliest_possible, min(latest_possible, preferred_start))
        
        # 如果没有交集，选择惩罚最小的时间
        candidate_times = [earliest_possible, latest_possible, preferred_start, preferred_end]
        candidate_times = [t for t in candidate_times if earliest_possible <= t <= latest_possible]
        
        if not candidate_times:
            return earliest_possible
        
        # 计算每个候选时间的惩罚
        best_time = candidate_times[0]
        best_penalty = float('inf')
        
        for time in candidate_times:
            penalty, _ = self.soft_tw_manager.calculate_violation_penalty(location_id, time)
            if penalty < best_penalty:
                best_penalty = penalty
                best_time = time
        
        return best_time
    
    def suggest_time_window_adjustments(self, location_id: int) -> Dict[str, float]:
        """
        建议时间窗调整
        
        参数:
        - location_id: 位置ID
        
        返回:
        - 调整建议字典
        """
        # 分析该位置的违规历史
        location_violations = [
            v for v in self.soft_tw_manager.violation_history 
            if v.location_id == location_id
        ]
        
        if not location_violations:
            return {"no_adjustments_needed": True}
        
        early_violations = [v for v in location_violations if v.violation_type == ViolationType.EARLY_ARRIVAL]
        late_violations = [v for v in location_violations if v.violation_type == ViolationType.LATE_ARRIVAL]
        
        suggestions = {}
        
        if early_violations:
            avg_early_violation = np.mean([v.violation_amount for v in early_violations])
            suggestions["extend_early_start"] = avg_early_violation
        
        if late_violations:
            avg_late_violation = np.mean([v.violation_amount for v in late_violations])
            suggestions["extend_late_end"] = avg_late_violation
        
        return suggestions