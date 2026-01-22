"""
动态规划无人机调度系统
作者: Dionysus
联系方式: wechat:gzw1546484791

实现基于动态规划的无人机调度算法，在给定的300秒时间窗内
智能调度无人机完成快递柜服务，支持软时间窗约束。
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from config import Config


@dataclass
class DroneTask:
    """无人机任务定义"""
    locker_id: int
    location: Tuple[float, float]
    delivery_demand: int
    return_demand: int
    distance: float
    priority: float
    estimated_time: float  # 预估完成时间（飞行+服务）


@dataclass
class DroneSchedule:
    """无人机调度方案"""
    drone_id: int
    tasks: List[DroneTask]
    total_time: float
    total_service_count: int
    efficiency_score: float


class DynamicDroneScheduler:
    """
    动态规划无人机调度器
    
    功能：
    1. 在300秒时间窗内动态调度无人机
    2. 基于动态规划优化任务分配
    3. 支持软时间窗约束和超时惩罚
    4. 实时监控无人机状态和任务进度
    """
    
    def __init__(self, max_service_time: int = 300, drone_speed: float = None, 
                 service_time_per_item: float = None):
        """
        初始化调度器
        
        Args:
            max_service_time: 最大服务时间（秒）
            drone_speed: 无人机飞行速度（距离单位/秒）
            service_time_per_item: 每个需求项的服务时间（秒）
        """
        self.max_service_time = max_service_time
        self.drone_speed = drone_speed or Config.DRONE_SPEED
        self.service_time_per_item = service_time_per_item or Config.DRONE_SERVICE_TIME
        self.num_drones = 4  # 调整为4个无人机，符合用户建议
        
        # 调度状态
        self.active_schedules: List[DroneSchedule] = []
        self.completed_tasks: List[DroneTask] = []
        self.current_time = 0
        self.overtime_penalty = 0.0
        
        # 强化学习相关参数
        self.rl_exploration_rate = 0.2  # 探索率
        self.rl_learning_enabled = True  # 启用强化学习
        self.rl_decision_history = []  # 决策历史记录
        
    def schedule_drones(self, truck_location: Tuple[float, float], 
                       available_lockers: List[Dict], 
                       drone_range: float,
                       rl_preferences: Optional[Dict] = None) -> Dict:
        """
        为卡车停靠点动态调度无人机 - 结合强化学习偏好
        
        Args:
            truck_location: 卡车当前位置
            available_lockers: 可服务的快递柜列表
            drone_range: 无人机最大航程
            rl_preferences: 强化学习智能体的调度偏好（可选）
            
        Returns:
            调度结果字典，包含任务分配、预估时间、服务统计等
        """
        # 重置调度状态
        self._reset_scheduling_state()
        
        # 筛选可达的快递柜并创建任务
        valid_tasks = self._create_valid_tasks(truck_location, available_lockers, drone_range)
        
        if not valid_tasks:
            return self._create_empty_result()
        
        # 优先使用强化学习引导的分配策略
        if rl_preferences:
            optimal_schedules = self._rl_guided_task_allocation(valid_tasks, rl_preferences)
        else:
            # 使用强化学习友好的简化分配，减少传统算法依赖
            optimal_schedules = self._rl_friendly_allocation(valid_tasks)
        
        # 增加强化学习探索机制
        if rl_preferences and rl_preferences.get('exploration_enabled', True):
            optimal_schedules = self._apply_rl_exploration(optimal_schedules, rl_preferences)
        
        # 执行调度并计算结果
        result = self._execute_scheduling(optimal_schedules)
        
        return result
    
    def _reset_scheduling_state(self):
        """重置调度状态"""
        self.active_schedules.clear()
        self.completed_tasks.clear()
        self.current_time = 0
        self.overtime_penalty = 0.0
    
    def _create_valid_tasks(self, truck_location: Tuple[float, float], 
                           available_lockers: List[Dict], 
                           drone_range: float) -> List[DroneTask]:
        """
        创建有效的无人机任务列表
        
        Args:
            truck_location: 卡车位置
            available_lockers: 可用快递柜
            drone_range: 无人机航程
            
        Returns:
            有效任务列表
        """
        valid_tasks = []
        
        for locker in available_lockers:
            if locker.get('served', False):
                continue
                
            # 计算距离
            distance = self._calculate_distance(truck_location, locker['location'])
            
            # 检查航程限制
            if 2 * distance > drone_range:
                continue
            
            # 获取需求信息（支持多种字段名格式）
            delivery_demand = locker.get('demand_del', locker.get('delivery_demand', 0))
            return_demand = locker.get('demand_ret', locker.get('return_demand', 0))
            
            # 如果没有任何需求，跳过
            if delivery_demand <= 0 and return_demand <= 0:
                continue
            
            # 创建合并任务：一次飞行处理配送和退货需求
            # 计算预估完成时间（飞行时间 + 服务时间）
            flight_time = (2 * distance) / self.drone_speed
            total_service_demand = delivery_demand + return_demand
            service_time = total_service_demand * self.service_time_per_item
            estimated_time = flight_time + service_time
            
            # 检查时间约束
            if estimated_time <= self.max_service_time:
                # 计算优先级
                priority = self._calculate_task_priority(locker, distance, total_service_demand)
                
                combined_task = DroneTask(
                    locker_id=locker['id'],
                    location=locker['location'],
                    delivery_demand=delivery_demand,
                    return_demand=return_demand,
                    distance=distance,
                    priority=priority,
                    estimated_time=estimated_time
                )
                valid_tasks.append(combined_task)
        
        return valid_tasks
    
    def _calculate_task_priority(self, locker: Dict, distance: float, 
                                total_demand: int) -> float:
        """
        计算任务优先级
        
        Args:
            locker: 快递柜信息
            distance: 距离
            total_demand: 总需求量
            
        Returns:
            优先级分数（越高越优先）
        """
        # 基础优先级：需求量越多优先级越高
        demand_score = total_demand * 10
        
        # 距离惩罚：距离越近优先级越高
        distance_penalty = distance * 2
        
        # 时间窗奖励（如果有时间窗信息）
        time_window_bonus = 0
        if 'time_window' in locker:
            # 根据时间窗紧急程度调整优先级
            urgency = locker.get('urgency', 1.0)
            time_window_bonus = urgency * 5
        
        priority = demand_score - distance_penalty + time_window_bonus
        return max(priority, 0.1)  # 确保优先级为正
    
    def _rl_guided_task_allocation(self, tasks: List[DroneTask], 
                                  rl_preferences: Dict) -> List[DroneSchedule]:
        """
        基于强化学习偏好的任务分配
        
        Args:
            tasks: 待分配的任务列表
            rl_preferences: 强化学习智能体的偏好设置
            
        Returns:
            调度方案列表
        """
        # 初始化调度方案
        schedules = [DroneSchedule(
            drone_id=i, 
            tasks=[], 
            total_time=0.0, 
            total_service_count=0,
            efficiency_score=0.0
        ) for i in range(self.num_drones)]
        
        # 根据RL偏好调整任务优先级
        adjusted_tasks = self._adjust_task_priorities_by_rl(tasks, rl_preferences)
        
        # 使用RL引导的分配策略
        for task in adjusted_tasks:
            drone_idx = self._rl_select_drone(task, schedules, rl_preferences)
            if drone_idx is not None:
                self._assign_task_to_drone(task, schedules[drone_idx])
        
        return schedules
    
    def _simplified_task_allocation(self, tasks: List[DroneTask]) -> List[DroneSchedule]:
        """
        简化的任务分配 - 减少算法优化，增加学习空间
        
        Args:
            tasks: 待分配的任务列表
            
        Returns:
            调度方案列表
        """
        # 初始化调度方案
        schedules = [DroneSchedule(
            drone_id=i, 
            tasks=[], 
            total_time=0.0, 
            total_service_count=0,
            efficiency_score=0.0
        ) for i in range(self.num_drones)]
        
        # 简单的轮询分配，增加随机性
        np.random.shuffle(tasks)  # 随机打乱任务顺序
        
        for i, task in enumerate(tasks):
            # 简单轮询分配，而不是最优分配
            drone_idx = i % self.num_drones
            
            # 检查时间约束
            if schedules[drone_idx].total_time + task.estimated_time <= self.max_service_time:
                self._assign_task_to_drone(task, schedules[drone_idx])
            else:
                # 尝试分配给其他无人机
                for j in range(self.num_drones):
                    alt_idx = (drone_idx + j + 1) % self.num_drones
                    if schedules[alt_idx].total_time + task.estimated_time <= self.max_service_time:
                        self._assign_task_to_drone(task, schedules[alt_idx])
                        break
        
        return schedules
    
    def _rl_friendly_allocation(self, tasks: List[DroneTask]) -> List[DroneSchedule]:
        """
        强化学习友好的任务分配方法
        
        减少传统算法优化，为强化学习提供更多学习空间
        
        Args:
            tasks: 待分配的任务列表
            
        Returns:
            调度方案列表
        """
        # 初始化调度方案
        schedules = [DroneSchedule(
            drone_id=i, 
            tasks=[], 
            total_time=0.0, 
            total_service_count=0,
            efficiency_score=0.0
        ) for i in range(self.num_drones)]
        
        # 使用基于概率的分配策略，而非确定性算法
        for task in tasks:
            # 计算每个无人机的分配概率
            probabilities = self._calculate_assignment_probabilities(task, schedules)
            
            # 基于概率选择无人机
            if probabilities:
                drone_idx = np.random.choice(len(probabilities), p=probabilities)
                if schedules[drone_idx].total_time + task.estimated_time <= self.max_service_time:
                    self._assign_task_to_drone(task, schedules[drone_idx])
                else:
                    # 寻找可行的备选方案
                    for i, schedule in enumerate(schedules):
                        if schedule.total_time + task.estimated_time <= self.max_service_time:
                            self._assign_task_to_drone(task, schedule)
                            break
        
        return schedules
    
    def _calculate_assignment_probabilities(self, task: DroneTask, 
                                          schedules: List[DroneSchedule]) -> List[float]:
        """
        计算任务分配到各无人机的概率
        
        Args:
            task: 待分配任务
            schedules: 当前调度方案
            
        Returns:
            分配概率列表
        """
        scores = []
        for schedule in schedules:
            if schedule.total_time + task.estimated_time > self.max_service_time:
                scores.append(0.0)
            else:
                # 基于负载均衡和时间效率计算得分
                load_factor = 1.0 / (1.0 + schedule.total_service_count)
                time_factor = 1.0 / (1.0 + schedule.total_time / self.max_service_time)
                scores.append(load_factor * time_factor)
        
        # 转换为概率分布
        total_score = sum(scores)
        if total_score == 0:
            return []
        
        probabilities = [score / total_score for score in scores]
        return probabilities
    
    def _apply_rl_exploration(self, schedules: List[DroneSchedule], 
                             rl_preferences: Dict) -> List[DroneSchedule]:
        """
        应用强化学习探索机制
        
        Args:
            schedules: 当前调度方案
            rl_preferences: 强化学习偏好设置
            
        Returns:
            应用探索后的调度方案
        """
        exploration_rate = rl_preferences.get('exploration_rate', self.rl_exploration_rate)
        
        # 记录决策历史
        decision_info = {
            'timestamp': self.current_time,
            'schedules_count': len(schedules),
            'total_tasks': sum(len(s.tasks) for s in schedules),
            'exploration_applied': False
        }
        
        # 应用探索策略
        if np.random.random() < exploration_rate:
            schedules = self._explore_schedule_variations(schedules, rl_preferences)
            decision_info['exploration_applied'] = True
        
        # 记录决策历史
        self.rl_decision_history.append(decision_info)
        
        return schedules
    
    def _explore_schedule_variations(self, schedules: List[DroneSchedule], 
                                   rl_preferences: Dict) -> List[DroneSchedule]:
        """
        探索调度方案的变化
        
        Args:
            schedules: 原始调度方案
            rl_preferences: 强化学习偏好设置
            
        Returns:
            探索后的调度方案
        """
        exploration_type = rl_preferences.get('exploration_type', 'task_swap')
        
        if exploration_type == 'task_swap' and len(schedules) >= 2:
            # 随机交换两个无人机的任务
            return self._swap_random_tasks(schedules)
        elif exploration_type == 'task_redistribution':
            # 重新分配部分任务
            return self._redistribute_tasks(schedules)
        else:
            # 默认：轻微调整任务顺序
            return self._shuffle_task_order(schedules)
    
    def _optimize_task_allocation(self, tasks: List[DroneTask]) -> List[DroneSchedule]:
        """
        使用动态规划优化任务分配
        
        Args:
            tasks: 待分配的任务列表
            
        Returns:
            优化后的无人机调度方案
        """
        if not tasks:
            return []
        
        # 按优先级排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # 初始化无人机调度
        schedules = [DroneSchedule(
            drone_id=i,
            tasks=[],
            total_time=0.0,
            total_service_count=0,
            efficiency_score=0.0
        ) for i in range(self.num_drones)]
        
        # 动态规划分配任务
        for task in sorted_tasks:
            best_drone = self._find_best_drone_for_task(task, schedules)
            if best_drone is not None:
                self._assign_task_to_drone(task, schedules[best_drone])
        
        # 过滤掉空调度
        active_schedules = [s for s in schedules if s.tasks]
        
        return active_schedules
    
    def _find_best_drone_for_task(self, task: DroneTask, 
                                 schedules: List[DroneSchedule]) -> Optional[int]:
        """
        为任务找到最佳的无人机
        
        Args:
            task: 待分配任务
            schedules: 当前调度方案
            
        Returns:
            最佳无人机索引，如果没有合适的返回None
        """
        best_drone = None
        best_score = float('-inf')
        
        for i, schedule in enumerate(schedules):
            # 检查时间约束
            new_total_time = schedule.total_time + task.estimated_time
            if new_total_time > self.max_service_time:
                continue  # 超时，跳过
            
            # 计算分配分数
            score = self._calculate_assignment_score(task, schedule)
            
            # 负载均衡奖励：优先选择任务较少的无人机
            load_balance_bonus = max(0, (self.num_drones - len(schedule.tasks)) * 5.0)
            score += load_balance_bonus
            
            if score > best_score:
                best_score = score
                best_drone = i
        
        return best_drone
    
    def _calculate_assignment_score(self, task: DroneTask, 
                                   schedule: DroneSchedule) -> float:
        """
        计算任务分配分数
        
        Args:
            task: 待分配任务
            schedule: 无人机调度方案
            
        Returns:
            分配分数（越高越好）
        """
        # 基础分数：任务优先级
        base_score = task.priority
        
        # 负载均衡奖励：优先分配给任务较少的无人机
        load_balance_bonus = (self.num_drones - len(schedule.tasks)) * 5
        
        # 时间利用率奖励：优先分配给剩余时间较多的无人机
        remaining_time = self.max_service_time - schedule.total_time
        time_utilization_bonus = (remaining_time / self.max_service_time) * 10
        
        # 效率奖励：考虑无人机当前效率
        efficiency_bonus = schedule.efficiency_score * 2
        
        total_score = (base_score + load_balance_bonus + 
                      time_utilization_bonus + efficiency_bonus)
        
        return total_score
    
    def _assign_task_to_drone(self, task: DroneTask, schedule: DroneSchedule):
        """
        将任务分配给无人机
        
        Args:
            task: 待分配任务
            schedule: 无人机调度方案
        """
        schedule.tasks.append(task)
        schedule.total_time += task.estimated_time
        schedule.total_service_count += (task.delivery_demand + task.return_demand)
        
        # 更新效率分数
        if schedule.total_time > 0:
            schedule.efficiency_score = schedule.total_service_count / schedule.total_time
        else:
            schedule.efficiency_score = 0.0
    
    def _execute_scheduling(self, schedules: List[DroneSchedule]) -> Dict:
        """
        执行调度并计算结果
        
        Args:
            schedules: 无人机调度方案
            
        Returns:
            调度执行结果
        """
        total_service_count = 0
        total_lockers_served = 0
        max_completion_time = 0.0
        total_flight_distance = 0.0
        
        served_lockers = set()
        deployment_details = []
        
        for schedule in schedules:
            if not schedule.tasks:
                continue
                
            drone_details = {
                'drone_id': schedule.drone_id,
                'tasks': [],
                'total_time': schedule.total_time,
                'service_count': schedule.total_service_count
            }
            
            for task in schedule.tasks:
                # 记录服务统计
                total_service_count += (task.delivery_demand + task.return_demand)
                served_lockers.add(task.locker_id)
                total_flight_distance += task.distance * 2  # 往返距离
                
                # 记录任务详情
                task_detail = {
                    'locker_id': task.locker_id,
                    'location': task.location,
                    'delivery_demand': task.delivery_demand,
                    'return_demand': task.return_demand,
                    'distance': task.distance,
                    'estimated_time': task.estimated_time,
                    'priority': task.priority
                }
                drone_details['tasks'].append(task_detail)
            
            deployment_details.append(drone_details)
            max_completion_time = max(max_completion_time, schedule.total_time)
        
        total_lockers_served = len(served_lockers)
        
        # 计算超时惩罚
        if max_completion_time > self.max_service_time:
            self.overtime_penalty = (max_completion_time - self.max_service_time) * 10
        
        # 计算效率指标
        efficiency_metrics = self._calculate_efficiency_metrics(
            schedules, total_service_count, total_lockers_served, max_completion_time
        )
        
        # 计算总任务数
        total_tasks = sum(len(schedule.tasks) for schedule in schedules)
        
        # 计算效率评分
        efficiency_score = 0.0
        if total_tasks > 0 and max_completion_time > 0:
            efficiency_score = (total_service_count / total_tasks) / (max_completion_time / 60.0)
        
        return {
            'success': True,
            'total_tasks': total_tasks,
            'total_service_count': total_service_count,
            'total_lockers_served': total_lockers_served,
            'max_completion_time': max_completion_time,
            'overtime_penalty': self.overtime_penalty,
            'total_flight_distance': total_flight_distance,
            'active_drones': len(schedules),
            'efficiency_score': efficiency_score,
            'schedules': deployment_details,
            'deployment_details': deployment_details,
            'efficiency_metrics': efficiency_metrics,
            'served_locker_ids': list(served_lockers)
        }
    
    def _calculate_efficiency_metrics(self, schedules: List[DroneSchedule], 
                                     total_service_count: int,
                                     total_lockers_served: int, 
                                     max_completion_time: float) -> Dict:
        """
        计算效率指标
        
        Args:
            schedules: 调度方案
            total_service_count: 总服务数量
            total_lockers_served: 总服务快递柜数
            max_completion_time: 最大完成时间
            
        Returns:
            效率指标字典
        """
        if max_completion_time <= 0:
            return {
                'service_rate': 0.0,
                'time_utilization': 0.0,
                'drone_utilization': 0.0,
                'average_efficiency': 0.0
            }
        
        # 服务效率：每秒服务的需求数量
        service_rate = total_service_count / max_completion_time
        
        # 时间利用率：实际使用时间 / 最大允许时间
        time_utilization = min(max_completion_time / self.max_service_time, 1.0)
        
        # 无人机利用率：活跃无人机数 / 总无人机数
        drone_utilization = len(schedules) / self.num_drones
        
        # 平均效率：所有无人机的平均效率分数
        if schedules:
            average_efficiency = sum(s.efficiency_score for s in schedules) / len(schedules)
        else:
            average_efficiency = 0.0
        
        return {
            'service_rate': service_rate,
            'time_utilization': time_utilization,
            'drone_utilization': drone_utilization,
            'average_efficiency': average_efficiency
        }
    
    def _create_empty_result(self) -> Dict:
        """创建空结果"""
        return {
            'success': False,
            'total_tasks': 0,
            'total_service_count': 0,
            'total_lockers_served': 0,
            'max_completion_time': 0.0,
            'overtime_penalty': 0.0,
            'total_flight_distance': 0.0,
            'active_drones': 0,
            'efficiency_score': 0.0,
            'schedules': [],
            'deployment_details': [],
            'efficiency_metrics': {
                'service_rate': 0.0,
                'time_utilization': 0.0,
                'drone_utilization': 0.0,
                'average_efficiency': 0.0
            },
            'served_locker_ids': []
        }
    
    def _adjust_task_priorities_by_rl(self, tasks: List[DroneTask], 
                                     rl_preferences: Dict) -> List[DroneTask]:
        """
        根据强化学习偏好调整任务优先级
        
        Args:
            tasks: 原始任务列表
            rl_preferences: RL偏好设置
            
        Returns:
            调整后的任务列表
        """
        # 获取RL偏好参数
        distance_weight = rl_preferences.get('distance_preference', 1.0)
        demand_weight = rl_preferences.get('demand_preference', 1.0)
        efficiency_weight = rl_preferences.get('efficiency_preference', 1.0)
        
        # 调整任务优先级
        for task in tasks:
            # 基于RL偏好重新计算优先级
            distance_factor = 1.0 / (1.0 + task.distance * distance_weight)
            demand_factor = (task.delivery_demand + task.return_demand) * demand_weight
            efficiency_factor = task.efficiency_score if hasattr(task, 'efficiency_score') else 1.0
            
            # 重新计算优先级
            task.priority = (distance_factor * demand_factor * efficiency_factor * efficiency_weight)
        
        # 按调整后的优先级排序
        return sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    def _rl_select_drone(self, task: DroneTask, schedules: List[DroneSchedule], 
                        rl_preferences: Dict) -> Optional[int]:
        """
        基于强化学习偏好选择无人机
        
        Args:
            task: 待分配任务
            schedules: 当前调度方案
            rl_preferences: RL偏好设置
            
        Returns:
            选中的无人机索引，如果无法分配则返回None
        """
        # 获取RL偏好参数
        load_balance_weight = rl_preferences.get('load_balance_preference', 1.0)
        time_efficiency_weight = rl_preferences.get('time_efficiency_preference', 1.0)
        
        best_drone_idx = None
        best_score = float('-inf')
        
        for i, schedule in enumerate(schedules):
            # 检查时间约束
            if schedule.total_time + task.estimated_time > self.max_service_time:
                continue
            
            # 计算RL引导的分配得分
            load_balance_score = 1.0 / (1.0 + schedule.total_service_count) * load_balance_weight
            time_efficiency_score = 1.0 / (1.0 + schedule.total_time) * time_efficiency_weight
            
            total_score = load_balance_score + time_efficiency_score
            
            if total_score > best_score:
                best_score = total_score
                best_drone_idx = i
        
        return best_drone_idx
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """计算两点间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_scheduling_statistics(self) -> Dict:
        """
        获取调度统计信息
        
        Returns:
            调度统计字典
        """
        return {
            'max_service_time': self.max_service_time,
            'drone_speed': self.drone_speed,
            'service_time_per_item': self.service_time_per_item,
            'num_drones': self.num_drones,
            'current_time': self.current_time,
            'overtime_penalty': self.overtime_penalty,
            'active_schedules_count': len(self.active_schedules),
            'completed_tasks_count': len(self.completed_tasks),
            'rl_exploration_rate': self.rl_exploration_rate,
            'rl_learning_enabled': self.rl_learning_enabled,
            'rl_decision_history_count': len(self.rl_decision_history)
        }
    
    def _swap_random_tasks(self, schedules: List[DroneSchedule]) -> List[DroneSchedule]:
        """
        随机交换两个无人机的任务
        
        Args:
            schedules: 原始调度方案
            
        Returns:
            交换任务后的调度方案
        """
        if len(schedules) < 2:
            return schedules
        
        # 选择两个有任务的无人机
        valid_drones = [i for i, s in enumerate(schedules) if s.tasks]
        if len(valid_drones) < 2:
            return schedules
        
        # 随机选择两个无人机
        drone1_idx, drone2_idx = np.random.choice(valid_drones, 2, replace=False)
        
        # 随机选择要交换的任务
        if schedules[drone1_idx].tasks and schedules[drone2_idx].tasks:
            task1_idx = np.random.randint(len(schedules[drone1_idx].tasks))
            task2_idx = np.random.randint(len(schedules[drone2_idx].tasks))
            
            # 交换任务
            task1 = schedules[drone1_idx].tasks[task1_idx]
            task2 = schedules[drone2_idx].tasks[task2_idx]
            
            schedules[drone1_idx].tasks[task1_idx] = task2
            schedules[drone2_idx].tasks[task2_idx] = task1
            
            # 重新计算时间
            self._recalculate_schedule_time(schedules[drone1_idx])
            self._recalculate_schedule_time(schedules[drone2_idx])
        
        return schedules
    
    def _redistribute_tasks(self, schedules: List[DroneSchedule]) -> List[DroneSchedule]:
        """
        重新分配部分任务
        
        Args:
            schedules: 原始调度方案
            
        Returns:
            重新分配后的调度方案
        """
        # 收集所有任务
        all_tasks = []
        for schedule in schedules:
            all_tasks.extend(schedule.tasks)
        
        if not all_tasks:
            return schedules
        
        # 随机选择要重新分配的任务数量（10-30%）
        redistribute_count = max(1, int(len(all_tasks) * np.random.uniform(0.1, 0.3)))
        tasks_to_redistribute = np.random.choice(all_tasks, redistribute_count, replace=False)
        
        # 从原调度中移除这些任务
        for schedule in schedules:
            schedule.tasks = [t for t in schedule.tasks if t not in tasks_to_redistribute]
            self._recalculate_schedule_time(schedule)
        
        # 重新分配这些任务
        for task in tasks_to_redistribute:
            # 找到最适合的无人机
            best_drone_idx = None
            best_score = float('-inf')
            
            for i, schedule in enumerate(schedules):
                if schedule.total_time + task.estimated_time <= self.max_service_time:
                    # 简单的负载均衡得分
                    score = 1.0 / (1.0 + schedule.total_service_count)
                    if score > best_score:
                        best_score = score
                        best_drone_idx = i
            
            if best_drone_idx is not None:
                self._assign_task_to_drone(task, schedules[best_drone_idx])
        
        return schedules
    
    def _shuffle_task_order(self, schedules: List[DroneSchedule]) -> List[DroneSchedule]:
        """
        轻微调整任务顺序
        
        Args:
            schedules: 原始调度方案
            
        Returns:
            调整顺序后的调度方案
        """
        for schedule in schedules:
            if len(schedule.tasks) > 1:
                # 随机打乱任务顺序
                np.random.shuffle(schedule.tasks)
        
        return schedules
    
    def _recalculate_schedule_time(self, schedule: DroneSchedule):
        """
        重新计算调度方案的时间
        
        Args:
            schedule: 调度方案
        """
        schedule.total_time = sum(task.estimated_time for task in schedule.tasks)
        schedule.total_service_count = sum(
            task.delivery_demand + task.return_demand for task in schedule.tasks
        )
    
    def get_rl_decision_history(self) -> List[Dict]:
        """
        获取强化学习决策历史
        
        Returns:
            决策历史列表
        """
        return self.rl_decision_history.copy()
    
    def reset_rl_history(self):
        """重置强化学习决策历史"""
        self.rl_decision_history.clear()
    
    def update_rl_parameters(self, exploration_rate: Optional[float] = None,
                           learning_enabled: Optional[bool] = None):
        """
        更新强化学习参数
        
        Args:
            exploration_rate: 新的探索率
            learning_enabled: 是否启用强化学习
        """
        if exploration_rate is not None:
            self.rl_exploration_rate = max(0.0, min(1.0, exploration_rate))
        
        if learning_enabled is not None:
            self.rl_learning_enabled = learning_enabled