# truck_routing.py
import time
import config
from tqdm import tqdm
from config import Config
from generate_model_dataset import (  # 从文档1导入必要的函数
    calculate_cp, calculate_di, calculate_sd, calculate_ci,
    calculate_ldp, calculate_ic, calculate_dle, calculate_dii,
    calculate_cci, calculate_cli
)
from typing import List, Tuple, Dict, Any, Optional
from torch.distributions import Categorical, Bernoulli
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from sklearn.linear_model import LinearRegression
import random
from reward_function import RewardFunction, AdaptiveRewardScheduler
from state_representation import StateRepresentation, TimeWindowConstraints
from action_mask import ActionMaskManager
from demand_model import UncertaintyHandler
from soft_time_window import SoftTimeWindowManager, TimeWindow, TimeWindowOptimizer, PenaltyFunction
from truck_replenishment import (ReplenishmentOptimizer, ReplenishmentStrategy, TruckState, 
                                LockerDemand, ReplenishmentDecision)
from dynamic_drone_scheduler import DynamicDroneScheduler
from dynamic_step_implementation import dynamic_step, get_serviceable_lockers
from dataclasses import dataclass
from enum import Enum

num_lockers = config.num_lockers


# ======================
# 课程学习策略
# ======================
class DifficultyLevel(Enum):
    """难度等级"""
    BEGINNER = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class CurriculumStage:
    """课程学习阶段配置"""
    name: str
    difficulty: DifficultyLevel
    num_lockers: int
    num_trucks: int
    demand_variance: float
    time_pressure: float
    episodes_required: int
    success_threshold: float


class CurriculumManager:
    """
    课程学习管理器
    实现从简单到复杂的渐进式训练，稳定学习曲线，提高收敛速度和鲁棒性
    """
    
    def __init__(self, max_lockers: int = 15, max_trucks: int = 4, start_difficulty: str = "expert"):
        """
        初始化课程学习管理器
        
        Args:
            max_lockers: 最大快递柜数量
            max_trucks: 最大卡车数量
            start_difficulty: 起始难度级别 (beginner, easy, medium, hard, expert)
        """
        self.max_lockers = max_lockers
        self.max_trucks = max_trucks
        
        # 定义课程阶段 - 总计约10万轮训练，专家阶段占主要比重
        self.stages = [
            CurriculumStage("初学者", DifficultyLevel.BEGINNER, 3, 1, 0.1, 0.5, 2000, 0.6),    # 2K轮：基础入门
            CurriculumStage("简单", DifficultyLevel.EASY, 6, 2, 0.2, 0.6, 5000, 0.65),         # 5K轮：双车协调
            CurriculumStage("中等", DifficultyLevel.MEDIUM, 9, 3, 0.3, 0.7, 8000, 0.7),        # 8K轮：三车协调
            CurriculumStage("困难", DifficultyLevel.HARD, 12, 3, 0.4, 0.8, 15000, 0.75),       # 15K轮：复杂场景
            CurriculumStage("专家", DifficultyLevel.EXPERT, max_lockers, max_trucks, 0.5, 1.0, 70000, 0.8)  # 70K轮：专家级训练
        ]
        
        # 根据起始难度设置当前阶段
        difficulty_map = {
            "beginner": 0,
            "easy": 1, 
            "medium": 2,
            "hard": 3,
            "expert": 4
        }
        self.current_stage_index = difficulty_map.get(start_difficulty.lower(), 4)  # 默认专家级
        self.current_stage = self.stages[self.current_stage_index]
        self.episodes_in_stage = 0
        self.performance_history = []
        self.performance_window = 50
        
    def get_current_config(self) -> Dict[str, Any]:
        """
        获取当前阶段配置
        
        Returns:
            环境配置字典
        """
        stage = self.current_stage
        return {
            'num_lockers': stage.num_lockers,
            'num_trucks': stage.num_trucks,
            'demand_variance': stage.demand_variance,
            'time_pressure': stage.time_pressure,
            'difficulty_level': stage.difficulty.value,
            'stage_name': stage.name
        }
    
    def update_performance(self, episode_reward: float, episode_success: bool):
        """
        更新性能记录
        
        Args:
            episode_reward: 回合奖励
            episode_success: 回合是否成功
        """
        # 改进的性能计算：基于当前阶段的期望奖励范围
        stage_difficulty = self.current_stage.difficulty.value
        
        # 根据阶段调整期望奖励范围 - 修正为更符合实际情况的范围
        if stage_difficulty <= 1:  # 初学者和简单阶段
            expected_min, expected_max = -10, 50
        elif stage_difficulty == 2:  # 中等阶段
            expected_min, expected_max = -5, 80
        elif stage_difficulty == 3:  # 困难阶段
            expected_min, expected_max = 0, 120
        else:  # 专家阶段
            expected_min, expected_max = 5, 150
        
        # 计算基础性能分数 (0-1)
        performance_score = max(0, min(1, (episode_reward - expected_min) / (expected_max - expected_min)))
        
        # 成功奖励：根据阶段难度调整
        if episode_success:
            success_bonus = 0.4 - stage_difficulty * 0.05  # 难度越高，成功奖励越小
            performance_score = min(1.0, performance_score + success_bonus)
        
        self.performance_history.append(performance_score)
        self.episodes_in_stage += 1
        
        # 检查阶段转换
        self._check_stage_transition()
    
    def _check_stage_transition(self):
        """检查是否需要阶段转换"""
        # 提高检查要求，确保充分训练
        min_episodes_for_check = max(100, self.performance_window)  # 最少100个episodes进行评估
        
        if len(self.performance_history) < min_episodes_for_check:
            return
        
        recent_performance = np.mean(self.performance_history[-min_episodes_for_check:])
        
        # 更严格的前进条件，确保每个阶段充分训练
        min_episodes_required = max(self.current_stage.episodes_required // 2, 500)  # 至少500个episodes
        
        # 检查是否可以前进
        if (recent_performance >= self.current_stage.success_threshold and 
            self.episodes_in_stage >= min_episodes_required and
            self.current_stage_index < len(self.stages) - 1):
            
            print(f"\n🎓 课程学习：从 '{self.current_stage.name}' 前进到下一阶段")
            print(f"当前性能: {recent_performance:.3f}, 目标: {self.current_stage.success_threshold:.3f}")
            print(f"已完成episodes: {self.episodes_in_stage}, 最少要求: {min_episodes_required}")
            
            self.current_stage_index += 1
            self.current_stage = self.stages[self.current_stage_index]
            self.episodes_in_stage = 0
            self.performance_history = []
            
            print(f"新阶段: '{self.current_stage.name}' (难度: {self.current_stage.difficulty.value})")
            return True
        
        # 静默等待，不输出冗余信息
        
        return False
    
    def get_adaptive_hyperparameters(self) -> Dict[str, float]:
        """
        获取自适应超参数
        
        Returns:
            超参数字典
        """
        difficulty = self.current_stage.difficulty.value
        
        # 根据难度调整学习率 - 降低学习率以提高稳定性
        learning_rates = [2e-4, 1.5e-4, 1e-4, 8e-5, 5e-5]  # 大幅降低学习率
        exploration_rates = [0.25, 0.2, 0.15, 0.1, 0.08]   # 降低探索率
        batch_sizes = [64, 64, 128, 256, 256]               # 增加批次大小
        
        return {
            'learning_rate': learning_rates[difficulty],
            'exploration_rate': exploration_rates[difficulty],
            'batch_size': batch_sizes[difficulty],
            'entropy_coef': exploration_rates[difficulty] * 0.3,  # 降低熵系数
            'value_loss_coef': 0.3,                              # 降低价值损失权重
            'max_grad_norm': 0.3                                 # 更严格的梯度裁剪
        }
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """
        获取课程进度
        
        Returns:
            进度信息字典
        """
        recent_performance = 0.0
        if len(self.performance_history) >= self.performance_window:
            recent_performance = np.mean(self.performance_history[-self.performance_window:])
        elif len(self.performance_history) > 0:
            recent_performance = np.mean(self.performance_history)
        
        return {
            'current_stage': self.current_stage.name,
            'stage_index': self.current_stage_index,
            'total_stages': len(self.stages),
            'episodes_in_stage': self.episodes_in_stage,
            'required_episodes': self.current_stage.episodes_required,
            'recent_performance': recent_performance,
            'target_performance': self.current_stage.success_threshold,
            'difficulty_level': self.current_stage.difficulty.value
        }
    
    def should_use_reward_shaping(self) -> bool:
        """
        判断是否使用奖励塑形
        
        Returns:
            是否使用奖励塑形
        """
        return self.current_stage.difficulty.value <= 1  # 前两个阶段使用


# ======================
# 无人机惩罚预测模型
# ======================
class DronePenaltyPredictor:
    def __init__(self):
        """
        无人机路径规划惩罚预测模型
        """
        # 加载预训练的线性回归模型
        self.model = LinearRegression()
        self._load_trained_model()

        # 特征列顺序（必须与训练时一致）
        self.feature_order = [
            'CP', 'DI', 'SD', 'CI', 'LDP', 'IC',
            'DLE', 'DII', 'CCI', 'CLI'
        ]

    def _load_trained_model(self):
        """加载训练好的模型参数"""
        # 创建虚拟模型参数
        self.model.coef_ = np.array([
            0.12, -0.08, 0.05, 0.07, 0.15,
            -0.03, 0.09, -0.06, 0.04, 0.11
        ])
        self.model.intercept_ = 2.5

    def predict(self, truck_location, service_area):
        """
        预测无人机路径规划的惩罚

        参数:
        truck_location: 卡车停靠点位置 (x, y)
        service_area: 服务区域内的快递柜信息列表
                      [{'location': (x,y), 'delivery_demand': int, 'return_demand': int}, ...]

        返回:
        penalty: 预测的无人机路径规划惩罚
        """
        # 计算特征
        features = self._calculate_features(truck_location, service_area)

        # 预测惩罚
        penalty = self.model.predict([features])[0]

        # 确保惩罚值为正
        return max(penalty, 0)

    def _calculate_features(self, truck_location, service_area):
        """
        计算无人机路径规划特征

        参数:
        truck_location: 卡车停靠点位置 (x, y)
        service_area: 服务区域内的快递柜列表

        返回:
        features: 特征向量
        """
        # 如果没有快递柜，返回零向量
        if not service_area:
            return np.zeros(len(self.feature_order))

        # 准备快递柜数据
        lockers = []
        for locker in service_area:
            x, y = locker['location']
            delivery = locker['delivery_demand']
            return_d = locker['return_demand']
            lockers.append((x, y, delivery, return_d))

        # 设置中心点
        center = truck_location

        # 计算特征
        features = {
            'CP': calculate_cp(lockers, center),
            'DI': calculate_di(lockers),
            'SD': calculate_sd(lockers),
            'CI': calculate_ci(lockers),
            'LDP': calculate_ldp(lockers, center),
            'IC': calculate_ic(lockers),
            'DLE': calculate_dle(lockers, center),
            'DII': calculate_dii(lockers),
            'CCI': calculate_cci(lockers, center),
            'CLI': calculate_cli(lockers, center)
        }

        # 按指定顺序返回特征向量
        return np.array([features[col] for col in self.feature_order])


class TruckSchedulingEnv:
    def __init__(self, verbose=False):
        """
        卡车调度环境 - 优化版
        """
        # 配置参数
        self.depot = Config.DEPOT  # 仓库位置
        self.drone_max_range = Config.DRONE_MAX_RANGE  # 无人机最大续航距离
        self.truck_capacity = Config.TRUCK_CAPACITY  # 卡车容量
        self.penalty_weight = Config.PENALTY_WEIGHT  # 无人机惩罚权重
        self.max_timesteps = Config.MAX_TIMESTEPS  # 最大时间步
        self.num_lockers = config.num_lockers
        self.lockers_info = config.locker_info
        self.verbose = verbose

        # 计算总期望需求
        self.total_lambda_del = sum(self.lockers_info[2])
        self.total_lambda_ret = sum(self.lockers_info[3])

        # 计算初始装载比例和卡车数量
        self.initial_load_ratio = self.total_lambda_del / (self.total_lambda_del + self.total_lambda_ret)
        self.initial_delivery_load = int(self.initial_load_ratio * self.truck_capacity)
        self.num_trucks = max(1, math.ceil(self.total_lambda_del / self.initial_delivery_load))

        # 初始化无人机预测模型
        self.drone_predictor = DronePenaltyPredictor()

        # 初始化新的奖励函数
        self.reward_function = RewardFunction(max_timesteps=self.max_timesteps)
        self.reward_scheduler = AdaptiveRewardScheduler(self.reward_function)

        # 初始化增强状态表示
        self.state_representation = StateRepresentation(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity,
            depot_location=self.depot,
            max_timesteps=self.max_timesteps
        )

        # 初始化时间窗约束
        self.time_window_constraints = TimeWindowConstraints(
            num_lockers=self.num_lockers,
            soft_penalty_factor=0.1
        )
        
        # 初始化动作掩码管理器
        self.action_mask_manager = ActionMaskManager(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity,
            depot_location=self.depot,
            max_distance=100.0
        )
        
        # 初始化不确定性处理器
        self.uncertainty_handler = UncertaintyHandler(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity
        )
        
        # 初始化软时间窗管理器
        self.soft_time_window_manager = SoftTimeWindowManager(
            penalty_function=PenaltyFunction.QUADRATIC,
            early_penalty_weight=0.5,
            late_penalty_weight=2.0,
            max_penalty=50.0
        )
        
        # 初始化时间窗优化器
        self.time_window_optimizer = TimeWindowOptimizer(self.soft_time_window_manager)
        
        # 为每个快递柜设置时间窗
        self._initialize_time_windows()
        
        # 初始化补货优化器
        self.replenishment_optimizer = ReplenishmentOptimizer(
            truck_capacity=self.truck_capacity,
            depot_location=self.depot,
            strategy=ReplenishmentStrategy.ADAPTIVE
        )
        
        # 初始化动态无人机调度器
        self.drone_scheduler = DynamicDroneScheduler(
            max_service_time=300,  # 300秒时间窗
            drone_speed=Config.DRONE_SPEED,
            service_time_per_item=Config.DRONE_SERVICE_TIME
        )
        
        # 初始化增强的多卡车协调组件
        self.coordination_history = []  # 协调历史记录
        self.truck_performance_metrics = {}  # 卡车性能指标
        self.global_coordination_info = {}  # 全局协调信息
        self.load_balancing_weights = np.ones(self.num_trucks)  # 负载均衡权重
        self.coordination_update_interval = 5  # 协调更新间隔
        self.last_coordination_update = 0  # 上次协调更新时间
        
        # 初始化卡车性能跟踪
        for truck_id in range(self.num_trucks):
            self.truck_performance_metrics[truck_id] = {
                'total_distance': 0.0,
                'total_service_time': 0.0,
                'items_delivered': 0,
                'items_returned': 0,
                'efficiency_score': 0.0,
                'load_utilization': 0.0,
                'recent_actions': [],
                'predicted_completion_time': 0.0
            }

        # 初始化维度（使用增强状态表示）
        self.state_dim = self.state_representation.get_state_dimension()

        if self.verbose:
            self.print_initial_info()

        self.reset()
    
    def update_curriculum_config(self, curriculum_config: Dict[str, Any]):
        """
        更新课程学习配置
        
        Args:
            curriculum_config: 课程配置字典，包含num_trucks, num_lockers, boundary等
        """
        config_changed = False
        
        # 更新边界参数
        if 'boundary' in curriculum_config:
            new_boundary = curriculum_config['boundary']
            import config
            if new_boundary != config.boundary:
                config.boundary = new_boundary
                config_changed = True
                # 重新生成快递柜信息以应用新边界
                config.generate_locker_info()
                self.lockers_info = config.locker_info
                
                # 重新计算总需求
                self.total_lambda_del = sum(self.lockers_info[2])
                self.total_lambda_ret = sum(self.lockers_info[3])
                
                # 重新计算初始装载比例
                self.initial_load_ratio = self.total_lambda_del / (self.total_lambda_del + self.total_lambda_ret)
                self.initial_delivery_load = int(self.initial_load_ratio * self.truck_capacity)
        
        # 更新快递柜数量
        if 'num_lockers' in curriculum_config:
            new_num_lockers = curriculum_config['num_lockers']
            if new_num_lockers != self.num_lockers:
                self.num_lockers = new_num_lockers
                config_changed = True
                # 重新生成快递柜信息
                import config
                config.num_lockers = new_num_lockers
                config.generate_locker_info()
                self.lockers_info = config.locker_info
                
                # 重新计算总需求
                self.total_lambda_del = sum(self.lockers_info[2])
                self.total_lambda_ret = sum(self.lockers_info[3])
                
                # 重新计算初始装载比例
                self.initial_load_ratio = self.total_lambda_del / (self.total_lambda_del + self.total_lambda_ret)
                self.initial_delivery_load = int(self.initial_load_ratio * self.truck_capacity)
        
        # 更新卡车数量
        if 'num_trucks' in curriculum_config:
            new_num_trucks = curriculum_config['num_trucks']
            if new_num_trucks is None:
                # 动态计算卡车数量
                calculated_trucks = max(1, math.ceil(self.total_lambda_del / self.initial_delivery_load))
                if calculated_trucks != self.num_trucks:
                    self.num_trucks = calculated_trucks
                    config_changed = True
            elif new_num_trucks != self.num_trucks:
                self.num_trucks = new_num_trucks
                config_changed = True
        
        # 如果配置发生变化，重新初始化相关组件
        if config_changed:
            self._reinitialize_components()
            
            if self.verbose:
                print(f"🚛 课程学习配置更新:")
                print(f"   快递柜数量: {self.num_lockers}")
                print(f"   卡车数量: {self.num_trucks}")
                if 'boundary' in curriculum_config:
                    print(f"   边界范围: ±{curriculum_config['boundary']}")
                print(f"   状态维度: {self.state_dim}")
                print(f"   总配送需求: {self.total_lambda_del:.2f}")
                print(f"   总返回需求: {self.total_lambda_ret:.2f}")
                print(f"   初始装载量: {self.initial_delivery_load}")
    
    def _reinitialize_components(self):
        """
        重新初始化依赖于卡车数量的组件
        """
        # 重新初始化状态表示
        self.state_representation = StateRepresentation(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity,
            depot_location=self.depot,
            max_timesteps=self.max_timesteps
        )
        
        # 重新初始化动作掩码管理器
        self.action_mask_manager = ActionMaskManager(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity,
            depot_location=self.depot,
            max_distance=100.0
        )
        
        # 重新初始化不确定性处理器
        self.uncertainty_handler = UncertaintyHandler(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity
        )
        
        # 重新初始化负载均衡权重
        self.load_balancing_weights = np.ones(self.num_trucks)
        
        # 重新初始化卡车性能跟踪
        self.truck_performance_metrics = {}
        for truck_id in range(self.num_trucks):
            self.truck_performance_metrics[truck_id] = {
                'total_distance': 0.0,
                'total_service_time': 0.0,
                'items_delivered': 0,
                'items_returned': 0,
                'efficiency_score': 0.0,
                'load_utilization': 0.0,
                'recent_actions': [],
                'predicted_completion_time': 0.0
            }
        
        # 更新状态维度
        self.state_dim = self.state_representation.get_state_dimension()

    def _euclidean_distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def _initialize_time_windows(self):
        """
        初始化每个快递柜的时间窗约束
        根据快递柜的位置、需求量等因素设置合理的时间窗
        """
        for locker_id in range(self.num_lockers):
            # 获取快递柜信息
            locker_x, locker_y = self.lockers_info[0][locker_id], self.lockers_info[1][locker_id]
            lambda_del = self.lockers_info[2][locker_id]  # 配送需求率
            lambda_ret = self.lockers_info[3][locker_id]  # 取件需求率
            
            # 计算距离仓库的距离
            distance_to_depot = self._euclidean_distance((locker_x, locker_y), self.depot)
            
            # 根据距离和需求量计算基础时间窗
            # 距离越远，时间窗越宽松；需求量越大，优先级越高
            base_travel_time = distance_to_depot / Config.TRUCK_SPEED  # 使用配置的卡车速度
            demand_priority = (lambda_del + lambda_ret) / (self.total_lambda_del + self.total_lambda_ret)
            
            # 设置时间窗参数
            if demand_priority > 0.15:  # 高需求快递柜
                early_start = max(0, base_travel_time - 2)
                preferred_start = base_travel_time
                preferred_end = base_travel_time + 3
                late_end = base_travel_time + 8
                priority = 1.5
            elif demand_priority > 0.08:  # 中等需求快递柜
                early_start = max(0, base_travel_time - 3)
                preferred_start = base_travel_time
                preferred_end = base_travel_time + 5
                late_end = base_travel_time + 12
                priority = 1.0
            else:  # 低需求快递柜
                early_start = max(0, base_travel_time - 5)
                preferred_start = base_travel_time
                preferred_end = base_travel_time + 8
                late_end = base_travel_time + 20
                priority = 0.7
            
            # 创建时间窗对象
            time_window = TimeWindow(
                early_start=early_start,
                preferred_start=preferred_start,
                preferred_end=preferred_end,
                late_end=late_end,
                service_time=1.0,  # 假设服务时间为1个时间单位
                priority=priority
            )
            
            # 设置到软时间窗管理器
            self.soft_time_window_manager.set_time_window(locker_id, time_window)



    def print_initial_info(self):
        """打印初始配置信息"""
        print("\n" + "=" * 50)
        print("初始卡车配置信息:")
        print(f"快递柜数量: {self.num_lockers}")
        print(f"总期望取货需求: {self.total_lambda_del:.2f}")
        print(f"总期望退货需求: {self.total_lambda_ret:.2f}")
        print(f"初始装载比例: {self.initial_load_ratio * 100:.2f}%")
        print(f"卡车容量: {self.truck_capacity}")
        print(f"卡车数量: {self.num_trucks}")
        print(f"每辆卡车初始取货货物: {self.initial_delivery_load}")
        print("=" * 50)

    def reset(self):
        """重置环境到初始状态"""
        # 初始化时间步
        self.current_timestep = 0
        
        # 添加重置计数器，用于控制环境变化频率
        if not hasattr(self, 'reset_count'):
            self.reset_count = 0
        self.reset_count += 1
        
        # 每个episode都重新生成随机需求，确保训练多样性
        import config
        if self.reset_count == 1:
            # 第一次重置：生成快递柜位置和需求
            config.generate_locker_info()
            self.lockers_info = config.locker_info
        else:
            # 后续重置：保持快递柜位置不变，但每次都重新生成随机需求
            config.generate_demand_only()
            # 更新需求信息但保持位置不变
            self.lockers_info[2] = list(config.lambda_del.values())  # lambda_del
            self.lockers_info[3] = list(config.lambda_ret.values())  # lambda_ret
            self.lockers_info[4] = list(config.demand_del.values())  # demand_del
            self.lockers_info[5] = list(config.demand_ret.values())  # demand_ret
        
        # 重新计算总需求
        self.total_lambda_del = sum(self.lockers_info[2])
        self.total_lambda_ret = sum(self.lockers_info[3])
        
        # 重新计算初始装载比例，添加随机性
        base_load_ratio = self.total_lambda_del / (self.total_lambda_del + self.total_lambda_ret)
        # 在基础比例上添加±10%的随机变化
        random_factor = random.uniform(0.9, 1.1)
        self.initial_load_ratio = min(0.9, max(0.1, base_load_ratio * random_factor))
        self.initial_delivery_load = int(self.initial_load_ratio * self.truck_capacity)
        
        # 重新初始化不确定性处理器，确保需求生成的多样性
        self.uncertainty_handler = UncertaintyHandler(
            num_trucks=self.num_trucks,
            num_lockers=self.num_lockers,
            truck_capacity=self.truck_capacity
        )
        
        # 初始化所有快递柜为未服务状态
        self.lockers_state = []
        for i in range(self.num_lockers):
            locker_id = i + 1
            
            # 直接使用config中重新生成的需求值，确保每次重置都有新的需求分布
            actual_demand_del = self.lockers_info[4][i]  # 直接使用config中的demand_del
            actual_demand_ret = self.lockers_info[5][i]  # 直接使用config中的demand_ret
            
            # 使用UncertaintyHandler生成不确定性信息（但不覆盖实际需求）
            delivery_estimate = self.uncertainty_handler.get_robust_demand_estimate(
                locker_id, 'delivery', self.current_timestep
            )
            return_estimate = self.uncertainty_handler.get_robust_demand_estimate(
                locker_id, 'return', self.current_timestep
            )
            
            locker = {
                'id': locker_id,  # 快递柜ID从1开始
                'location': (self.lockers_info[0][i], self.lockers_info[1][i]),  # 位置
                'lambda_del': self.lockers_info[2][i],  # 送货需求率（基础）
                'lambda_ret': self.lockers_info[3][i],  # 退货需求率（基础）
                'demand_del': actual_demand_del,  # 使用config中重新生成的实际送货需求
                'demand_ret': actual_demand_ret,   # 使用config中重新生成的实际退货需求
                'expected_del': delivery_estimate['expected'],  # 期望送货需求
                'expected_ret': return_estimate['expected'],   # 期望退货需求
                'uncertainty_del': delivery_estimate['uncertainty'],  # 送货需求不确定性
                'uncertainty_ret': return_estimate['uncertainty'],   # 退货需求不确定性
                'served': False  # 初始化为未服务
            }
            self.lockers_state.append(locker)

        # 初始化卡车状态
        self.trucks = []
        for i in range(self.num_trucks):
            self.trucks.append({
                'id': i,
                'current_location': 0,  # 0表示仓库，其他为快递柜ID
                'position': (0, 0),     # 当前位置坐标
                'current_delivery_load': self.initial_delivery_load,
                'current_return_load': 0,
                'remaining_space': self.truck_capacity - self.initial_delivery_load,
                'capacity': self.truck_capacity,  # 添加容量字段
                'visited_stops': [],
                'total_distance': 0.0,
                'returned': False,  # 添加返回状态标志
                'service_start_time': None,  # 服务开始时间
                'service_end_time': None,    # 服务结束时间
                'is_servicing': False,       # 是否正在服务
                'time_at_location': 0,       # 在当前位置停留的时间
                'last_position': (0, 0),     # 上一个位置，用于检测位置变化
                'drone_deployments': []      # 无人机部署记录
            })

        # 环境状态
        self.time_step = 0
        self.total_truck_distance = 0.0
        self.total_drone_cost = 0.0
        self.served_delivery = 0
        self.served_return = 0
        
        # 回合级别统计变量（用于终端奖励计算）
        self.episode_truck_distance = 0.0      # 回合总卡车行驶距离
        self.episode_drone_cost = 0.0          # 回合总无人机成本
        self.episode_drone_deliveries = 0      # 回合总无人机配送完成次数
        self.episode_satisfied_lockers = 0     # 回合总完全满足需求的快递柜数量

        # 重置状态表示器
        self.state_representation.reset()

        return self._get_state_with_mask()

    def _get_action_mask(self):
        """创建动作屏蔽向量"""
        # 构建环境状态
        env_state = {
            'trucks': self.trucks,
            'lockers': self.lockers_state,
            'time_step': self.time_step,
            'max_timesteps': self.max_timesteps
        }
        
        # 使用动作掩码管理器获取所有卡车的掩码
        action_masks = self.action_mask_manager.get_action_masks(env_state)
        
        # 为了兼容现有代码，返回第一个卡车的掩码
        # 在多智能体训练中，应该使用get_action_masks方法
        if action_masks:
            return action_masks[0]
        else:
            # 返回默认掩码
            return {
                'stop_mask': torch.ones(self.num_lockers + 1),
                'service_mask': torch.ones(self.num_lockers)
            }
    
    def get_action_masks(self):
        """获取所有卡车的动作掩码"""
        env_state = {
            'trucks': self.trucks,
            'lockers': self.lockers_state,
            'time_step': self.time_step,
            'max_timesteps': self.max_timesteps
        }
        
        return self.action_mask_manager.get_action_masks(env_state)

    def _get_current_state(self):
        """
        获取当前环境状态信息，用于奖励计算
        """
        # 计算路径效率
        path_efficiency = self._calculate_path_efficiency()
        
        # 计算完成率
        completion_rate = self._calculate_completion_rate()
        
        # 计算容量利用率
        capacity_utilization = self._calculate_capacity_utilization()
        
        return {
            'time_step': self.time_step,
            'max_timesteps': self.max_timesteps,
            'trucks': [truck.copy() for truck in self.trucks],
            'lockers': [locker.copy() for locker in self.lockers_state],
            'served_delivery': self.served_delivery,
            'served_return': self.served_return,
            'total_truck_distance': self.total_truck_distance,
            'total_drone_cost': self.total_drone_cost,
            'num_trucks': self.num_trucks,
            'num_lockers': self.num_lockers,
            'truck_capacity': self.truck_capacity,
            'initial_delivery_load': self.initial_delivery_load,
            'path_efficiency': path_efficiency,
            'completion_rate': completion_rate,
            'capacity_utilization': capacity_utilization
        }

    def _get_state_with_mask(self):
        """获取当前环境状态向量和动作掩码"""
        # 使用新的状态表示器
        state_vector = self.state_representation.get_state_vector(
            trucks=self.trucks,
            lockers=self.lockers_state,
            time_step=self.time_step,
            total_distance=self.total_truck_distance,
            total_drone_cost=self.total_drone_cost
        )

        # 获取动作掩码
        action_mask = self._get_action_mask()

        return state_vector, action_mask

    def get_truck_specific_states(self):
        """
        为每个卡车生成增强的特定状态表示，包含全局协调信息和动态负载均衡
        
        Returns:
            List[np.ndarray]: 每个卡车的增强状态向量列表
        """
        # 更新全局协调信息
        self._update_global_coordination_info()
        
        # 更新卡车性能指标
        self._update_truck_performance_metrics()
        
        # 动态调整负载均衡权重
        self._update_load_balancing_weights()
        
        states = []
        
        # 构建增强的环境状态字典
        env_state = {
            'trucks': self.trucks,
            'lockers': self.lockers_state,
            'time_step': self.time_step,
            'max_timesteps': self.max_timesteps,
            'total_truck_distance': self.total_truck_distance,
            'total_drone_cost': self.total_drone_cost,
            'served_delivery': self.served_delivery,
            'served_return': self.served_return,
            'global_coordination_info': self.global_coordination_info,
            'truck_performance_metrics': self.truck_performance_metrics,
            'load_balancing_weights': self.load_balancing_weights
        }
        
        # 为每个卡车生成增强的特定状态
        for truck_id in range(self.num_trucks):
            # 获取基础状态
            base_state = self.state_representation.get_enhanced_state(env_state, truck_id)
            
            # 添加协调特征
            coordination_features = self._get_coordination_features(truck_id)
            
            # 添加预测特征
            prediction_features = self._get_prediction_features(truck_id)
            
            # 合并所有特征
            enhanced_state = np.concatenate([
                base_state,
                coordination_features,
                prediction_features
            ])
            
            states.append(enhanced_state)
        
        return states
    
    def _update_global_coordination_info(self):
        """
        更新全局协调信息，包括卡车分布、需求热点、协调冲突等
        """
        # 计算卡车分布密度
        truck_positions = [truck['position'] for truck in self.trucks]
        truck_density = self._calculate_truck_density(truck_positions)
        
        # 识别需求热点
        demand_hotspots = self._identify_demand_hotspots()
        
        # 计算协调冲突
        coordination_conflicts = self._detect_coordination_conflicts()
        
        # 计算全局负载分布
        global_load_distribution = self._calculate_global_load_distribution()
        
        # 预测未来需求趋势
        future_demand_trend = self._predict_future_demand_trend()
        
        self.global_coordination_info = {
            'truck_density': truck_density,
            'demand_hotspots': demand_hotspots,
            'coordination_conflicts': coordination_conflicts,
            'global_load_distribution': global_load_distribution,
            'future_demand_trend': future_demand_trend,
            'total_active_trucks': sum(1 for truck in self.trucks if truck['current_location'] != 0),
            'average_truck_distance': np.mean([truck.get('total_distance', 0) for truck in self.trucks]),
            'coordination_efficiency': self._calculate_coordination_efficiency()
        }
    
    def _update_truck_performance_metrics(self):
        """
        更新每个卡车的性能指标
        """
        for truck_id, truck in enumerate(self.trucks):
            metrics = self.truck_performance_metrics[truck_id]
            
            # 更新基础指标
            metrics['total_distance'] = truck.get('total_distance', 0)
            metrics['items_delivered'] = truck.get('delivery_items', 0)
            metrics['items_returned'] = truck.get('return_items', 0)
            
            # 计算负载利用率
            current_load = truck.get('delivery_items', 0) + truck.get('return_items', 0)
            metrics['load_utilization'] = current_load / self.truck_capacity if self.truck_capacity > 0 else 0
            
            # 计算效率分数
            if metrics['total_distance'] > 0:
                metrics['efficiency_score'] = (metrics['items_delivered'] + metrics['items_returned']) / metrics['total_distance']
            else:
                metrics['efficiency_score'] = 0
            
            # 预测完成时间
            metrics['predicted_completion_time'] = self._predict_truck_completion_time(truck_id)
            
            # 更新最近动作历史（保留最近10个动作）
            if len(metrics['recent_actions']) > 10:
                metrics['recent_actions'] = metrics['recent_actions'][-10:]
    
    def _update_load_balancing_weights(self):
        """
        动态更新负载均衡权重
        """
        if self.time_step - self.last_coordination_update >= self.coordination_update_interval:
            # 计算每个卡车的负载和效率
            truck_loads = []
            truck_efficiencies = []
            
            for truck_id in range(self.num_trucks):
                metrics = self.truck_performance_metrics[truck_id]
                truck_loads.append(metrics['load_utilization'])
                truck_efficiencies.append(metrics['efficiency_score'])
            
            # 标准化负载和效率
            if len(truck_loads) > 1:
                load_std = np.std(truck_loads) if np.std(truck_loads) > 0 else 1
                efficiency_mean = np.mean(truck_efficiencies) if np.mean(truck_efficiencies) > 0 else 1
                
                # 计算新的权重：低负载高效率的卡车获得更高权重
                for truck_id in range(self.num_trucks):
                    load_factor = 1.0 - truck_loads[truck_id]  # 负载越低权重越高
                    efficiency_factor = truck_efficiencies[truck_id] / efficiency_mean  # 效率越高权重越高
                    
                    self.load_balancing_weights[truck_id] = 0.6 * load_factor + 0.4 * efficiency_factor
                
                # 归一化权重
                weight_sum = np.sum(self.load_balancing_weights)
                if weight_sum > 0:
                    self.load_balancing_weights = self.load_balancing_weights / weight_sum * self.num_trucks
            
            self.last_coordination_update = self.time_step
    
    def _get_coordination_features(self, truck_id: int) -> np.ndarray:
        """
        获取指定卡车的协调特征
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            np.ndarray: 协调特征向量
        """
        features = []
        
        # 当前卡车的负载均衡权重
        features.append(self.load_balancing_weights[truck_id])
        
        # 与其他卡车的相对位置信息
        current_truck = self.trucks[truck_id]
        current_pos = current_truck['position']
        
        min_distance_to_other = float('inf')
        avg_distance_to_other = 0
        active_trucks_count = 0
        
        for other_id, other_truck in enumerate(self.trucks):
            if other_id != truck_id and other_truck['current_location'] != 0:
                distance = self._euclidean_distance(current_pos, other_truck['position'])
                min_distance_to_other = min(min_distance_to_other, distance)
                avg_distance_to_other += distance
                active_trucks_count += 1
        
        if active_trucks_count > 0:
            avg_distance_to_other /= active_trucks_count
        else:
            min_distance_to_other = 0
            avg_distance_to_other = 0
        
        features.extend([min_distance_to_other / 100.0, avg_distance_to_other / 100.0])  # 归一化
        
        # 全局协调信息
        coord_info = self.global_coordination_info
        features.extend([
            coord_info.get('total_active_trucks', 0) / self.num_trucks,
            coord_info.get('coordination_efficiency', 0),
            len(coord_info.get('coordination_conflicts', [])) / max(1, self.num_trucks),
            len(coord_info.get('demand_hotspots', [])) / max(1, self.num_lockers)
        ])
        
        # 相对性能指标
        my_metrics = self.truck_performance_metrics[truck_id]
        all_efficiencies = [self.truck_performance_metrics[i]['efficiency_score'] for i in range(self.num_trucks)]
        all_loads = [self.truck_performance_metrics[i]['load_utilization'] for i in range(self.num_trucks)]
        
        avg_efficiency = np.mean(all_efficiencies) if all_efficiencies else 0
        avg_load = np.mean(all_loads) if all_loads else 0
        
        relative_efficiency = my_metrics['efficiency_score'] - avg_efficiency
        relative_load = my_metrics['load_utilization'] - avg_load
        
        features.extend([relative_efficiency, relative_load])
        
        return np.array(features, dtype=np.float32)
    
    def _get_prediction_features(self, truck_id: int) -> np.ndarray:
        """
        获取指定卡车的预测特征
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            np.ndarray: 预测特征向量
        """
        features = []
        
        # 预测完成时间
        predicted_time = self.truck_performance_metrics[truck_id]['predicted_completion_time']
        normalized_time = predicted_time / self.max_timesteps if self.max_timesteps > 0 else 0
        features.append(normalized_time)
        
        # 未来需求趋势
        future_trend = self.global_coordination_info.get('future_demand_trend', {})
        features.extend([
            future_trend.get('delivery_trend', 0),
            future_trend.get('return_trend', 0),
            future_trend.get('hotspot_shift', 0)
        ])
        
        # 路径优化潜力
        path_optimization_potential = self._calculate_path_optimization_potential(truck_id)
        features.append(path_optimization_potential)
        
        # 协调机会评分
        coordination_opportunity = self._calculate_coordination_opportunity(truck_id)
        features.append(coordination_opportunity)
        
        return np.array(features, dtype=np.float32)

    def get_locker(self, locker_id):
        """根据ID获取快递柜"""
        for locker in self.lockers_state:
            if locker['id'] == locker_id:
                return locker
        return None

    def _calculate_path_efficiency(self) -> float:
        """
        计算路径效率 - 基于服务密度的动态理想步数评估
        
        核心原则: 基于服务密度动态计算理想步数，考虑无人机并行服务能力
        
        考虑因素:
        - 服务密度动态理想步数（基于快递柜分布和无人机覆盖范围）
        - 无人机并行服务效率（多无人机同时工作的时间优势）
        - 服务完成度和多卡车协调复杂度
        
        返回:
        - path_efficiency: 路径效率（0-1，期望值0.6-0.9为良好）
        """
        # 获取当前步数
        current_steps = getattr(self, 'time_step', 0)
        if current_steps <= 0:
            return 0.0
        
        # 获取已服务的快递柜数量
        served_lockers_count = sum(1 for locker in self.lockers_state if locker.get('served', False))
        if served_lockers_count == 0:
            return 0.0
        
        # 1. 基于服务密度计算动态理想步数
        ideal_steps = self._calculate_service_density_based_ideal_steps()
        
        if ideal_steps <= 0:
            return 0.0
        
        # 2. 基础步数效率
        step_efficiency = min(ideal_steps / current_steps, 1.0)
        
        # 3. 服务完成度奖励
        total_lockers = len(self.lockers_state)
        completion_rate = served_lockers_count / total_lockers
        completion_bonus = completion_rate * 0.3  # 最多30%的奖励
        
        # 4. 多卡车协调复杂度调整
        coordination_factor = 1.0 + (self.num_trucks - 1) * 0.1  # 每增加一辆卡车，期望提高10%
        
        # 5. 综合效率计算
        base_efficiency = step_efficiency + completion_bonus
        final_efficiency = base_efficiency * coordination_factor
        
        return min(final_efficiency, 1.0)
    
    def _calculate_service_density_based_ideal_steps(self) -> float:
        """
        基于服务密度计算动态理想步数
        
        核心思想：
        - 分析每个停靠点的服务密度（可服务快递柜数量和需求密度）
        - 考虑无人机并行服务能力和飞行时间
        - 根据快递柜分布优化停靠点选择，最小化总步数
        
        返回:
        - ideal_steps: 基于服务密度的理想步数
        """
        # 获取已服务的快递柜
        served_lockers = [locker for locker in self.lockers_state if locker.get('served', False)]
        if not served_lockers:
            return 0.0
        
        # 1. 分析服务密度分布
        service_density_analysis = self._analyze_service_density_distribution(served_lockers)
        
        # 2. 计算最优停靠点数量
        optimal_stops = self._calculate_optimal_stop_count(service_density_analysis)
        
        # 3. 考虑无人机并行服务时间
        drone_service_time = self._calculate_drone_parallel_service_time(service_density_analysis)
        
        # 4. 计算理想步数
        # 基础步数 = 停靠点数量（移动步数）
        base_steps = optimal_stops
        
        # 服务时间步数 = 无人机并行服务时间 / 每步时间
        # 假设每步代表Config.TRUCK_SERVICE_TIME的时间
        time_per_step = Config.TRUCK_SERVICE_TIME
        service_steps = max(1, int(drone_service_time / time_per_step))
        
        # 总理想步数 = 移动步数 + 服务步数
        ideal_steps = base_steps + service_steps
        
        # 5. 多卡车并行优化
        if self.num_trucks > 1:
            # 多卡车可以并行工作，减少总步数
            parallel_factor = min(self.num_trucks, optimal_stops)
            if parallel_factor > 1:
                ideal_steps = max(service_steps, ideal_steps / parallel_factor)
        
        return ideal_steps
    
    def _analyze_service_density_distribution(self, served_lockers: List[Dict]) -> Dict[str, Any]:
        """
        分析已服务快递柜的服务密度分布
        
        参数:
        - served_lockers: 已服务的快递柜列表
        
        返回:
        - 服务密度分析结果
        """
        if not served_lockers:
            return {'clusters': [], 'total_demand': 0, 'coverage_efficiency': 0.0}
        
        # 计算总需求量
        total_demand = sum(
            locker.get('demand_del', 0) + locker.get('demand_ret', 0) 
            for locker in served_lockers
        )
        
        # 使用聚类分析找到服务密度集中区域
        clusters = self._identify_service_clusters(served_lockers)
        
        # 计算覆盖效率
        coverage_efficiency = self._calculate_coverage_efficiency(clusters)
        
        return {
            'clusters': clusters,
            'total_demand': total_demand,
            'coverage_efficiency': coverage_efficiency,
            'served_count': len(served_lockers)
        }
    
    def _identify_service_clusters(self, served_lockers: List[Dict]) -> List[Dict]:
        """
        识别服务密度集中的区域（聚类）
        
        参数:
        - served_lockers: 已服务的快递柜列表
        
        返回:
        - 服务聚类列表
        """
        if not served_lockers:
            return []
        
        clusters = []
        drone_range = Config.DRONE_MAX_RANGE
        
        # 简单的基于距离的聚类算法
        unprocessed = served_lockers.copy()
        
        while unprocessed:
            # 选择第一个快递柜作为聚类中心
            center_locker = unprocessed.pop(0)
            cluster = {
                'center': center_locker['location'],
                'lockers': [center_locker],
                'total_demand': center_locker.get('demand_del', 0) + center_locker.get('demand_ret', 0),
                'coverage_radius': 0.0
            }
            
            # 找到在无人机范围内的其他快递柜
            remaining = []
            for locker in unprocessed:
                distance = np.sqrt(
                    (center_locker['location'][0] - locker['location'][0])**2 + 
                    (center_locker['location'][1] - locker['location'][1])**2
                )
                
                if distance <= drone_range:
                    cluster['lockers'].append(locker)
                    cluster['total_demand'] += locker.get('demand_del', 0) + locker.get('demand_ret', 0)
                    cluster['coverage_radius'] = max(cluster['coverage_radius'], distance)
                else:
                    remaining.append(locker)
            
            unprocessed = remaining
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_coverage_efficiency(self, clusters: List[Dict]) -> float:
        """
        计算服务覆盖效率
        
        参数:
        - clusters: 服务聚类列表
        
        返回:
        - 覆盖效率（0-1）
        """
        if not clusters:
            return 0.0
        
        total_efficiency = 0.0
        total_weight = 0.0
        
        for cluster in clusters:
            # 聚类效率 = 服务的快递柜数量 / 理论最大覆盖数量
            lockers_in_cluster = len(cluster['lockers'])
            
            # 理论最大覆盖：基于无人机数量和服务时间
            max_drones = Config.DRONE_NUM
            max_coverage = min(lockers_in_cluster, max_drones * 2)  # 假设每个无人机可以服务2个快递柜
            
            if max_coverage > 0:
                cluster_efficiency = lockers_in_cluster / max_coverage
                weight = cluster['total_demand']
                
                total_efficiency += cluster_efficiency * weight
                total_weight += weight
        
        return total_efficiency / total_weight if total_weight > 0 else 0.0
    
    def _calculate_optimal_stop_count(self, service_density_analysis: Dict[str, Any]) -> int:
        """
        计算最优停靠点数量
        
        参数:
        - service_density_analysis: 服务密度分析结果
        
        返回:
        - 最优停靠点数量
        """
        clusters = service_density_analysis.get('clusters', [])
        if not clusters:
            return 1
        
        # 基础停靠点数量 = 聚类数量
        base_stops = len(clusters)
        
        # 根据覆盖效率调整
        coverage_efficiency = service_density_analysis.get('coverage_efficiency', 0.0)
        
        # 如果覆盖效率低，可能需要更多停靠点
        if coverage_efficiency < 0.6:
            # 效率低时，增加停靠点
            adjustment_factor = 1.5
        elif coverage_efficiency > 0.8:
            # 效率高时，可以减少停靠点
            adjustment_factor = 0.8
        else:
            adjustment_factor = 1.0
        
        optimal_stops = max(1, int(base_stops * adjustment_factor))
        
        return optimal_stops
    
    def _calculate_drone_parallel_service_time(self, service_density_analysis: Dict[str, Any]) -> float:
        """
        计算无人机并行服务时间
        
        参数:
        - service_density_analysis: 服务密度分析结果
        
        返回:
        - 并行服务时间（秒）
        """
        clusters = service_density_analysis.get('clusters', [])
        if not clusters:
            return Config.DRONE_SERVICE_TIME
        
        max_service_time = 0.0
        
        for cluster in clusters:
            lockers_count = len(cluster['lockers'])
            total_demand = cluster['total_demand']
            
            # 计算该聚类的服务时间
            # 考虑无人机数量和并行能力
            available_drones = Config.DRONE_NUM
            
            # 每个无人机的服务时间 = 飞行时间 + 服务时间
            avg_flight_distance = cluster['coverage_radius']
            flight_time = (avg_flight_distance * 2) / Config.DRONE_SPEED  # 往返时间
            service_time_per_demand = Config.DRONE_SERVICE_TIME
            
            # 总服务时间 = 飞行时间 + 需求服务时间
            total_service_time_per_drone = flight_time + (total_demand * service_time_per_demand / available_drones)
            
            # 并行服务时间 = 最长的单个无人机服务时间
            cluster_service_time = total_service_time_per_drone
            
            max_service_time = max(max_service_time, cluster_service_time)
        
        return max_service_time
    
    def _calculate_demand_weighted_optimal_distance(self, served_lockers: List[Dict]) -> float:
        """
        计算需求量加权的理想路径距离
        
        核心思想:
        - 高需求量的快递柜应该优先访问（距离权重更低）
        - 路径规划应该考虑需求密度，而不仅仅是地理距离
        - 使用需求量对距离进行加权，反映实际的服务价值
        
        参数:
        - served_lockers: 已服务的快递柜列表，包含位置和需求量信息
        
        返回:
        - weighted_optimal_distance: 需求量加权的理想距离
        """
        if not served_lockers:
            return 0.0
        
        if len(served_lockers) == 1:
            # 单个快递柜：仓库往返距离，按需求量调整
            locker = served_lockers[0]
            base_distance = self._euclidean_distance(self.depot, locker['location']) * 2
            # 需求量越高，理想距离相对越短（效率越高）
            demand_factor = 1.0 / (1.0 + locker['demand'] * 0.1)  # 需求量越高，因子越小
            return base_distance * demand_factor
        
        # 多个快递柜：使用需求量加权的TSP算法
        weighted_distance = self._estimate_demand_weighted_tsp(served_lockers)
        
        return weighted_distance
    
    def _estimate_demand_weighted_tsp(self, lockers_with_demand: List[Dict]) -> float:
        """
        使用需求量加权的最近邻算法估算TSP距离
        
        算法思路:
        1. 计算所有快递柜之间的需求量加权距离
        2. 优先选择高需求量/距离比的快递柜
        3. 从仓库出发并返回仓库
        
        参数:
        - lockers_with_demand: 包含位置和需求量的快递柜列表
        
        返回:
        - estimated_weighted_distance: 估算的加权距离
        """
        if not lockers_with_demand:
            return 0.0
        
        # 从仓库开始
        current_pos = self.depot
        unvisited = lockers_with_demand.copy()
        total_weighted_distance = 0.0
        
        # 需求量加权的最近邻算法
        while unvisited:
            best_idx = 0
            best_score = float('inf')
            
            for i, locker in enumerate(unvisited):
                # 计算地理距离
                geo_distance = self._euclidean_distance(current_pos, locker['location'])
                
                # 计算需求量加权分数：距离 / 需求量（越小越好）
                # 高需求量的快递柜会有更低的分数，优先被选择
                demand_weight = max(locker['demand'], 1)  # 避免除零
                weighted_score = geo_distance / demand_weight
                
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_idx = i
            
            # 移动到最优快递柜
            selected_locker = unvisited.pop(best_idx)
            move_distance = self._euclidean_distance(current_pos, selected_locker['location'])
            
            # 根据需求量调整实际距离权重
            demand_factor = 1.0 / (1.0 + selected_locker['demand'] * 0.05)
            weighted_move_distance = move_distance * demand_factor
            
            total_weighted_distance += weighted_move_distance
            current_pos = selected_locker['location']
        
        # 返回仓库的距离
        return_distance = self._euclidean_distance(current_pos, self.depot)
        total_weighted_distance += return_distance
        
        return total_weighted_distance
    
    def _calculate_realistic_optimal_distance(self, served_lockers: List[Dict]) -> float:
        """
        计算更现实的理想路径距离
        
        相比需求量加权TSP，这个方法：
        - 减少需求量权重的过度影响
        - 考虑实际约束条件
        - 提供更合理的基准
        
        参数:
        - served_lockers: 已服务的快递柜列表
        
        返回:
        - realistic_optimal_distance: 现实理想距离
        """
        if not served_lockers:
            return 0.0
        
        if len(served_lockers) == 1:
            # 单个快递柜：仓库往返距离，轻微需求量调整
            locker = served_lockers[0]
            base_distance = self._euclidean_distance(self.depot, locker['location']) * 2
            # 需求量调整因子更保守
            demand_factor = 0.9 + 0.1 / (1.0 + locker['demand'] * 0.02)  # 最多10%的调整
            return base_distance * demand_factor
        
        # 多个快递柜：基础TSP + 轻微需求量调整
        positions = [locker['location'] for locker in served_lockers]
        base_tsp_distance = self._estimate_tsp_distance(positions)
        
        # 需求量调整：计算平均需求密度
        total_demand = sum(locker['demand'] for locker in served_lockers)
        avg_demand = total_demand / len(served_lockers)
        
        # 需求密度调整因子（更保守）
        demand_adjustment = 0.95 + 0.05 / (1.0 + avg_demand * 0.01)  # 最多5%的调整
        
        return base_tsp_distance * demand_adjustment
    
    def _calculate_optimal_route_distance(self, served_lockers: List[Dict]) -> float:
        """
        计算服务已完成快递柜的理论最优路径距离（保留原方法用于兼容性）
        
        参数:
        - served_lockers: 已服务的快递柜列表
        
        返回:
        - optimal_distance: 理论最优距离
        """
        if not served_lockers:
            return 0.0
        
        # 获取快递柜位置
        locker_positions = [locker['location'] if isinstance(locker, dict) and 'location' in locker 
                          else locker for locker in served_lockers]
        
        if len(locker_positions) == 1:
            # 单个快递柜：仓库往返距离
            distance_to_locker = self._euclidean_distance(self.depot, locker_positions[0])
            return distance_to_locker * 2  # 往返
        
        # 多个快递柜：使用最近邻算法估算最优路径
        optimal_distance = self._estimate_tsp_distance(locker_positions)
        
        return optimal_distance
    
    def _estimate_tsp_distance(self, positions: List[Tuple[float, float]]) -> float:
        """
        使用最近邻算法估算TSP距离（从仓库出发并返回）
        
        参数:
        - positions: 快递柜位置列表
        
        返回:
        - estimated_distance: 估算的总距离
        """
        if not positions:
            return 0.0
        
        # 从仓库开始
        current_pos = self.depot
        unvisited = positions.copy()
        total_distance = 0.0
        
        # 最近邻算法
        while unvisited:
            # 找到最近的未访问快递柜
            nearest_idx = 0
            min_distance = self._euclidean_distance(current_pos, unvisited[0])
            
            for i, pos in enumerate(unvisited[1:], 1):
                distance = self._euclidean_distance(current_pos, pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
            
            # 移动到最近的快递柜
            total_distance += min_distance
            current_pos = unvisited.pop(nearest_idx)
        
        # 返回仓库
        total_distance += self._euclidean_distance(current_pos, self.depot)
        
        return total_distance
    
    def _calculate_completion_rate(self) -> float:
        """
        计算完成率
        
        返回:
        - completion_rate: 完成率（0-1）
        """
        if not self.lockers_state:
            return 0.0
        
        served_count = sum(1 for locker in self.lockers_state if locker.get('served', False))
        total_count = len(self.lockers_state)
        
        return served_count / total_count
    
    def _calculate_capacity_utilization(self) -> float:
        """
        计算容量利用率
        
        返回:
        - capacity_utilization: 容量利用率（0-1）
        """
        if not self.trucks:
            return 0.0
        
        total_utilization = 0.0
        for truck in self.trucks:
            current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
            capacity = truck.get('capacity', self.truck_capacity)
            utilization = current_load / capacity if capacity > 0 else 0
            total_utilization += utilization
        
        return total_utilization / len(self.trucks)

    def get_locker_location(self, locker_id):
        """根据ID获取快递柜位置"""
        locker = self.get_locker(locker_id)
        return locker['location'] if locker else self.depot

    def step(self, actions):
        """
        新的动态step方法：使用外部实现的动态调度逻辑
        """
        return dynamic_step(self, actions)
    
    def _update_demand_and_handle_uncertainty(self):
        """
        更新需求模型和处理不确定性
        """
        # 收集当前观察到的需求数据
        observed_demands = {}
        for locker in self.lockers_state:
            observed_demands[locker['id']] = {
                'delivery': locker['demand_del'],
                'return': locker['demand_ret']
            }
        
        # 更新需求模型
        self.uncertainty_handler.update_demand_model(self.time_step, observed_demands)
        
        # 检测和处理需求冲击
        shock_response = self.uncertainty_handler.handle_demand_shock(
            self.trucks, self.lockers_state
        )
        
        # 如果检测到容量短缺，记录相关信息（移除频繁的时间步输出）
        if shock_response['shortage_analysis']['shortage_detected']:
            # 容量短缺信息已记录，但不在每个时间步输出
            pass
        
        # 为未服务的快递柜更新需求（模拟需求变化）
        for locker in self.lockers_state:
            if not locker['served']:
                # 获取更新的需求估计
                delivery_estimate = self.uncertainty_handler.get_robust_demand_estimate(
                    locker['id'], 'delivery', self.time_step
                )
                return_estimate = self.uncertainty_handler.get_robust_demand_estimate(
                    locker['id'], 'return', self.time_step
                )
                
                # 更新需求（添加小幅随机变化）
                demand_change_factor = 0.05  # 5%的变化幅度
                delivery_change = (delivery_estimate['actual'] - locker['demand_del']) * demand_change_factor
                return_change = (return_estimate['actual'] - locker['demand_ret']) * demand_change_factor
                
                locker['demand_del'] = max(0, locker['demand_del'] + delivery_change)
                locker['demand_ret'] = max(0, locker['demand_ret'] + return_change)
                
                # 更新不确定性信息
                locker['uncertainty_del'] = delivery_estimate['uncertainty']
                locker['uncertainty_ret'] = return_estimate['uncertainty']

    def optimize_time_windows(self):
        """
        优化时间窗参数，基于历史性能数据调整时间窗设置
        """
        # 收集性能数据
        performance_data = {}
        for locker_id in range(self.num_lockers):
            locker = self.get_locker(locker_id)
            performance_data[locker_id] = {
                'served': locker['served'],
                'demand_del': locker['demand_del'],
                'demand_ret': locker['demand_ret'],
                'service_time': self.time_step if locker['served'] else None
            }
        
        # 使用时间窗优化器进行优化
        self.time_window_optimizer.optimize_time_windows(performance_data)
        
        # 时间窗优化完成（移除频繁的时间步输出）
    
    def get_time_window_statistics(self):
        """
        获取时间窗约束的统计信息
        """
        stats = {
            'total_violations': 0,
            'early_violations': 0,
            'late_violations': 0,
            'total_penalty': 0.0,
            'average_penalty': 0.0
        }
        
        try:
            violations = self.soft_time_window_manager.get_violation_statistics()
            for violation in violations:
                stats['total_violations'] += 1
                # 如果violation是字符串，跳过penalty计算
                if hasattr(violation, 'penalty'):
                    stats['total_penalty'] += violation.penalty
                    
                    if hasattr(violation, 'violation_type') and hasattr(violation.violation_type, 'name'):
                        if violation.violation_type.name == 'EARLY':
                            stats['early_violations'] += 1
                        elif violation.violation_type.name == 'LATE':
                            stats['late_violations'] += 1
        except Exception as e:
            # 如果获取统计信息失败，返回默认值
            pass
        
        if stats['total_violations'] > 0:
            stats['average_penalty'] = stats['total_penalty'] / stats['total_violations']
        
        return stats

    def _check_replenishment_need(self, truck: Dict, truck_id: int) -> ReplenishmentDecision:
        """
        检查卡车是否需要补货
        
        Args:
            truck: 卡车状态字典
            truck_id: 卡车ID
            
        Returns:
            ReplenishmentDecision: 补货决策结果
        """
        # 转换卡车状态为补货模块格式
        truck_state = TruckState(
            truck_id=truck_id,
            current_location=truck['current_location'],
            current_delivery_load=truck['current_delivery_load'],
            current_return_load=truck['current_return_load'],
            remaining_capacity=truck['remaining_space'],
            total_distance=truck['total_distance'],
            visited_stops=truck['visited_stops'].copy(),
            returned=truck['current_location'] == 0  # 根据位置判断是否在仓库
        )
        
        # 获取剩余未服务的快递柜
        remaining_lockers = []
        for locker in self.lockers_state:
            if not locker['served']:
                # 获取时间窗信息
                time_window = self.soft_time_window_manager.get_time_window(locker['id'])
                
                locker_demand = LockerDemand(
                    locker_id=locker['id'],
                    location=locker['location'],
                    delivery_demand=locker['demand_del'],
                    return_demand=locker['demand_ret'],
                    served=locker['served'],
                    priority=self._calculate_locker_priority(locker),
                    time_window_start=time_window.preferred_start if time_window else 0,
                    time_window_end=time_window.preferred_end if time_window else 100
                )
                remaining_lockers.append(locker_demand)
        
        # 使用补货优化器进行决策
        return self.replenishment_optimizer.should_replenish(
            truck_state, remaining_lockers, self.time_step
        )
    
    def _calculate_locker_priority(self, locker: Dict) -> float:
        """
        计算快递柜优先级
        
        Args:
            locker: 快递柜状态字典
            
        Returns:
            float: 优先级分数 (0-1)
        """
        # 基于需求量计算基础优先级
        total_demand = locker['demand_del'] + locker['demand_ret']
        demand_priority = min(1.0, total_demand / 30.0)  # 假设最大需求为30
        
        # 基于距离计算优先级（距离越近优先级越高）
        locker_location = locker['location']
        distance_to_depot = self._euclidean_distance(locker_location, self.depot)
        distance_priority = 1.0 / (1.0 + distance_to_depot / 50.0)
        
        # 基于不确定性计算优先级（不确定性越高优先级越高）
        uncertainty_del = locker.get('uncertainty_del', 0.0)
        uncertainty_ret = locker.get('uncertainty_ret', 0.0)
        uncertainty_priority = (uncertainty_del + uncertainty_ret) / 2.0
        
        # 综合优先级
        return (demand_priority * 0.5 + distance_priority * 0.3 + uncertainty_priority * 0.2)
    
    def _execute_replenishment(self, truck: Dict, decision: ReplenishmentDecision, truck_id: int):
        """
        执行补货决策
        
        Args:
            truck: 卡车状态字典
            decision: 补货决策
            truck_id: 卡车ID
        """
        if self.verbose:
            print(f"卡车 {truck_id} 执行补货决策:")
            print(f"  触发原因: {decision.trigger_reason.value}")
            print(f"  紧急程度: {decision.urgency_level:.2f}")
            print(f"  预期收益: {decision.expected_benefit:.2f}")
            print(f"  风险评估: {decision.risk_assessment:.2f}")
            print(f"  置信度: {decision.confidence:.2f}")
        
        # 记录补货前状态
        old_location_id = truck['current_location']
        old_location = self.depot if old_location_id == 0 else self.get_locker_location(old_location_id)
        
        # 计算返回仓库的距离
        depot_distance = self._euclidean_distance(old_location, self.depot)
        truck['total_distance'] += depot_distance
        self.total_truck_distance += depot_distance
        
        # 更新卡车状态 - 返回仓库补货
        truck['current_location'] = 0
        truck['current_delivery_load'] = self.initial_delivery_load  # 重新装载配送货物
        truck['current_return_load'] = 0  # 卸载取件货物
        truck['remaining_space'] = self.truck_capacity - truck['current_delivery_load']
        
        # 记录补货事件
        self._record_replenishment_event(truck_id, decision, depot_distance)
        
        if self.verbose:
            print(f"  补货完成，行驶距离: {depot_distance:.2f}")
            print(f"  新的配送载量: {truck['current_delivery_load']}")
            print(f"  剩余容量: {truck['remaining_space']}")
    
    def _record_replenishment_event(self, truck_id: int, decision: ReplenishmentDecision, distance: float):
        """
        记录补货事件用于性能分析
        
        Args:
            truck_id: 卡车ID
            decision: 补货决策
            distance: 补货行驶距离
        """
        # 这里可以记录补货事件的详细信息，用于后续性能分析
        # 暂时只在verbose模式下输出
        pass
    
    def get_replenishment_statistics(self) -> Dict[str, Any]:
        """
        获取补货策略统计信息
        
        Returns:
            Dict[str, Any]: 补货统计信息
        """
        return self.replenishment_optimizer.get_strategy_statistics()
    
    def get_truck_capacity_status(self) -> Dict[str, Any]:
        """
        获取详细的卡车容量状态信息
        
        Returns:
            Dict: 包含所有卡车容量状态的详细信息
        """
        truck_status = []
        total_delivery_load = 0
        total_return_load = 0
        total_remaining_space = 0
        
        for i, truck in enumerate(self.trucks):
            delivery_load = truck['current_delivery_load']
            return_load = truck['current_return_load']
            remaining_space = truck['remaining_space']
            
            # 计算利用率
            used_capacity = delivery_load + return_load
            utilization_rate = used_capacity / self.truck_capacity if self.truck_capacity > 0 else 0
            
            # 计算容量状态
            capacity_status = "正常"
            if utilization_rate > 0.9:
                capacity_status = "接近满载"
            elif utilization_rate > 0.7:
                capacity_status = "高负载"
            elif utilization_rate < 0.3:
                capacity_status = "低负载"
            
            truck_info = {
                'truck_id': i,
                'current_location': truck['current_location'],
                'delivery_load': delivery_load,
                'return_load': return_load,
                'total_load': used_capacity,
                'remaining_space': remaining_space,
                'capacity': self.truck_capacity,
                'utilization_rate': utilization_rate,
                'capacity_status': capacity_status,
                'returned': truck['returned'],
                'visited_stops': len(truck['visited_stops']),
                'total_distance': truck['total_distance']
            }
            truck_status.append(truck_info)
            
            # 累计统计
            total_delivery_load += delivery_load
            total_return_load += return_load
            total_remaining_space += remaining_space
        
        # 计算车队级别统计
        fleet_capacity = self.num_trucks * self.truck_capacity
        fleet_used = total_delivery_load + total_return_load
        fleet_utilization = fleet_used / fleet_capacity if fleet_capacity > 0 else 0
        
        return {
            'individual_trucks': truck_status,
            'fleet_summary': {
                'total_trucks': self.num_trucks,
                'total_capacity': fleet_capacity,
                'total_delivery_load': total_delivery_load,
                'total_return_load': total_return_load,
                'total_used_capacity': fleet_used,
                'total_remaining_space': total_remaining_space,
                'fleet_utilization_rate': fleet_utilization,
                'average_utilization': np.mean([truck['utilization_rate'] for truck in truck_status]),
                'max_utilization': max([truck['utilization_rate'] for truck in truck_status]) if truck_status else 0,
                'min_utilization': min([truck['utilization_rate'] for truck in truck_status]) if truck_status else 0
            }
        }
    
    def validate_truck_capacity_consistency(self) -> Dict[str, Any]:
        """
        验证卡车容量状态的一致性
        
        Returns:
            Dict: 容量一致性检查结果
        """
        issues = []
        warnings = []
        
        for i, truck in enumerate(self.trucks):
            delivery_load = truck['current_delivery_load']
            return_load = truck['current_return_load']
            remaining_space = truck['remaining_space']
            
            # 检查容量计算是否正确
            calculated_remaining = self.truck_capacity - delivery_load - return_load
            if abs(calculated_remaining - remaining_space) > 0.01:  # 允许小的浮点误差
                issues.append(f"卡车{i}容量计算不一致: 计算值={calculated_remaining}, 记录值={remaining_space}")
            
            # 检查是否超载
            total_load = delivery_load + return_load
            if total_load > self.truck_capacity:
                issues.append(f"卡车{i}超载: 总负载={total_load}, 容量={self.truck_capacity}")
            
            # 检查负值
            if delivery_load < 0:
                issues.append(f"卡车{i}送货负载为负值: {delivery_load}")
            if return_load < 0:
                issues.append(f"卡车{i}退货负载为负值: {return_load}")
            if remaining_space < 0:
                issues.append(f"卡车{i}剩余空间为负值: {remaining_space}")
            
            # 检查警告情况
            if total_load == 0 and not truck['returned']:
                warnings.append(f"卡车{i}负载为0但未返回")
            
            utilization = total_load / self.truck_capacity
            if utilization > 0.95:
                warnings.append(f"卡车{i}利用率过高: {utilization:.2%}")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_issues': len(issues),
            'total_warnings': len(warnings)
        }
    
    def _calculate_regional_density(self, center_point: Tuple[float, float], radius: float = 20.0) -> Dict[str, float]:
        """
        计算指定区域的快递柜密集度和需求聚合信息
        
        Args:
            center_point: 中心点坐标 (x, y)
            radius: 覆盖半径
            
        Returns:
            Dict[str, float]: 区域特征信息
        """
        lockers_in_range = []
        total_pickup_demand = 0.0
        total_return_demand = 0.0
        
        for i, locker in enumerate(self.lockers_state):
            locker_pos = locker['location']
            distance = self._euclidean_distance(center_point, locker_pos)
            
            if distance <= radius:
                lockers_in_range.append(i)
                total_pickup_demand += locker.get('demand_del', 0)
                total_return_demand += locker.get('demand_ret', 0)
        
        # 计算密集度指标
        area = math.pi * radius * radius
        density = len(lockers_in_range) / area if area > 0 else 0
        
        # 计算需求密度
        demand_density = (total_pickup_demand + total_return_demand) / area if area > 0 else 0
        
        # 计算服务效率潜力（未服务的需求比例）
        unserved_lockers = sum(1 for i in lockers_in_range 
                              if not self.lockers_state[i].get('served', False))
        
        service_potential = unserved_lockers / max(len(lockers_in_range), 1)
        
        return {
            'locker_count': len(lockers_in_range),
            'locker_density': density,
            'total_demand': total_pickup_demand + total_return_demand,
            'demand_density': demand_density,
            'service_potential': service_potential,
            'coverage_efficiency': len(lockers_in_range) / max(self.num_lockers, 1)
        }
    
    def _get_global_features(self) -> Dict[str, Any]:
        """
        计算全局特征，用于策略网络的全局信息感知
        
        Returns:
            Dict[str, Any]: 全局特征信息
        """
        # 计算所有快递柜的中心点
        if not self.lockers_state:
            return {}
            
        center_x = sum(locker['location'][0] for locker in self.lockers_state) / len(self.lockers_state)
        center_y = sum(locker['location'][1] for locker in self.lockers_state) / len(self.lockers_state)
        
        # 分析不同半径下的区域特征
        regional_features = {}
        for radius in [15, 25, 35]:  # 不同覆盖半径
            features = self._calculate_regional_density((center_x, center_y), radius)
            regional_features[f'radius_{radius}'] = features
        
        # 计算快递柜分布的离散程度
        distances_from_center = [
            self._euclidean_distance((center_x, center_y), locker['location'])
            for locker in self.lockers_state
        ]
        spread = np.std(distances_from_center) if distances_from_center else 0
        
        # 计算需求热点区域
        demand_hotspots = self._identify_demand_hotspots()
        
        # 计算卡车当前位置的战略价值
        truck_strategic_values = []
        for truck in self.trucks:
            truck_pos = truck['position']
            strategic_value = self._calculate_strategic_value(truck_pos)
            truck_strategic_values.append(strategic_value)
        
        return {
            'center_point': (center_x, center_y),
            'distribution_spread': spread,
            'regional_features': regional_features,
            'demand_hotspots': demand_hotspots,
            'truck_strategic_values': truck_strategic_values,
            'total_unserved_demand': self._calculate_total_unserved_demand()
        }
    
    def _identify_demand_hotspots(self, grid_size: int = 5) -> List[Dict[str, Any]]:
        """
        识别需求热点区域
        
        Args:
            grid_size: 网格划分大小
            
        Returns:
            List[Dict[str, Any]]: 热点区域信息
        """
        # 计算边界
        min_x = min(locker['location'][0] for locker in self.lockers_state)
        max_x = max(locker['location'][0] for locker in self.lockers_state)
        min_y = min(locker['location'][1] for locker in self.lockers_state)
        max_y = max(locker['location'][1] for locker in self.lockers_state)
        
        # 创建网格
        x_step = (max_x - min_x) / grid_size
        y_step = (max_y - min_y) / grid_size
        
        hotspots = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_x = min_x + (i + 0.5) * x_step
                grid_y = min_y + (j + 0.5) * y_step
                
                # 计算该网格的需求密度
                grid_features = self._calculate_regional_density((grid_x, grid_y), radius=15.0)
                
                if grid_features['locker_count'] > 0:
                    hotspots.append({
                        'center': (grid_x, grid_y),
                        'locker_count': grid_features['locker_count'],
                        'demand_density': grid_features['demand_density'],
                        'service_potential': grid_features['service_potential']
                    })
        
        # 按需求密度排序
        hotspots.sort(key=lambda x: x['demand_density'], reverse=True)
        return hotspots[:3]  # 返回前3个热点
    
    def _calculate_strategic_value(self, position: Tuple[float, float]) -> float:
        """
        计算位置的战略价值
        
        Args:
            position: 位置坐标
            
        Returns:
            float: 战略价值分数
        """
        # 计算该位置的区域特征
        regional_features = self._calculate_regional_density(position, radius=25.0)
        
        # 综合评分：密集度 + 需求密度 + 服务潜力
        strategic_value = (
            regional_features['locker_density'] * 0.3 +
            regional_features['demand_density'] * 0.4 +
            regional_features['service_potential'] * 0.3
        )
        
        return strategic_value
    
    def _calculate_total_unserved_demand(self) -> float:
        """
        计算总的未服务需求
        
        Returns:
            float: 未服务需求总量
        """
        total_unserved = 0.0
        for locker in self.lockers_state:
            if not locker.get('pickup_served', False):
                total_unserved += locker.get('pickup_demand', 0)
            if not locker.get('return_served', False):
                total_unserved += locker.get('return_demand', 0)
        return total_unserved
    
    def _calculate_truck_density(self, truck_positions: List[Tuple[float, float]]) -> float:
        """
        计算卡车分布密度
        
        Args:
            truck_positions: 卡车位置列表
            
        Returns:
            float: 密度值
        """
        if len(truck_positions) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(truck_positions)):
            for j in range(i + 1, len(truck_positions)):
                total_distance += self._euclidean_distance(truck_positions[i], truck_positions[j])
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        # 密度与平均距离成反比
        return 1.0 / (1.0 + avg_distance / 100.0)
    
    def _detect_coordination_conflicts(self) -> List[Dict[str, Any]]:
        """
        检测卡车间的协调冲突
        
        Returns:
            List[Dict]: 冲突列表
        """
        conflicts = []
        
        # 检测目标冲突：多个卡车前往同一个快递柜
        target_conflicts = {}
        for truck_id, truck in enumerate(self.trucks):
            if truck['current_location'] != 0:  # 0表示仓库
                # 简化：假设卡车正在前往最近的有需求的快递柜
                nearest_locker = self._find_nearest_locker_with_demand(truck['position'])
                if nearest_locker is not None:
                    if nearest_locker not in target_conflicts:
                        target_conflicts[nearest_locker] = []
                    target_conflicts[nearest_locker].append(truck_id)
        
        for locker_id, truck_ids in target_conflicts.items():
            if len(truck_ids) > 1:
                conflicts.append({
                    'type': 'target_conflict',
                    'locker_id': locker_id,
                    'truck_ids': truck_ids,
                    'severity': len(truck_ids) / self.num_trucks
                })
        
        # 检测路径冲突：卡车过于接近
        for i in range(len(self.trucks)):
            for j in range(i + 1, len(self.trucks)):
                truck1, truck2 = self.trucks[i], self.trucks[j]
                if truck1['current_location'] != 0 and truck2['current_location'] != 0:  # 0表示仓库
                    distance = self._euclidean_distance(truck1['position'], truck2['position'])
                    if distance < 10.0:  # 阈值：10单位距离
                        conflicts.append({
                            'type': 'proximity_conflict',
                            'truck_ids': [i, j],
                            'distance': distance,
                            'severity': max(0, (10.0 - distance) / 10.0)
                        })
        
        return conflicts
    
    def _calculate_global_load_distribution(self) -> Dict[str, float]:
        """
        计算全局负载分布
        
        Returns:
            Dict: 负载分布统计
        """
        loads = []
        for truck in self.trucks:
            total_load = truck.get('delivery_items', 0) + truck.get('return_items', 0)
            load_ratio = total_load / self.truck_capacity if self.truck_capacity > 0 else 0
            loads.append(load_ratio)
        
        if not loads:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'balance_score': 1.0}
        
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        min_load = np.min(loads)
        max_load = np.max(loads)
        
        # 平衡分数：标准差越小，平衡性越好
        balance_score = 1.0 / (1.0 + std_load)
        
        return {
            'mean': mean_load,
            'std': std_load,
            'min': min_load,
            'max': max_load,
            'balance_score': balance_score
        }
    
    def _predict_future_demand_trend(self) -> Dict[str, float]:
        """
        预测未来需求趋势
        
        Returns:
            Dict: 需求趋势预测
        """
        # 简化的趋势预测：基于当前需求分布和时间
        current_delivery_demand = sum(locker.get('delivery_demand', 0) for locker in self.lockers_state)
        current_return_demand = sum(locker.get('return_demand', 0) for locker in self.lockers_state)
        
        # 基于时间步的趋势预测
        time_factor = self.time_step / self.max_timesteps if self.max_timesteps > 0 else 0
        
        # 假设送货需求在前期较高，退货需求在后期较高
        delivery_trend = max(0, 1.0 - time_factor * 1.5)  # 递减趋势
        return_trend = min(1.0, time_factor * 1.2)  # 递增趋势
        
        # 热点转移：基于当前需求分布的变化
        hotspot_shift = abs(current_delivery_demand - current_return_demand) / max(1, current_delivery_demand + current_return_demand)
        
        return {
            'delivery_trend': delivery_trend,
            'return_trend': return_trend,
            'hotspot_shift': hotspot_shift
        }
    
    def _calculate_coordination_efficiency(self) -> float:
        """
        计算协调效率
        
        Returns:
            float: 协调效率分数
        """
        # 基于多个因素计算协调效率
        
        # 1. 负载平衡性
        load_distribution = self._calculate_global_load_distribution()
        load_balance_score = load_distribution['balance_score']
        
        # 2. 冲突程度
        conflicts = self._detect_coordination_conflicts()
        conflict_penalty = sum(conflict['severity'] for conflict in conflicts) / max(1, len(conflicts))
        conflict_score = max(0, 1.0 - conflict_penalty)
        
        # 3. 覆盖效率
        active_trucks = sum(1 for truck in self.trucks if truck['current_location'] != 0)
        coverage_score = active_trucks / self.num_trucks if self.num_trucks > 0 else 0
        
        # 综合评分
        efficiency = 0.4 * load_balance_score + 0.3 * conflict_score + 0.3 * coverage_score
        return min(1.0, max(0.0, efficiency))
    
    def _predict_truck_completion_time(self, truck_id: int) -> float:
        """
        预测卡车完成当前任务的时间
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            float: 预测完成时间
        """
        truck = self.trucks[truck_id]
        
        # 如果卡车已返回仓库，完成时间为0
        if truck['current_location'] == 0:
            return 0.0
        
        # 计算返回仓库的距离
        distance_to_depot = self._euclidean_distance(truck['position'], (0, 0))
        
        # 估算剩余服务时间（基于当前负载）
        current_load = truck.get('delivery_items', 0) + truck.get('return_items', 0)
        service_time = current_load * 2  # 假设每个物品需要2个时间单位
        
        # 估算移动时间（假设速度为1单位/时间步）
        travel_time = distance_to_depot
        
        return service_time + travel_time
    
    def _calculate_path_optimization_potential(self, truck_id: int) -> float:
        """
        计算路径优化潜力
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            float: 优化潜力分数
        """
        truck = self.trucks[truck_id]
        current_pos = truck['position']
        
        # 找到附近的需求点
        nearby_demands = []
        for i, locker in enumerate(self.lockers_state):
            if locker.get('delivery_demand', 0) > 0 or locker.get('return_demand', 0) > 0:
                distance = self._euclidean_distance(current_pos, locker['location'])
                if distance <= 50.0:  # 50单位范围内
                    nearby_demands.append((i, distance, locker.get('delivery_demand', 0) + locker.get('return_demand', 0)))
        
        if not nearby_demands:
            return 0.0
        
        # 计算当前路径效率
        total_demand = sum(demand for _, _, demand in nearby_demands)
        total_distance = sum(distance for _, distance, _ in nearby_demands)
        
        if total_distance == 0:
            return 0.0
        
        current_efficiency = total_demand / total_distance
        
        # 计算理论最优效率（按距离排序）
        nearby_demands.sort(key=lambda x: x[1])  # 按距离排序
        optimal_distance = sum(nearby_demands[i][1] for i in range(min(3, len(nearby_demands))))  # 最近3个
        optimal_demand = sum(nearby_demands[i][2] for i in range(min(3, len(nearby_demands))))
        
        if optimal_distance == 0:
            return 0.0
        
        optimal_efficiency = optimal_demand / optimal_distance
        
        # 优化潜力 = (最优效率 - 当前效率) / 最优效率
        if optimal_efficiency == 0:
            return 0.0
        
        potential = max(0, (optimal_efficiency - current_efficiency) / optimal_efficiency)
        return min(1.0, potential)
    
    def _calculate_coordination_opportunity(self, truck_id: int) -> float:
        """
        计算协调机会评分
        
        Args:
            truck_id: 卡车ID
            
        Returns:
            float: 协调机会分数
        """
        truck = self.trucks[truck_id]
        current_pos = truck['position']
        
        # 计算与其他卡车的协调机会
        coordination_score = 0.0
        active_trucks = 0
        
        for other_id, other_truck in enumerate(self.trucks):
            if other_id != truck_id and other_truck['position'] != 0:
                distance = self._euclidean_distance(current_pos, other_truck['position'])
                
                # 如果距离适中（不太近也不太远），有协调机会
                if 20.0 <= distance <= 80.0:
                    # 检查是否有共同的服务区域
                    common_area_score = self._calculate_common_service_area(truck_id, other_id)
                    
                    # 检查负载互补性
                    load_complementarity = self._calculate_load_complementarity(truck_id, other_id)
                    
                    truck_coordination = 0.5 * common_area_score + 0.5 * load_complementarity
                    coordination_score += truck_coordination
                    active_trucks += 1
        
        if active_trucks == 0:
            return 0.0
        
        return coordination_score / active_trucks
    
    def _find_nearest_locker_with_demand(self, position: Tuple[float, float]) -> Optional[int]:
        """
        找到最近的有需求的快递柜
        
        Args:
            position: 当前位置
            
        Returns:
            Optional[int]: 快递柜ID，如果没有找到则返回None
        """
        min_distance = float('inf')
        nearest_locker = None
        
        for i, locker in enumerate(self.lockers_state):
            if locker.get('delivery_demand', 0) > 0 or locker.get('return_demand', 0) > 0:
                distance = self._euclidean_distance(position, locker['location'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_locker = i
        
        return nearest_locker
    
    def _calculate_common_service_area(self, truck1_id: int, truck2_id: int) -> float:
        """
        计算两个卡车的共同服务区域重叠度
        
        Args:
            truck1_id: 卡车1 ID
            truck2_id: 卡车2 ID
            
        Returns:
            float: 重叠度分数
        """
        truck1_pos = self.trucks[truck1_id]['position']
        truck2_pos = self.trucks[truck2_id]['position']
        
        # 计算两个卡车服务范围的重叠
        service_radius = 30.0  # 假设服务半径为30单位
        distance_between = self._euclidean_distance(truck1_pos, truck2_pos)
        
        if distance_between >= 2 * service_radius:
            return 0.0  # 没有重叠
        
        if distance_between == 0:
            return 1.0  # 完全重叠
        
        # 计算圆形区域重叠比例（简化计算）
        overlap_ratio = max(0, (2 * service_radius - distance_between) / (2 * service_radius))
        return overlap_ratio
    
    def _calculate_load_complementarity(self, truck1_id: int, truck2_id: int) -> float:
        """
        计算两个卡车的负载互补性
        
        Args:
            truck1_id: 卡车1 ID
            truck2_id: 卡车2 ID
            
        Returns:
            float: 互补性分数
        """
        truck1 = self.trucks[truck1_id]
        truck2 = self.trucks[truck2_id]
        
        # 计算负载差异
        load1 = (truck1.get('delivery_items', 0) + truck1.get('return_items', 0)) / self.truck_capacity
        load2 = (truck2.get('delivery_items', 0) + truck2.get('return_items', 0)) / self.truck_capacity
        
        # 互补性：负载差异越大，互补性越强
        load_difference = abs(load1 - load2)
        
        # 计算类型互补性（送货vs退货）
        delivery1 = truck1.get('delivery_items', 0) / max(1, truck1.get('delivery_items', 0) + truck1.get('return_items', 0))
        delivery2 = truck2.get('delivery_items', 0) / max(1, truck2.get('delivery_items', 0) + truck2.get('return_items', 0))
        
        type_complementarity = abs(delivery1 - delivery2)
        
        # 综合互补性分数
        complementarity = 0.6 * load_difference + 0.4 * type_complementarity
        return min(1.0, complementarity)


class RouteAwareValueNetwork(nn.Module):
    """
    路线规划感知的价值网络
    
    专门针对路线规划任务设计的价值网络，能够理解：
    - 路径效率和优化潜力
    - 需求权重和时序决策
    - 长期路线规划价值
    
    Author: Dionysus
    Contact: wechat:gzw1546484791
    """
    
    def __init__(self, state_dim: int):
        super(RouteAwareValueNetwork, self).__init__()
        
        # 输入特征编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 路径效率感知分支
        self.path_efficiency_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 需求权重感知分支
        self.demand_weight_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 时序决策感知分支
        self.temporal_decision_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局路线规划分支
        self.global_route_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 4 + 256, 256),  # 4个分支 + 原始特征
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 路线规划价值预测头（辅助任务）
        self.route_value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态张量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (主价值, 路线价值)
        """
        # 编码输入状态
        encoded_state = self.state_encoder(state)
        
        # 多分支特征提取
        path_features = self.path_efficiency_branch(encoded_state)
        demand_features = self.demand_weight_branch(encoded_state)
        temporal_features = self.temporal_decision_branch(encoded_state)
        global_features = self.global_route_branch(encoded_state)
        
        # 特征融合
        fused_features = torch.cat([
            encoded_state,
            path_features,
            demand_features,
            temporal_features,
            global_features
        ], dim=-1)
        
        # 融合处理
        processed_features = self.feature_fusion(fused_features)
        
        # 价值预测
        main_value = self.value_head(processed_features)
        route_value = self.route_value_head(processed_features)
        
        return main_value, route_value


# 增强的策略网络
class TruckPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TruckPolicyNetwork, self).__init__()
        
        # 增强的状态编码器 - 6层深度网络，提升特征表达能力
        self.state_encoder = nn.Sequential(
            # 第一层：输入处理 - 增加宽度
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # 第二层：深层特征提取 - 新增层
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # 第三层：特征融合
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 需求感知特征提取器（新增）
        self.demand_encoder = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.ReLU()
        )
        
        # 地图全局信息编码器（新增）
        self.global_encoder = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 64),
            nn.ReLU()
        )
        
        # 多头注意力机制 - 分别处理需求和地图信息
        self.demand_attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4,
            dropout=0.05,
            batch_first=True
        )
        
        self.global_attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 原有的自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4,  # 从8减少到4
            dropout=0.15,  # 增加Dropout率从0.05到0.15
            batch_first=True
        )
        
        # 特征融合层（新增）
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),  # 融合原始特征、需求特征、地图特征
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 增强停靠点选择头 - 3层深度网络，增强正则化
        self.stop_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout率
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.15), # 增加dropout率
            nn.Linear(64, action_dim["select_stop"])
        )

        # 增强服务区域选择头 - 3层深度网络，增强正则化
        self.service_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout率
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.15), # 增加dropout率
            nn.Linear(64, action_dim["service_area"])
        )
        
        # 新增：区域优先级计算模块
        self.regional_priority_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出单一优先级分数
        )
        
        # 新增：密集度感知层
        self.density_aware_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Tanh()  # 使用Tanh激活函数，输出范围[-1,1]
        )
        
        # 新增：覆盖效率评估层
        self.coverage_efficiency_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Sigmoid()  # 使用Sigmoid激活函数，输出范围[0,1]
        )
        
        # 新增：路线规划感知层
        self.route_planning_layer = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 新增：路径优化潜力评估层
        self.path_optimization_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 24),
            nn.Tanh()  # 输出范围[-1,1]，表示优化潜力
        )
        
        # 新增：多卡车协调感知层
        self.coordination_awareness_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 20),
            nn.Sigmoid()  # 输出范围[0,1]，表示协调机会
        )
        
        # 新增：历史路径学习层
        self.path_history_layer = nn.Sequential(
            nn.Linear(128, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 16),
            nn.ReLU()
        )
        
        # 新增：未来需求预测感知层
        self.future_demand_layer = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 12),
            nn.Softmax(dim=-1)  # 输出概率分布
        )
        
        # 新增：路线规划特征融合层
        self.route_feature_fusion = nn.Sequential(
            nn.Linear(32 + 24 + 20 + 16 + 12, 64),  # 融合所有路线规划特征
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, state, stop_mask=None, service_mask=None):
        # 状态编码
        x = self.state_encoder(state)
        batch_size = x.size(0)
        
        # 1. 需求感知特征提取
        demand_features = self.demand_encoder(x)  # [batch_size, 64]
        demand_seq = demand_features.unsqueeze(1)  # [batch_size, 1, 64]
        
        # 需求感知注意力机制
        demand_attn_output, demand_attn_weights = self.demand_attention(
            demand_seq, demand_seq, demand_seq
        )
        demand_enhanced = demand_attn_output.squeeze(1)  # [batch_size, 64]
        
        # 2. 地图全局信息提取
        global_features = self.global_encoder(x)  # [batch_size, 64]
        global_seq = global_features.unsqueeze(1)  # [batch_size, 1, 64]
        
        # 地图全局注意力机制
        global_attn_output, global_attn_weights = self.global_attention(
            global_seq, global_seq, global_seq
        )
        global_enhanced = global_attn_output.squeeze(1)  # [batch_size, 64]
        
        # 3. 原有的自注意力机制
        x_seq = x.unsqueeze(1)  # [batch_size, 1, 128]
        self_attn_output, _ = self.self_attention(x_seq, x_seq, x_seq)
        self_enhanced = self_attn_output.squeeze(1)  # [batch_size, 128]
        
        # 残差连接
        self_final = x + self_enhanced
        
        # 4. 特征融合
        # 将需求特征、地图特征和自注意力特征融合
        fused_features = torch.cat([
            self_final,      # [batch_size, 128] - 原始+自注意力特征
            demand_enhanced, # [batch_size, 64]  - 需求感知特征
            global_enhanced  # [batch_size, 64]  - 地图全局特征
        ], dim=1)  # [batch_size, 256]
        
        # 通过融合层处理
        x_final = self.feature_fusion(fused_features)  # [batch_size, 128]
        
        # 添加层归一化提升训练稳定性
        x_final = F.layer_norm(x_final, x_final.shape[1:])

        # 停靠点选择 - 基础logits
        stop_logits_base = self.stop_head(x_final)
        
        # 计算区域优先级分数
        regional_priority = self.regional_priority_head(x_final)  # [batch_size, 1]
        
        # 计算密集度感知特征
        density_features = self.density_aware_layer(x_final)  # [batch_size, 32]
        
        # 计算覆盖效率特征
        coverage_features = self.coverage_efficiency_layer(x_final)  # [batch_size, 16]
        
        # 新增：路线规划感知特征提取
        route_planning_features = self.route_planning_layer(x_final)  # [batch_size, 32]
        path_optimization_features = self.path_optimization_layer(x_final)  # [batch_size, 24]
        coordination_features = self.coordination_awareness_layer(x_final)  # [batch_size, 20]
        path_history_features = self.path_history_layer(x_final)  # [batch_size, 16]
        future_demand_features = self.future_demand_layer(x_final)  # [batch_size, 12]
        
        # 融合所有路线规划特征
        route_combined_features = torch.cat([
            route_planning_features,    # [batch_size, 32]
            path_optimization_features, # [batch_size, 24]
            coordination_features,      # [batch_size, 20]
            path_history_features,      # [batch_size, 16]
            future_demand_features      # [batch_size, 12]
        ], dim=1)  # [batch_size, 104]
        
        # 通过路线特征融合层处理
        route_enhanced_features = self.route_feature_fusion(route_combined_features)  # [batch_size, 32]
        
        # 增强停靠点选择：结合区域优先级、密集度信息和路线规划特征
        # 将区域优先级广播到所有停靠点选择
        priority_boost = regional_priority.expand(-1, stop_logits_base.size(1))  # [batch_size, num_stops]
        
        # 密集度加权：将密集度特征转换为停靠点权重
        density_weight = torch.mean(density_features, dim=1, keepdim=True)  # [batch_size, 1]
        density_boost = density_weight.expand(-1, stop_logits_base.size(1))  # [batch_size, num_stops]
        
        # 覆盖效率加权：将覆盖效率特征转换为停靠点权重
        coverage_weight = torch.mean(coverage_features, dim=1, keepdim=True)  # [batch_size, 1]
        coverage_boost = coverage_weight.expand(-1, stop_logits_base.size(1))  # [batch_size, num_stops]
        
        # 路线规划加权：将路线规划特征转换为停靠点权重
        route_weight = torch.mean(route_enhanced_features, dim=1, keepdim=True)  # [batch_size, 1]
        route_boost = route_weight.expand(-1, stop_logits_base.size(1))  # [batch_size, num_stops]
        
        # 综合停靠点选择logits：基础logits + 各种增强特征加权
        stop_logits = (
            stop_logits_base + 
            0.25 * priority_boost +     # 25% 区域优先级权重
            0.30 * density_boost +      # 30% 密集度权重  
            0.25 * coverage_boost +     # 25% 覆盖效率权重
            0.20 * route_boost          # 20% 路线规划权重
        )

        # 服务区域选择 - 基础logits
        service_logits_base = self.service_head(x_final)
        
        # 增强服务区域选择：结合路线规划特征
        # 将路线规划特征转换为服务区域权重
        route_service_weight = torch.mean(route_enhanced_features, dim=1, keepdim=True)  # [batch_size, 1]
        route_service_boost = route_service_weight.expand(-1, service_logits_base.size(1))  # [batch_size, num_service_areas]
        
        # 路径优化潜力加权
        path_opt_weight = torch.mean(path_optimization_features, dim=1, keepdim=True)  # [batch_size, 1]
        path_opt_boost = path_opt_weight.expand(-1, service_logits_base.size(1))  # [batch_size, num_service_areas]
        
        # 协调感知加权
        coord_weight = torch.mean(coordination_features, dim=1, keepdim=True)  # [batch_size, 1]
        coord_boost = coord_weight.expand(-1, service_logits_base.size(1))  # [batch_size, num_service_areas]
        
        # 综合服务区域选择logits：基础logits + 路线规划增强
        service_logits = (
            service_logits_base +
            0.35 * route_service_boost +  # 35% 路线规划权重
            0.35 * path_opt_boost +       # 35% 路径优化权重
            0.30 * coord_boost            # 30% 协调感知权重
        )

        return stop_logits, service_logits


# 多智能体PPO算法
class MAPPO:
    def __init__(self, num_trucks, state_dim, action_dim, lr=None):
        # 设备检测和配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 MAPPO使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 如果lr为None，使用config中的学习率
        if lr is None:
            lr = config.LEARNING_RATE
        
        self.policy_net = TruckPolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy_net = TruckPolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        # 路线规划感知的价值网络 - 专门针对路线优化任务设计
        self.value_net = RouteAwareValueNetwork(state_dim).to(self.device)

        # 保存基础学习率用于预热机制 - 防止过快收敛
        self.policy_base_lr = lr * 0.3  # 策略网络学习率大幅降低
        self.value_base_lr = lr * 0.2   # 价值网络学习率大幅降低
        
        # 深度网络优化器：针对更深网络调整学习率和正则化
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.policy_base_lr,  # 深度网络使用更小的学习率
            weight_decay=5e-4,  # 大幅增强L2正则化防止过拟合
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=self.value_base_lr,  # 价值网络学习率进一步降低
            weight_decay=3e-4,  # 价值网络增强权重衰减
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        self.num_trucks = num_trucks
        
        # 深度网络训练参数 - 修复关键超参数
        self.max_grad_norm = 0.5  # 适中的梯度裁剪
        self.clip_ratio = 0.2     # 标准PPO裁剪比率，确保足够的策略更新幅度
        
        # 自适应探索机制 - 防止过快收敛和局部最优
        self.initial_entropy_coef = 0.15  # 提高初始熵系数，增强早期探索
        self.min_entropy_coef = 0.02      # 提高最小熵系数，保持长期探索
        self.entropy_decay_rate = 0.9995  # 大幅减缓熵系数衰减率
        self.entropy_coef = self.initial_entropy_coef
        
        self.value_loss_coef = 0.5  # 标准价值损失系数
        
        # 探索增强参数
        self.exploration_bonus_coef = 0.08  # 增加探索奖励系数
        self.action_diversity_threshold = 0.7  # 降低动作多样性阈值，更容易触发探索奖励
        
        # 防止过拟合的训练设置
        self.batch_size = 256     # 减少批处理大小，增加随机性
        self.mini_batch_size = 64 # 减少小批次大小，增加梯度噪声
        self.update_epochs = 3    # 减少更新轮数，防止过度拟合
        
        # 并行处理优化
        self.num_workers = 4      # 数据加载并行工作进程数
        
        # 训练性能跟踪
        self.best_performance = 0.0  # 最佳性能记录
        self.pin_memory = True    # 启用内存锁定，加速GPU传输
        
        # GPU内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # 优化卷积操作
            torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提升性能
        
        # 深度网络学习率调度器 - 添加预热机制
        # 策略网络：预热 + 阶梯衰减
        self.policy_warmup_steps = 500
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=1500, gamma=0.9  # 更频繁的衰减
        )
        
        # 价值网络：预热 + 阶梯衰减
        self.value_warmup_steps = 300
        self.value_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=1200, gamma=0.95
        )
        
        # 预热计数器
        self.training_step = 0
        
        # 探索机制：访问状态计数器
        self.visited_state_count = {}
        
        # 早停机制
        self.best_performance = float('-inf')
        self.patience = 50
        self.patience_counter = 0
        self.early_stop = False
        self.performance_history = []
        print("早停机制已重置")

    def update_hyperparameters(self, hyperparams: Dict[str, float]):
        """
        更新超参数
        
        Args:
            hyperparams: 超参数字典
        """
        if 'learning_rate' in hyperparams:
            # 更新策略网络学习率
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = hyperparams['learning_rate']
            # 更新价值网络学习率（使用较小的学习率）
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] = hyperparams['learning_rate'] * 0.5
        
        if 'clip_ratio' in hyperparams:
            self.clip_ratio = hyperparams['clip_ratio']
        
        if 'entropy_coef' in hyperparams:
            self.entropy_coef = hyperparams['entropy_coef']
        
        if 'value_loss_coef' in hyperparams:
            self.value_loss_coef = hyperparams['value_loss_coef']
        
        if 'max_grad_norm' in hyperparams:
            self.max_grad_norm = hyperparams['max_grad_norm']

    def optimize_action_space(self, env, truck_id: int) -> Dict[str, Any]:
        """
        优化卡车动作空间设计，提供更多样化的停靠点选择策略
        
        Args:
            env: 环境实例
            truck_id: 卡车ID
            
        Returns:
            优化后的动作空间信息
        """
        truck = env.trucks[truck_id]
        truck_pos = truck['position']
        
        # 1. 基于距离的分层选择
        distance_tiers = {
            'immediate': [],  # 0-10km
            'nearby': [],     # 10-20km  
            'distant': []     # 20km+
        }
        
        for locker_id, locker in enumerate(env.lockers_state):
            if locker['demand_del'] + locker['demand_ret'] > 0:
                distance = env._euclidean_distance(truck_pos, locker['location'])
                if distance <= 10:
                    distance_tiers['immediate'].append(locker_id)
                elif distance <= 20:
                    distance_tiers['nearby'].append(locker_id)
                else:
                    distance_tiers['distant'].append(locker_id)
        
        # 2. 基于需求密度的聚类选择
        demand_clusters = self._identify_demand_clusters(env, truck_pos)
        
        # 3. 基于协调机会的选择
        coordination_opportunities = self._find_coordination_opportunities(env, truck_id)
        
        # 4. 基于探索价值的选择
        exploration_targets = self._identify_exploration_targets(env, truck_pos)
        
        # 5. 动态权重分配
        strategy_weights = self._calculate_strategy_weights()
        
        return {
            'distance_tiers': distance_tiers,
            'demand_clusters': demand_clusters,
            'coordination_opportunities': coordination_opportunities,
            'exploration_targets': exploration_targets,
            'strategy_weights': strategy_weights,
            'recommended_actions': self._generate_action_recommendations(
                distance_tiers, demand_clusters, coordination_opportunities, 
                exploration_targets, strategy_weights
            )
        }
    
    def _identify_demand_clusters(self, env, truck_pos: Tuple[float, float]) -> List[Dict]:
        """识别需求聚集区域"""
        clusters = []
        grid_size = 3
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 定义网格区域
                x_min = i * (100 / grid_size)
                x_max = (i + 1) * (100 / grid_size)
                y_min = j * (100 / grid_size)
                y_max = (j + 1) * (100 / grid_size)
                
                # 计算区域内的需求密度
                total_demand = 0
                locker_count = 0
                lockers_in_cluster = []
                
                for locker_id, locker in enumerate(env.lockers_state):
                    locker_x, locker_y = locker['location']
                    if (x_min <= locker_x < x_max and 
                        y_min <= locker_y < y_max):
                        demand = locker['demand_del'] + locker['demand_ret']
                        if demand > 0:
                            total_demand += demand
                            locker_count += 1
                            lockers_in_cluster.append(locker_id)
                
                if locker_count > 0:
                    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                    distance_to_truck = env._euclidean_distance(truck_pos, center)
                    
                    clusters.append({
                        'center': center,
                        'total_demand': total_demand,
                        'locker_count': locker_count,
                        'density': total_demand / locker_count,
                        'distance': distance_to_truck,
                        'lockers': lockers_in_cluster,
                        'priority': total_demand / (1 + distance_to_truck)
                    })
        
        # 按优先级排序
        clusters.sort(key=lambda x: x['priority'], reverse=True)
        return clusters[:5]  # 返回前5个最优聚集区域
    
    def _find_coordination_opportunities(self, env, truck_id: int) -> List[Dict]:
        """寻找协调机会"""
        opportunities = []
        truck = env.trucks[truck_id]
        truck_pos = truck['position']
        
        for other_truck_id, other_truck in enumerate(env.trucks):
            if other_truck_id != truck_id:
                other_pos = other_truck['position']
                distance = env._euclidean_distance(truck_pos, other_pos)
                
                # 寻找两车之间的中间区域
                if distance < 30:  # 在协调范围内
                    mid_point = (
                        (truck_pos[0] + other_pos[0]) / 2,
                        (truck_pos[1] + other_pos[1]) / 2
                    )
                    
                    # 寻找中间区域的需求点
                    nearby_lockers = []
                    for locker_id, locker in enumerate(env.lockers_state):
                        locker_pos = locker['location']
                        if (env._euclidean_distance(mid_point, locker_pos) < 15 and
                            locker['demand_del'] + locker['demand_ret'] > 0):
                            nearby_lockers.append(locker_id)
                    
                    if nearby_lockers:
                        opportunities.append({
                            'partner_truck': other_truck_id,
                            'coordination_point': mid_point,
                            'distance_to_partner': distance,
                            'shared_lockers': nearby_lockers,
                            'coordination_value': len(nearby_lockers) / (1 + distance)
                        })
        
        opportunities.sort(key=lambda x: x['coordination_value'], reverse=True)
        return opportunities[:3]  # 返回前3个协调机会
    
    def _identify_exploration_targets(self, env, truck_pos: Tuple[float, float]) -> List[Dict]:
        """识别探索目标"""
        exploration_targets = []
        
        for locker_id, locker in enumerate(env.lockers_state):
            visit_count = self.visited_state_count.get(locker_id, 0)
            if visit_count < 3:  # 访问次数少的位置
                locker_pos = locker['location']
                distance = env._euclidean_distance(truck_pos, locker_pos)
                demand = locker['demand_del'] + locker['demand_ret']
                
                # 探索价值 = 需求潜力 / (1 + 访问次数) / (1 + 距离)
                exploration_value = (demand + 1) / (1 + visit_count) / (1 + distance / 10)
                
                exploration_targets.append({
                    'locker_id': locker_id,
                    'position': locker_pos,
                    'distance': distance,
                    'visit_count': visit_count,
                    'exploration_value': exploration_value
                })
        
        exploration_targets.sort(key=lambda x: x['exploration_value'], reverse=True)
        return exploration_targets[:5]  # 返回前5个探索目标
    
    def _calculate_strategy_weights(self) -> Dict[str, float]:
        """计算策略权重"""
        # 根据训练进度动态调整策略权重
        progress = min(self.training_step / 10000, 1.0)
        
        return {
            'distance_priority': 0.3 + 0.2 * progress,      # 距离优先级随训练增加
            'demand_priority': 0.4 - 0.1 * progress,        # 需求优先级随训练减少
            'coordination_priority': 0.1 + 0.2 * progress,  # 协调优先级随训练增加
            'exploration_priority': 0.2 - 0.1 * progress    # 探索优先级随训练减少
        }
    
    def _generate_action_recommendations(self, distance_tiers, demand_clusters, 
                                       coordination_opportunities, exploration_targets, 
                                       strategy_weights) -> List[Dict]:
        """生成动作推荐"""
        recommendations = []
        
        # 基于距离的推荐
        for tier_name, lockers in distance_tiers.items():
            if lockers:
                weight = strategy_weights['distance_priority']
                if tier_name == 'immediate':
                    weight *= 1.5
                elif tier_name == 'nearby':
                    weight *= 1.0
                else:
                    weight *= 0.5
                
                recommendations.append({
                    'type': 'distance_based',
                    'tier': tier_name,
                    'lockers': lockers[:3],  # 最多推荐3个
                    'weight': weight,
                    'reason': f'基于{tier_name}距离的选择'
                })
        
        # 基于需求聚集的推荐
        for cluster in demand_clusters[:2]:  # 前2个聚集区域
            recommendations.append({
                'type': 'demand_cluster',
                'lockers': cluster['lockers'][:2],
                'weight': strategy_weights['demand_priority'] * cluster['priority'],
                'reason': f'需求聚集区域，密度: {cluster["density"]:.1f}'
            })
        
        # 基于协调的推荐
        for opportunity in coordination_opportunities[:1]:  # 最佳协调机会
            recommendations.append({
                'type': 'coordination',
                'lockers': opportunity['shared_lockers'][:2],
                'weight': strategy_weights['coordination_priority'] * opportunity['coordination_value'],
                'reason': f'与卡车{opportunity["partner_truck"]}协调机会'
            })
        
        # 基于探索的推荐
        for target in exploration_targets[:2]:  # 前2个探索目标
            recommendations.append({
                'type': 'exploration',
                'lockers': [target['locker_id']],
                'weight': strategy_weights['exploration_priority'] * target['exploration_value'],
                'reason': f'探索目标，访问次数: {target["visit_count"]}'
            })
        
        # 按权重排序
        recommendations.sort(key=lambda x: x['weight'], reverse=True)
        return recommendations[:5]  # 返回前5个推荐

    def act(self, states, action_masks, env=None):
        """
        智能体动作选择 - 集成区域优先级计算
        
        Args:
            states: 状态列表
            action_masks: 动作掩码列表（每个元素是包含stop_mask和service_mask的字典）
            env: 环境实例，用于获取全局特征信息
            
        Returns:
            actions: 选择的动作列表
            log_probs: 动作的对数概率列表
            values: 状态价值列表
        """
        actions = []
        log_probs = []
        values = []
        
        with torch.no_grad():
            for i, (state, action_mask) in enumerate(zip(states, action_masks)):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 获取动作概率和状态价值
                stop_logits, service_logits = self.policy_net(state_tensor)
                main_value, route_value = self.value_net(state_tensor)
                value = main_value  # 使用主价值作为状态价值
                
                # 增强探索机制，减少对预设算法的依赖
                exploration_bonus_weight = max(0.05, 0.3 - self.training_step / 10000)  # 探索奖励权重
                temperature = max(1.0, 2.0 - self.training_step / 5000)  # 温度参数，增加随机性
                
                # 探索奖励：鼓励访问较少访问的状态
                exploration_bonus = torch.zeros_like(stop_logits)
                for stop_idx in range(stop_logits.shape[-1]):
                    visit_count = self.visited_state_count.get(stop_idx, 0)
                    bonus = exploration_bonus_weight / (1 + visit_count)
                    exploration_bonus[0][stop_idx] = bonus
                
                # 应用探索奖励
                stop_logits = stop_logits + exploration_bonus
                
                # 仅在训练早期提供极轻微的需求引导（权重更小，时间更短）
                if env is not None and self.training_step < 500:  # 减少到前500步
                    guidance_weight = max(0.02 * (1.0 - self.training_step / 500), 0.0)  # 更小的引导权重
                    
                    # 简单的需求密度引导
                    basic_guidance = []
                    for stop_idx in range(stop_logits.shape[-1]):
                        if stop_idx < len(env.lockers_state):
                            locker = env.lockers_state[stop_idx]
                            demand_score = (locker.get('demand_del', 0) + locker.get('demand_ret', 0)) / 20.0
                            basic_guidance.append(min(demand_score, 0.5))  # 进一步限制影响
                        else:
                            basic_guidance.append(0.0)
                    
                    if guidance_weight > 0:
                        guidance_tensor = torch.FloatTensor(basic_guidance).unsqueeze(0).to(self.device)
                        stop_logits = stop_logits + guidance_weight * guidance_tensor
                
                # 增强的自适应探索噪声
                base_noise_scale = 0.25  # 增加基础噪声
                adaptive_noise_scale = base_noise_scale * max(0.5, self.entropy_coef / self.initial_entropy_coef)
                
                # 为每个卡车添加不同的随机噪声
                truck_specific_noise = torch.randn_like(stop_logits) * adaptive_noise_scale * (i + 1) / len(states)
                
                # 应用温度缩放增强探索
                stop_logits = (stop_logits + truck_specific_noise) / temperature
                
                # 应用停靠点动作掩码
                if action_mask is not None and 'stop_mask' in action_mask:
                    mask_tensor = action_mask['stop_mask'].unsqueeze(0).float().to(self.device)
                    # 将掩码应用到logits上，无效动作设为很小的值
                    masked_stop_logits = stop_logits + (mask_tensor - 1.0) * 1e9
                else:
                    masked_stop_logits = stop_logits
                
                # 选择停靠点动作
                stop_probs = F.softmax(masked_stop_logits, dim=-1)
                stop_dist = Categorical(stop_probs)
                select_stop = stop_dist.sample()
                stop_log_prob = stop_dist.log_prob(select_stop)
                
                # 更新访问计数
                stop_id = select_stop.item()
                self.visited_state_count[stop_id] = self.visited_state_count.get(stop_id, 0) + 1
                
                # 应用服务区域动作掩码并选择服务区域
                if action_mask is not None and 'service_mask' in action_mask:
                    service_mask_tensor = action_mask['service_mask'].unsqueeze(0).float().to(self.device)
                    masked_service_logits = service_logits + (service_mask_tensor - 1.0) * 1e9
                else:
                    masked_service_logits = service_logits
                
                # 使用Bernoulli分布为每个快递柜选择是否服务
                service_probs = torch.sigmoid(masked_service_logits)
                service_dist = Bernoulli(service_probs)
                service_area_tensor = service_dist.sample()
                service_log_prob = service_dist.log_prob(service_area_tensor).sum()
                
                # 构建复合动作
                action = {
                    'select_stop': select_stop.item(),
                    'service_area': service_area_tensor.squeeze(0).cpu().numpy().astype(int).tolist()
                }
                
                # 计算总的对数概率
                total_log_prob = stop_log_prob + service_log_prob
                
                actions.append(action)
                log_probs.append(total_log_prob.item())
                values.append(value.item())
        
        return actions, log_probs, values

    def update(self, states, actions, rewards, log_probs, values, dones, optimized_config=None):
        """
        更新策略网络
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            log_probs: 对数概率序列
            values: 价值序列
            dones: 结束标志序列
            optimized_config: 优化配置对象，包含批次大小和更新频率设置
            
        Returns:
            dict: 包含策略损失和价值损失的字典
        """
        # 更新训练步数
        self.training_step += 1
        
        # 应用预热机制
        self._apply_warmup()
        
        # 计算优势函数和目标价值
        advantages, returns = self._compute_advantages(rewards, values, dones)
        
        # 转换为张量并移动到设备
        if isinstance(states, torch.Tensor):
            states = states.to(self.device)
        else:
            states = torch.FloatTensor(states).to(self.device)
            
        if isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
            
        if isinstance(log_probs, torch.Tensor):
            old_log_probs = log_probs.to(self.device)
        else:
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势函数以提高训练稳定性
        if len(advantages) > 1:
            advantages_mean = advantages.mean()
            advantages_std = advantages.std() + 1e-8
            advantages = (advantages - advantages_mean) / advantages_std
        
        # 标准化回报值以提高价值网络训练稳定性
        if len(returns) > 1:
            returns_mean = returns.mean()
            returns_std = returns.std() + 1e-8
            normalized_returns = (returns - returns_mean) / returns_std
        else:
            normalized_returns = returns
            returns_mean = 0
            returns_std = 1
        
        # 标准化优势函数以提高数值稳定性
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        # 数据集大小
        dataset_size = states.size(0)
        
        # 使用优化配置的批次大小和更新频率
        if optimized_config is not None:
            mini_batch_size = optimized_config.BATCH_SIZE
            update_epochs = optimized_config.UPDATE_FREQUENCY
            print(f"🔧 使用优化批次配置 - 批次大小: {mini_batch_size}, 更新频率: {update_epochs}")
        else:
            mini_batch_size = self.mini_batch_size
            update_epochs = self.update_epochs
        
        # 多轮更新 - 使用优化的批次处理提高训练稳定性
        for update_round in range(update_epochs):  # PPO更新轮数
            # 随机打乱数据
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 获取小批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = normalized_returns[batch_indices]
                # 前向传播
                stop_logits, service_logits = self.policy_net(batch_states)
                
                # 处理复合动作的对数概率计算
                # 假设batch_actions是复合动作的字典或者已经转换为stop动作的索引
                if isinstance(batch_actions[0], dict) if len(batch_actions) > 0 else False:
                    # 如果是复合动作字典，提取stop动作
                    stop_actions = torch.LongTensor([action['select_stop'] for action in batch_actions]).to(self.device)
                else:
                    # 如果已经是stop动作索引
                    stop_actions = batch_actions
                
                # 计算stop动作的概率和熵
                stop_probs = F.softmax(stop_logits, dim=-1)
                stop_dist = Categorical(stop_probs)
                stop_log_probs = stop_dist.log_prob(stop_actions)
                stop_entropy = stop_dist.entropy().mean()
                
                # 计算service动作的概率和熵（使用伯努利分布）
                service_probs = torch.sigmoid(service_logits)
                service_dist = Bernoulli(service_probs)
                service_entropy = service_dist.entropy().mean()
                
                # 组合对数概率和熵
                new_log_probs = stop_log_probs  # 主要使用stop动作的对数概率
                entropy = stop_entropy + 0.1 * service_entropy  # 组合熵，给service较小权重
                
                # 获取价值网络的双输出：主价值和路线价值
                main_values, route_values = self.value_net(batch_states)
                new_values = main_values.squeeze()
                route_values = route_values.squeeze()
            
                # 计算比率，添加数值稳定性
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -20, 20)  # 防止数值溢出
                ratio = torch.exp(log_ratio)
                
                # 计算策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失，使用标准化的回报值和改进的损失函数
                # 主价值损失
                main_value_loss = F.smooth_l1_loss(new_values, batch_returns)
                
                # 路线规划价值损失 - 使用路线规划相关的奖励信号
                # 这里我们使用主价值作为路线价值的目标，但可以根据需要调整
                route_value_loss = F.smooth_l1_loss(route_values, batch_returns * 0.8)  # 路线价值稍微保守
                
                # 添加路线规划奖励项
                route_planning_bonus = self._calculate_route_planning_bonus(batch_states, batch_actions, batch_returns)
                
                # 添加长期价值估计奖励
                long_term_value_bonus = self._calculate_long_term_value_bonus(new_values, route_values, batch_returns)
                
                # 添加协调奖励项
                coordination_bonus = self._calculate_coordination_bonus(batch_states, batch_actions)
                
                # 添加路径效率奖励
                path_efficiency_bonus = self._calculate_path_efficiency_bonus(batch_states, batch_returns)
                
                # 组合价值损失，包含所有奖励项
                normalized_value_loss = (main_value_loss + 0.3 * route_value_loss 
                                       - 0.1 * route_planning_bonus 
                                       - 0.15 * long_term_value_bonus
                                       - 0.05 * coordination_bonus
                                       - 0.08 * path_efficiency_bonus)
                
                # 计算探索奖励 - 鼓励动作多样性
                exploration_bonus = 0.0
                if entropy > self.action_diversity_threshold:
                    exploration_bonus = self.exploration_bonus_coef * (entropy - self.action_diversity_threshold)
                
                # 策略损失计算 - 包含探索奖励
                policy_total_loss = policy_loss - self.entropy_coef * entropy - exploration_bonus
                
                # 价值损失计算（使用更小的系数）
                value_total_loss = self.value_loss_coef * normalized_value_loss
                
                # 分别进行反向传播
                # 1. 策略网络更新
                self.policy_optimizer.zero_grad()
                policy_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # 2. 价值网络更新
                self.value_optimizer.zero_grad()
                value_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm * 0.5)  # 价值网络使用更小的梯度裁剪
                self.value_optimizer.step()
                
                # 累积损失用于记录
                total_policy_loss += policy_loss.item()
                total_value_loss += normalized_value_loss.item()
        
        # 更新旧策略
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        # 计算总的小批次数量
        total_mini_batches = self.update_epochs * ((dataset_size + self.mini_batch_size - 1) // self.mini_batch_size)
        
        # 预热期结束后，使用学习率调度器
        if self.training_step > max(self.policy_warmup_steps, self.value_warmup_steps):
            self.policy_scheduler.step()
            self.value_scheduler.step()
        
        # 自适应熵系数衰减 - 防止过早收敛
        if self.training_step % 10 == 0:  # 每10步更新一次
            self.entropy_coef = max(
                self.min_entropy_coef,
                self.entropy_coef * self.entropy_decay_rate
            )
        
        # 返回平均损失和探索信息
        return {
            'policy_loss': total_policy_loss / total_mini_batches,
            'value_loss': total_value_loss / total_mini_batches,
            'entropy_coef': self.entropy_coef,
            'exploration_level': entropy.item() if 'entropy' in locals() else 0.0
        }

    def _apply_warmup(self):
        """
        应用学习率预热机制
        在训练初期逐渐增加学习率，帮助深度网络稳定训练
        """
        # 策略网络预热
        if self.training_step <= self.policy_warmup_steps:
            warmup_factor = self.training_step / self.policy_warmup_steps
            current_lr = self.policy_base_lr * warmup_factor
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # 价值网络预热
        if self.training_step <= self.value_warmup_steps:
            warmup_factor = self.training_step / self.value_warmup_steps
            current_lr = self.value_base_lr * warmup_factor
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] = current_lr

    def _compute_advantages(self, rewards, values, dones, gamma=None, gae_lambda=None):
        """
        计算优势函数和回报
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            dones: 结束标志序列
            gamma: 折扣因子（如果为None则使用config.GAMMA）
            gae_lambda: GAE参数（如果为None则使用config.GAE_LAMBDA）
            
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        # 使用config中的值作为默认值
        if gamma is None:
            gamma = config.GAMMA
        if gae_lambda is None:
            gae_lambda = config.GAE_LAMBDA
        advantages = []
        returns = []
        gae = torch.tensor(0.0, device=self.device)
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns

    def _calculate_route_planning_bonus(self, batch_states: torch.Tensor, batch_actions: torch.Tensor, batch_returns: torch.Tensor) -> torch.Tensor:
        """
        计算路线规划奖励项
        
        Args:
            batch_states: 批次状态
            batch_actions: 批次动作
            batch_returns: 批次回报
            
        Returns:
            路线规划奖励
        """
        try:
            # 提取状态中的路线规划相关特征
            # 假设状态的最后36维是路线规划特征 (12+8+6+10)
            route_features = batch_states[:, -36:]
            
            # 路径效率奖励 (前12维)
            path_efficiency_features = route_features[:, :12]
            path_efficiency_score = torch.mean(path_efficiency_features, dim=1)
            
            # 历史路径奖励 (13-20维)
            path_history_features = route_features[:, 12:20]
            path_history_score = torch.mean(path_history_features, dim=1)
            
            # 未来需求预测奖励 (21-26维)
            future_demand_features = route_features[:, 20:26]
            future_demand_score = torch.mean(future_demand_features, dim=1)
            
            # 协调特征奖励 (27-36维)
            coordination_features = route_features[:, 26:36]
            coordination_score = torch.mean(coordination_features, dim=1)
            
            # 组合路线规划奖励
            route_planning_bonus = (0.4 * path_efficiency_score + 
                                  0.2 * path_history_score + 
                                  0.2 * future_demand_score + 
                                  0.2 * coordination_score)
            
            # 归一化到合理范围
            route_planning_bonus = torch.tanh(route_planning_bonus)
            
            return route_planning_bonus.mean()
            
        except Exception as e:
            # 如果计算失败，返回零奖励
            return torch.tensor(0.0, device=self.device)

    def _calculate_long_term_value_bonus(self, main_values: torch.Tensor, route_values: torch.Tensor, batch_returns: torch.Tensor) -> torch.Tensor:
        """
        计算长期价值估计奖励
        
        Args:
            main_values: 主价值估计
            route_values: 路线价值估计
            batch_returns: 批次回报
            
        Returns:
            长期价值估计奖励
        """
        try:
            # 计算价值估计的一致性
            value_consistency = 1.0 - torch.abs(main_values - route_values).mean()
            
            # 计算价值估计的准确性
            main_accuracy = 1.0 - torch.abs(main_values - batch_returns).mean() / (torch.abs(batch_returns).mean() + 1e-8)
            route_accuracy = 1.0 - torch.abs(route_values - batch_returns * 0.8).mean() / (torch.abs(batch_returns).mean() + 1e-8)
            
            # 组合长期价值奖励
            long_term_bonus = 0.4 * value_consistency + 0.3 * main_accuracy + 0.3 * route_accuracy
            
            # 归一化到合理范围
            long_term_bonus = torch.tanh(long_term_bonus)
            
            return long_term_bonus
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)

    def _calculate_coordination_bonus(self, batch_states: torch.Tensor, batch_actions: torch.Tensor) -> torch.Tensor:
        """
        计算协调奖励项
        
        Args:
            batch_states: 批次状态
            batch_actions: 批次动作
            
        Returns:
            协调奖励
        """
        try:
            # 提取协调相关特征 (状态的最后10维)
            coordination_features = batch_states[:, -10:]
            
            # 计算协调效率
            coordination_efficiency = torch.mean(coordination_features[:, :5], dim=1)  # 前5维：协调效率指标
            
            # 计算团队合作指标
            team_cooperation = torch.mean(coordination_features[:, 5:], dim=1)  # 后5维：团队合作指标
            
            # 组合协调奖励
            coordination_bonus = 0.6 * coordination_efficiency + 0.4 * team_cooperation
            
            # 归一化
            coordination_bonus = torch.tanh(coordination_bonus)
            
            return coordination_bonus.mean()
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)

    def _calculate_path_efficiency_bonus(self, batch_states: torch.Tensor, batch_returns: torch.Tensor) -> torch.Tensor:
        """
        计算路径效率奖励项
        
        Args:
            batch_states: 批次状态
            batch_returns: 批次回报
            
        Returns:
            路径效率奖励
        """
        try:
            # 提取路径效率相关特征 (状态的-36到-24维，即路线规划特征的前12维)
            path_efficiency_features = batch_states[:, -36:-24]
            
            # 距离效率 (前4维)
            distance_efficiency = torch.mean(path_efficiency_features[:, :4], dim=1)
            
            # 时间效率 (5-8维)
            time_efficiency = torch.mean(path_efficiency_features[:, 4:8], dim=1)
            
            # 负载效率 (9-12维)
            load_efficiency = torch.mean(path_efficiency_features[:, 8:12], dim=1)
            
            # 组合路径效率奖励
            path_efficiency_bonus = (0.4 * distance_efficiency + 
                                   0.3 * time_efficiency + 
                                   0.3 * load_efficiency)
            
            # 根据回报调整奖励强度
            return_magnitude = torch.abs(batch_returns).mean()
            adjusted_bonus = path_efficiency_bonus * torch.tanh(return_magnitude)
            
            # 归一化
            adjusted_bonus = torch.tanh(adjusted_bonus)
            
            return adjusted_bonus.mean()
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)


def validate_model(mappo, validation_env, num_validation_episodes=5):
    """
    验证模型性能，检测过拟合
    
    Args:
        mappo: 训练的MAPPO模型
        validation_env: 验证环境
        num_validation_episodes: 验证轮数
    
    Returns:
        平均验证奖励
    """
    mappo.policy_net.eval()  # 设置为评估模式
    mappo.value_net.eval()
    
    validation_rewards = []
    
    for _ in range(num_validation_episodes):
        state, action_mask = validation_env.reset()  # 正确解包reset返回的元组
        episode_reward = 0
        done = False
        
        while not done:
            # 获取动作掩码
            action_masks = validation_env.get_action_masks()
            
            # 获取每个卡车的特定状态
            truck_states = validation_env.get_truck_specific_states()
            
            # 获取动作（不添加探索噪声，传递环境实例以启用区域优先级计算）
            with torch.no_grad():
                actions, _, _ = mappo.act(truck_states, action_masks, validation_env)
            
            # 执行动作
            next_state, reward, done, _ = validation_env.step(actions)
            # 如果reward是列表，求和；如果是单个值，直接使用
            if isinstance(reward, list):
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            state = next_state
        
        validation_rewards.append(episode_reward)
    
    mappo.policy_net.train()  # 恢复训练模式
    mappo.value_net.train()
    
    return np.mean(validation_rewards)

def train_marl(env, num_episodes=200000, training_manager=None, curriculum_manager=None, 
               optimized_config=None, reward_normalizer=None, lr_scheduler=None):
    """
    多智能体强化学习训练函数
    
    参数:
    - env: 环境实例
    - num_episodes: 训练轮数
    - training_manager: 训练管理器，用于记录训练进度
    - curriculum_manager: 课程学习管理器，如果为None则创建默认的
    - optimized_config: 优化的训练配置
    - reward_normalizer: 奖励归一化器
    - lr_scheduler: 学习率调度器
    
    返回:
    - 训练好的策略网络
    """
    # 如果没有传入课程学习管理器，则不使用课程学习，保持原始环境配置
    if curriculum_manager is None:
        print("🎓 使用原始环境配置，不应用课程学习")
        print(f"   快递柜数量: {env.num_lockers}")
        print(f"   卡车数量: {env.num_trucks}")
    else:
        # 使用传入的课程学习管理器配置
        initial_curriculum_config = curriculum_manager.get_current_config()
        print(f"🎓 应用课程学习配置: {curriculum_manager.current_stage.name}")
        print(f"   快递柜数量: {initial_curriculum_config['num_lockers']}")
        print(f"   卡车数量: {initial_curriculum_config['num_trucks']}")
        
        # 更新环境配置
        if hasattr(env, 'update_curriculum_config'):
            env.update_curriculum_config(initial_curriculum_config)
    
    # 使用更新后的环境配置创建MAPPO实例
    num_trucks = env.num_trucks
    # 获取单个卡车的状态维度（而不是所有卡车的总状态维度）
    dummy_state, _ = env.reset()
    truck_states = env.get_truck_specific_states()
    state_dim = len(truck_states[0]) if truck_states else env.state_dim  # 单个卡车的状态维度

    action_dim = {
        "select_stop": env.num_lockers + 1,  # 0:仓库, 1-n:快递柜
        "service_area": env.num_lockers  # 每个快递柜一个二进制选择
    }
    
    print(f"🤖 创建MAPPO实例: 卡车数量={num_trucks}, 状态维度={state_dim}")
    
    # 使用优化配置创建MAPPO智能体
    if optimized_config is not None:
        print(f"📊 使用优化训练配置: 学习率={optimized_config.LEARNING_RATE}, 裁剪范围={optimized_config.CLIP_RANGE}")
        mappo = MAPPO(num_trucks, state_dim, action_dim, lr=optimized_config.LEARNING_RATE)
        # 应用优化的超参数
        mappo.update_hyperparameters({
            'clip_ratio': optimized_config.CLIP_RANGE,
            'value_coef': optimized_config.VF_COEF,
            'entropy_coef': optimized_config.ENT_COEF,
            'max_grad_norm': optimized_config.MAX_GRAD_NORM
        })
    else:
        # 使用config中的学习率
        import config
        mappo = MAPPO(num_trucks, state_dim, action_dim, lr=config.LEARNING_RATE)
    
    # 初始化自适应奖励调度器
    reward_function = RewardFunction(max_timesteps=env.max_timesteps)
    adaptive_scheduler = AdaptiveRewardScheduler(reward_function)
    print(f"🎯 自适应奖励调度器已初始化")
    
    # 创建验证环境（用于检测过拟合）
    validation_env = TruckSchedulingEnv(verbose=False)
    if curriculum_manager is not None:
        validation_env.update_curriculum_config(curriculum_manager.get_current_config())
    else:
        # 使用与训练环境相同的配置
        validation_env.num_lockers = env.num_lockers
        validation_env.num_trucks = env.num_trucks
    
    # 训练统计
    episode_rewards = []
    validation_rewards = []
    best_reward = float('-inf')
    best_validation_reward = float('-inf')
    best_episode = 0
    
    # 性能监控
    performance_window = 50  # 修改为50个episode的窗口
    recent_rewards = []
    validation_frequency = 50  # 每50个episode进行一次验证
    
    # 早停机制参数（修改为50个episode不超过最佳就停止）
    early_stop_patience = 50  # 修改为50个episode没有改善就停止
    early_stop_min_delta = 0.0  # 设置为0，只要不超过最佳就停止
    validation_patience = 15    # 验证集性能下降容忍度
    no_improvement_count = 0
    validation_decline_count = 0
    best_avg_reward = float('-inf')
    
    # 模型保存路径
    best_model_path = "best_model.pth"
    
    # 初始化损失信息
    loss_info = {'policy_loss': 0.0, 'value_loss': 0.0}
    
    print("🚀 开始训练")
    if curriculum_manager is not None:
        print(f"轮数: {num_episodes} | 阶段: {curriculum_manager.current_stage.name} | 步数: {env.max_timesteps}")
    else:
        print(f"轮数: {num_episodes} | 原始环境配置 | 步数: {env.max_timesteps}")
    print("=" * 50)

    # 训练开始时间
    training_start_time = time.time()
    
    # 创建tqdm进度条，优化显示格式
    progress_bar = tqdm(range(num_episodes), desc="🚀 MAPPO训练", ncols=120, 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                       dynamic_ncols=True, leave=True)
    
    for episode in progress_bar:
        # 进度报告（每200轮）
        if episode % 200 == 0 and episode > 0:
            elapsed_time = time.time() - training_start_time
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            progress_percent = (episode / num_episodes) * 100
            
            # print(f"\n📊 训练进度报告 - Episode {episode}/{num_episodes} ({progress_percent:.1f}%)")
            # print(f"   ⏱️  已用时间: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
            # print(f"   🎯 平均奖励: {avg_reward:.2f} | 🏆 最佳奖励: {best_reward:.2f}")
            # print(f"   📚 课程阶段: {curriculum_manager.current_stage.name}")
            # print(f"   📈 阶段进度: {curriculum_manager.episodes_in_stage}/{curriculum_manager.current_stage.episodes_required}")
            
            # # 预估剩余时间
            # if episode > 0:
            #     avg_time_per_episode = elapsed_time / episode
            #     remaining_episodes = num_episodes - episode
            #     estimated_remaining = avg_time_per_episode * remaining_episodes
            #     print(f"   ⏳ 预计剩余: {estimated_remaining/60:.1f}分钟")
                
            # # 性能分析和优化建议
            # if len(recent_rewards) >= 100:
            #     recent_100_avg = np.mean(recent_rewards[-100:])
            #     if recent_100_avg > avg_reward * 1.1:
            #         print(f"   ✅ 性能提升中 (最近100轮: {recent_100_avg:.2f})")
            #     elif recent_100_avg < avg_reward * 0.9:
            #         print(f"   ⚠️  性能下降 (最近100轮: {recent_100_avg:.2f})")
            #     else:
            #         print(f"   📈 性能稳定 (最近100轮: {recent_100_avg:.2f})")
            # print("-" * 60)
        # 获取当前课程配置（如果启用课程学习）
        if curriculum_manager is not None:
            curriculum_config = curriculum_manager.get_current_config()
            
            # 记录更新前的环境配置
            old_num_trucks = env.num_trucks
            old_state_dim = env.state_dim
            
            # 更新环境配置（如果需要）
            if hasattr(env, 'update_curriculum_config'):
                env.update_curriculum_config(curriculum_config)
            
            # 获取当前单个卡车的状态维度
            current_truck_states = env.get_truck_specific_states()
            current_single_truck_state_dim = len(current_truck_states[0]) if current_truck_states else env.state_dim
            
            # 检查是否需要重新创建MAPPO实例
            if env.num_trucks != old_num_trucks or current_single_truck_state_dim != state_dim:
                print(f"🔄 检测到环境配置变化: 卡车数量 {old_num_trucks} -> {env.num_trucks}, 单卡车状态维度 {state_dim} -> {current_single_truck_state_dim}")
                print(f"   重新创建MAPPO实例以适应新配置...")
                
                # 更新相关变量
                num_trucks = env.num_trucks
                state_dim = current_single_truck_state_dim
                action_dim = {
                    "select_stop": env.num_lockers + 1,  # 0:仓库, 1-n:快递柜
                    "service_area": env.num_lockers  # 每个快递柜一个二进制选择
                }
                
                # 重新创建MAPPO智能体（使用config中的学习率）
                import config
                mappo = MAPPO(num_trucks, state_dim, action_dim, lr=config.LEARNING_RATE)
                print(f"   ✅ MAPPO实例重新创建完成")
            
            # 获取自适应超参数
            hyperparams = curriculum_manager.get_adaptive_hyperparameters()
            mappo.update_hyperparameters(hyperparams)
        
        state, action_mask = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        # 初始化奖励分解累计
        episode_breakdown = {
            "service_reward": 0.0,
            "efficiency_reward": 0.0,
            "cost_penalty": 0.0,
            # total_reward 已由 episode_reward 变量跟踪
        }
        
        # 存储轨迹数据
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'action_masks': []
        }

        while not done and step_count < env.max_timesteps:
            # 获取所有卡车的动作掩码
            action_masks = env.get_action_masks()
            
            # 获取每个卡车的特定状态
            truck_states = env.get_truck_specific_states()
            
            # 获取动作（带掩码，传递环境实例以启用区域优先级计算）
            actions, log_probs, values = mappo.act(
                truck_states,
                action_masks,
                env  # 传递环境实例以启用区域优先级计算
            )

            # 执行动作
            next_state, rewards, done, next_action_mask = env.step(actions)
            episode_reward += sum(rewards)
            
            # 累积奖励分解
            if hasattr(env, 'last_reward_breakdown') and env.last_reward_breakdown:
                for bd in env.last_reward_breakdown:
                    for key in episode_breakdown:
                        if key in bd:
                            episode_breakdown[key] += bd[key]
            
            # 存储轨迹数据（使用卡车特定状态）
            trajectory['states'].append(truck_states)
            trajectory['actions'].append(actions)
            trajectory['rewards'].append(rewards)
            trajectory['log_probs'].append(log_probs)
            trajectory['values'].append(values)
            trajectory['action_masks'].append(action_masks)

            # 更新状态和掩码
            state = next_state
            action_mask = next_action_mask
            step_count += 1

        # 应用奖励归一化（如果启用）
        if reward_normalizer is not None and len(trajectory['states']) > 0:
            # 收集所有奖励进行归一化
            all_episode_rewards = []
            for t in range(len(trajectory['rewards'])):
                all_episode_rewards.extend(trajectory['rewards'][t])
            
            # 更新归一化器并应用归一化
            reward_normalizer.update(all_episode_rewards)
            
            # 归一化轨迹中的奖励
            for t in range(len(trajectory['rewards'])):
                normalized_rewards = reward_normalizer.normalize(trajectory['rewards'][t])
                trajectory['rewards'][t] = normalized_rewards
        
        # 轨迹结束后进行策略更新
        if len(trajectory['states']) > 0:
            # 使用完整GAE算法处理整个轨迹
            T = len(trajectory['rewards'])
            
            # 准备数据
            rewards_tensor = []
            values_tensor = []
            next_values_tensor = []
            dones_list = []
            
            for t in range(T):
                rewards_tensor.append(torch.tensor(trajectory['rewards'][t], dtype=torch.float32, device=mappo.device))
                values_tensor.append(torch.tensor(trajectory['values'][t], dtype=torch.float32, device=mappo.device))
                
                # 计算下一状态的价值
                if t < T - 1:
                    next_values_tensor.append(torch.tensor(trajectory['values'][t + 1], dtype=torch.float32, device=mappo.device))
                else:
                    # 最后一步，计算终端状态价值
                    final_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(mappo.device)
                    final_value = mappo.value_net(final_state_tensor).squeeze() if not done else 0.0
                    next_values_tensor.append(torch.tensor([final_value] * num_trucks, dtype=torch.float32, device=mappo.device))
                
                dones_list.append(done if t == T - 1 else False)
            
            # 使用GAE计算优势函数和回报
            advantages, returns = mappo._compute_advantages(
                rewards_tensor, values_tensor, dones_list
            )
            
            # 准备数据用于MAPPO批量更新
            all_states = []
            all_actions = []
            all_rewards = []
            all_log_probs = []
            all_values = []
            all_dones = []
            
            for t in range(T):
                for i in range(num_trucks):
                    all_states.append(trajectory['states'][t][i])
                    all_actions.append(trajectory['actions'][t][i])
                    all_rewards.append(trajectory['rewards'][t][i])
                    all_log_probs.append(trajectory['log_probs'][t][i])
                    all_values.append(trajectory['values'][t][i])
                    all_dones.append(True if t == T - 1 else False)
            
            # 转换为张量 - 先转换为numpy数组以提高性能
            states_array = np.array(all_states)
            states_tensor = torch.FloatTensor(states_array).to(mappo.device)
            # 对于复合动作，我们只提取select_stop部分用于训练
            if isinstance(all_actions[0], dict):
                actions_tensor = torch.LongTensor([action['select_stop'] for action in all_actions]).to(mappo.device)
            else:
                actions_tensor = torch.LongTensor(all_actions).to(mappo.device)
            rewards_tensor = torch.FloatTensor(all_rewards).to(mappo.device)
            log_probs_tensor = torch.FloatTensor(all_log_probs).to(mappo.device)
            values_tensor = torch.FloatTensor(all_values).to(mappo.device)
            
            # 使用优化配置的批次大小进行更新
            if optimized_config is not None:
                # 使用优化配置的批次大小和更新频率
                batch_size = optimized_config.BATCH_SIZE
                n_epochs = optimized_config.N_EPOCHS
                
                # 分批次更新以提高稳定性
                total_samples = len(all_states)
                indices = torch.randperm(total_samples)
                
                for epoch in range(n_epochs):
                    for start_idx in range(0, total_samples, batch_size):
                        end_idx = min(start_idx + batch_size, total_samples)
                        batch_indices = indices[start_idx:end_idx]
                        
                        batch_states = states_tensor[batch_indices]
                        batch_actions = actions_tensor[batch_indices]
                        batch_rewards = rewards_tensor[batch_indices]
                        batch_log_probs = log_probs_tensor[batch_indices]
                        batch_values = values_tensor[batch_indices]
                        batch_dones = [all_dones[i] for i in batch_indices]
                        
                        # 执行批次更新
                        loss_info = mappo.update(
                            batch_states, batch_actions, batch_rewards, 
                            batch_log_probs, batch_values, batch_dones, optimized_config
                        )
            else:
                # 使用原始的更新方法
                loss_info = mappo.update(
                    states_tensor, actions_tensor, rewards_tensor, 
                    log_probs_tensor, values_tensor, all_dones, optimized_config
                )
        
        # 更新旧策略网络（每隔一定步数）
        if episode % 10 == 0:
            mappo.old_policy_net.load_state_dict(mappo.policy_net.state_dict())
        
        # 应用学习率调度（如果启用优化配置）
        if lr_scheduler is not None and optimized_config is not None:
            # 使用优化配置的学习率调度
            if episode % optimized_config.LR_SCHEDULE['step_size'] == 0 and episode > 0:
                new_lr = lr_scheduler.step(episode)
                # 更新MAPPO的学习率
                for param_group in mappo.policy_optimizer.param_groups:
                    param_group['lr'] = new_lr
                for param_group in mappo.value_optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📉 学习率调度更新 - Episode {episode}, 新学习率: {new_lr:.6f}")
        else:
            # 使用原始的学习率调度器（每2000个episode）
            if episode % 2000 == 0 and episode > 0:
                if hasattr(mappo, 'policy_scheduler') and hasattr(mappo, 'value_scheduler'):
                    mappo.policy_scheduler.step()
                    mappo.value_scheduler.step()
                    current_policy_lr = mappo.policy_optimizer.param_groups[0]['lr']
                    current_value_lr = mappo.value_optimizer.param_groups[0]['lr']
                    print(f"📉 学习率更新 - 策略: {current_policy_lr:.6f}, 价值: {current_value_lr:.6f}")
        
        # 奖励平滑处理（如果训练管理器启用了优化功能）
        smoothed_reward = episode_reward
        if training_manager and hasattr(training_manager, 'reward_smoother') and training_manager.reward_smoother:
            smoothed_reward = training_manager.reward_smoother.smooth(episode_reward)
        
        # 记录奖励和性能监控
        episode_rewards.append(episode_reward)
        recent_rewards.append(smoothed_reward)  # 使用平滑后的奖励计算平均值
        if len(recent_rewards) > performance_window:
            recent_rewards.pop(0)
        
        # 更新最佳奖励
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode
            # 同时更新MAPPO实例的最佳性能记录
            mappo.best_performance = best_reward
        
        # 计算当前性能指标
        current_avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        episode_success = episode_reward > 0  # 简单的成功判断标准
        
        # 收敛检测（如果训练管理器启用了优化功能）
        convergence_info = None
        if training_manager and hasattr(training_manager, 'convergence_detector') and training_manager.convergence_detector:
            convergence_info = training_manager.convergence_detector.check_convergence(smoothed_reward)
            
            # 根据收敛状态进行相应处理
            status = convergence_info['status']
            if status in ['converged', 'converging_with_improvement']:
                print(f"\n🎯 收敛检测: {convergence_info['message']}")
                print(f"   置信度: {convergence_info['confidence']:.3f}")
                
                # 如果是局部最优，增加探索
                if status == 'local_optimum':
                    if hasattr(training_manager, 'exploration_scheduler'):
                        current_params = training_manager.exploration_scheduler.update(episode, 0.5)  # 增加方差
                        print(f"   🔍 增加探索率: {current_params['epsilon']:.3f}")
                
                # 如果收敛且有改进，可以降低学习率以稳定训练
                elif status == 'converging_with_improvement':
                    if hasattr(training_manager, 'lr_scheduler'):
                        new_lr = training_manager.lr_scheduler.update(current_avg_reward, episode)
                        # 更新MAPPO的学习率
                        for param_group in mappo.policy_optimizer.param_groups:
                            param_group['lr'] = new_lr
                        for param_group in mappo.value_optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"   📉 调整学习率: {new_lr:.2e}")
        
        # 早停机制：检查性能改善
        if len(recent_rewards) >= performance_window:  # 只有在有足够数据时才进行早停检查
            if current_avg_reward > best_avg_reward + early_stop_min_delta:
                best_avg_reward = current_avg_reward
                no_improvement_count = 0
                
                # 保存最佳模型
                torch.save({
                    'policy_net_state_dict': mappo.policy_net.state_dict(),
                    'value_net_state_dict': mappo.value_net.state_dict(),
                    'policy_optimizer_state_dict': mappo.policy_optimizer.state_dict(),
                    'value_optimizer_state_dict': mappo.value_optimizer.state_dict(),
                    'episode': episode,
                    'best_avg_reward': best_avg_reward,
                    'episode_reward': episode_reward
                }, best_model_path)
                print(f"💾 保存最佳模型 - Episode {episode}, 平均奖励: {best_avg_reward:.2f}")
                
                # 简化的自适应奖励调度器状态报告（仅在保存最佳模型时）
                stability_metrics = adaptive_scheduler.get_stability_metrics()
            else:
                no_improvement_count += 1
                
            # 检查是否需要早停
            if no_improvement_count >= early_stop_patience:
                print(f"\n🛑 早停触发！连续 {early_stop_patience} 个episode没有改善")
                print(f"   最佳平均奖励: {best_avg_reward:.2f}")
                print(f"   当前平均奖励: {current_avg_reward:.2f}")
                print(f"   训练在 Episode {episode} 停止")
                break
        
        # 更新课程学习管理器（如果启用）
        if curriculum_manager is not None:
            curriculum_manager.update_performance(episode_reward, episode_success)
        
        # 更新自适应奖励调度器
        completion_rate = env._calculate_completion_rate()
        efficiency = env._calculate_path_efficiency()
        episode_performance = {
            'total_reward': episode_reward,
            'completion_rate': completion_rate,
            'efficiency': efficiency,
            'step_count': step_count,
            'episode_success': episode_success
        }
        adaptive_scheduler.update_weights(episode_performance)
        
        # 记录训练进度到训练管理器
        if training_manager is not None:
            metrics = {
                'episode_success': episode_success,
                'current_avg_reward': current_avg_reward,
                'best_reward': best_reward,
                'step_count': step_count,
                'completion_rate': completion_rate,
                'efficiency': efficiency,
                'smoothed_reward': smoothed_reward,  # 添加平滑奖励
                'raw_reward': episode_reward,  # 原始奖励
            }
            
            # 添加奖励分解信息
            if 'episode_breakdown' in locals():
                metrics.update({
                    'reward_service': episode_breakdown['service_reward'],
                    'reward_efficiency': episode_breakdown['efficiency_reward'],
                    'reward_cost': episode_breakdown['cost_penalty']
                })
            
            # 添加收敛信息（如果有）
            if convergence_info:
                metrics.update({
                    'convergence_status': convergence_info['status'],
                    'convergence_message': convergence_info['message'],
                    'convergence_confidence': convergence_info['confidence']
                })
                
                # 添加统计信息（如果有）
                if 'statistics' in convergence_info:
                    stats = convergence_info['statistics']
                    metrics.update({
                        'convergence_mean_reward': stats.get('mean_reward', 0),
                        'convergence_std_reward': stats.get('std_reward', 0),
                        'convergence_cv': stats.get('coefficient_of_variation', 0),
                        'convergence_improvement_rate': stats.get('improvement_rate', 0)
                    })
            
            # 添加优化器状态信息
            if hasattr(training_manager, 'lr_scheduler') and training_manager.lr_scheduler:
                metrics['current_learning_rate'] = training_manager.lr_scheduler.current_lr
            
            if hasattr(training_manager, 'exploration_scheduler') and training_manager.exploration_scheduler:
                exploration_params = training_manager.exploration_scheduler.get_current_params()
                metrics.update({
                    'exploration_epsilon': exploration_params.get('epsilon', 0),
                    'exploration_entropy': exploration_params.get('entropy_coef', 0)
                })

            if 'loss_info' in locals() and loss_info:
                metrics.update({
                    'policy_loss': loss_info.get('policy_loss', 0.0),
                    'value_loss': loss_info.get('value_loss', 0.0)
                })
            
            training_manager.log_training_progress(episode, episode_reward, metrics)
        
        # 每轮都更新进度条，保证终端只显示一行
        # 计算训练速度（episodes per second）
        current_time = time.time()
        elapsed = current_time - training_start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # 更新进度条描述，显示当前奖励和平均奖励
        progress_desc = f"🚀 MAPPO训练 | 当前: {episode_reward:.1f} | 平均: {current_avg_reward:.1f}"
        progress_bar.set_description(progress_desc)
        
        # 更新进度条，显示关键指标
        postfix_dict = {
            'Best': f'{best_reward:.1f}',
            'EPS': f'{eps_per_sec:.1f}/s'
        }
        
        # 只在有损失信息时添加损失指标
        if 'loss_info' in locals() and loss_info:
            postfix_dict['PLoss'] = f'{loss_info["policy_loss"]:.3f}'
            postfix_dict['VLoss'] = f'{loss_info["value_loss"]:.3f}'
        
        # 添加课程学习信息（如果有）
        if curriculum_manager:
            current_stage = curriculum_manager.current_stage
            postfix_dict['Stage'] = current_stage.name[:4]  # 缩短显示
        
        progress_bar.set_postfix(postfix_dict)
        progress_bar.refresh()
        
        # 课程学习已移除，无需检查阶段转换
        
        # 奖励调度器状态报告已移至模型保存时输出
        
        # 计算训练统计信息用于自适应奖励调度
        if episode > 0 and episode % 50 == 0:
            # 计算完成率
            completion_rate = sum(1 for r in episode_rewards[-50:] if r > 0) / 50
            
            # 计算效率（平均奖励）
            efficiency = np.mean(episode_rewards[-50:])
            
            # 更新自适应奖励权重
            performance_metrics = {
                'completion_rate': completion_rate,
                'efficiency': efficiency,
                'total_reward': efficiency  # 使用效率作为总奖励的代理
            }
            env.reward_scheduler.update_weights(performance_metrics)
        
        # 改进的早停机制：更宽松的条件，确保充分训练
        if episode > 5000:  # 至少训练5000轮
            # 计算最近100轮的平均奖励
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else current_avg_reward
            
            # 更宽松的早停条件：只有在性能严重下降且长时间无改善时才停止
            if (recent_avg < best_reward * 0.3 and  # 从0.5放宽到0.3
                episode - best_episode > 500 and   # 从200增加到500轮
                episode > 8000):                   # 从2000增加到8000轮
                print(f"🛑 长期性能下降，在第 {episode} 轮停止训练")
                print(f"   最近100轮平均奖励: {recent_avg:.2f}")
                print(f"   最佳奖励: {best_reward:.2f} (第{best_episode}轮)")
                print(f"   已连续 {episode - best_episode} 轮无改善")
                break
        
        # 设置简化的进度条描述
        if episode == 1:  # 只在第一轮设置一次
            progress_bar.set_description("🚀 训练进度 - 专家级配置")
        
        # 每50个episode进行验证集评估
        if episode % validation_frequency == 0 and episode > 0:
            current_validation_reward = validate_model(mappo, validation_env)
            validation_rewards.append(current_validation_reward)
            
            # 检测过拟合：验证集性能下降
            if len(validation_rewards) > 1:
                if current_validation_reward < validation_rewards[-2]:
                    validation_decline_count += 1
                else:
                    validation_decline_count = 0
                    
                # 如果验证集性能持续下降，提前停止
                if validation_decline_count >= validation_patience:
                    print(f"🛑 检测到过拟合，验证集性能连续{validation_patience}次下降，在第 {episode} 轮停止训练")
                    print(f"   当前验证奖励: {current_validation_reward:.2f}")
                    print(f"   最佳验证奖励: {max(validation_rewards):.2f}")
                    break
            
            print(f"📊 验证评估 (第{episode}轮): 验证奖励 = {current_validation_reward:.2f}, 训练奖励 = {current_avg_reward:.2f}")
        
        # 每50个episode进行详细报告
        if episode % 50 == 0 and episode > 0:
            # 获取环境状态信息
            env_state = env._get_current_state()
            
            # 计算服务完成率
            total_demand = sum(locker.get('demand_del', 0) + locker.get('demand_ret', 0) for locker in env.lockers_state)
            served_lockers = sum(1 for locker in env.lockers_state if locker.get('served', False))
            completion_rate = (served_lockers / len(env.lockers_state) * 100) if len(env.lockers_state) > 0 else 0
            
            # 计算平均卡车容量利用率
            total_capacity_used = 0
            total_capacity = 0
            for truck in env.trucks:
                current_load = truck.get('current_delivery_load', 0) + truck.get('current_return_load', 0)
                capacity = truck.get('capacity', 250)
                total_capacity_used += current_load
                total_capacity += capacity
            avg_capacity_utilization = (total_capacity_used / total_capacity * 100) if total_capacity > 0 else 0
            
            # 获取最后一步的动作信息（如果有的话）
            truck_decisions = ""
            if 'trajectory' in locals() and len(trajectory['actions']) > 0:
                last_actions = trajectory['actions'][-1]
                truck_decisions = "\n🚛 卡车决策:"
                for i, action in enumerate(last_actions):
                    if isinstance(action, dict):
                        stop_action = action.get('select_stop', 0)
                        if stop_action == 0:
                            action_desc = "ActionType.RETURN_TO_DEPOT -> None"
                        else:
                            action_desc = f"ActionType.MOVE_TO_LOCKER -> {stop_action}"
                    else:
                        action_desc = f"Action -> {action}"
                    truck_decisions += f"\n     卡车truck_{i}: {action_desc}"
            
            # 计算成本信息
            if 'trajectory' in locals() and len(trajectory['rewards']) > 0:
                step_reward = trajectory['rewards'][-1]
                avg_step_reward = sum(step_reward) / len(step_reward) if step_reward else 0
            else:
                avg_step_reward = 0
            
            # 使用tqdm.write输出详细报告
            detailed_report = (f"\n{'='*80}\n"
                             f"📊 Episode {episode} | 专家级配置训练\n"
                             f"🏆 平均奖励: {current_avg_reward:.2f} | 最佳奖励: {best_reward:.2f}")
            
            if len(trajectory['states']) > 0 and 'loss_info' in locals():
                detailed_report += f"\n📈 策略损失: {loss_info['policy_loss']:.4f} | 价值损失: {loss_info['value_loss']:.4f}"
            
            # 添加详细的决策和状态信息
            detailed_report += truck_decisions
            detailed_report += (f"\n📊 环境状态:"
                              f"\n     服务完成率: {completion_rate:.2f}%"
                              f"\n     平均卡车容量利用率: {avg_capacity_utilization:.2f}%")
            
            # 获取步数信息
            current_step_count = env.time_step if hasattr(env, 'time_step') else 0
            
            detailed_report += (f"\n💰 成本分析:"
                              f"\n     步骤奖励: {avg_step_reward:.2f}"
                              f"\n     总成本: {abs(avg_step_reward * current_step_count):.2f}")
            
            # 计算性能比率（基于奖励相对于最佳奖励的比例）
            performance_ratio = max(0.1, episode_reward / max(best_reward, 1.0)) if best_reward > 0 else 1.0
            
            # 计算真正的路径效率（基于步数）
            actual_path_efficiency = env._calculate_path_efficiency()
            
            detailed_report += (f"\n📈 第 {episode} 回合总结:"
                              f"\n    总步数: {current_step_count}"
                              f"\n    总奖励: {episode_reward:.2f}"
                              f"\n    完成率: {completion_rate:.2f}%"
                              f"\n    效率指标:"
                              f"\n       平均步骤奖励: {avg_step_reward:.2f}"
                              f"\n       路径效率: {(actual_path_efficiency * 100):.2f}%"
                              f"\n       资源利用率: {avg_capacity_utilization:.2f}%")
            
            detailed_report += f"\n{'='*80}"
            progress_bar.write(detailed_report)

    
    print("🎉 训练完成!")
    print(f"总训练轮数: {episode + 1}")
    print(f"最佳奖励: {best_reward:.2f}")
    print("最终环境配置: 专家级配置")
    
    print("\n📈 最终训练统计:")
    print(f"   训练轮数: {episode + 1}")
    print(f"   平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"   环境配置: 专家级 (4卡车, 15储物柜, 300无人机航程)")
    
    return mappo