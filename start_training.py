#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPPO训练启动脚本
作者: Dionysus
联系方式: wechat:gzw1546484791

启动多智能体强化学习训练，验证智能体协同学习效果
集成训练稳定性优化功能，包括学习率调度、探索策略、奖励平滑和收敛检测
"""

import sys
import os

# Force CPU to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import json
import traceback
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from sklearn.linear_model import LinearRegression


class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，处理不可序列化的类型
    """
    def default(self, obj):
        if isinstance(obj, bool):
            return bool(obj)  # 确保布尔值正确序列化
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class AdaptiveLearningRateScheduler:
    """
    自适应学习率调度器
    根据训练进度和性能动态调整学习率
    """
    
    def __init__(self, initial_lr: float = 3e-4, min_lr: float = 1e-5, 
                 patience: int = 100, decay_factor: float = 0.8):
        """
        初始化学习率调度器
        
        Args:
            initial_lr: 初始学习率
            min_lr: 最小学习率
            patience: 性能停滞容忍轮数
            decay_factor: 衰减因子
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.decay_factor = decay_factor
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.lr_history = []
        
    def update(self, current_reward: float, episode: int) -> float:
        """
        更新学习率
        
        Args:
            current_reward: 当前奖励
            episode: 当前轮数
            
        Returns:
            更新后的学习率
        """
        # 记录历史
        self.lr_history.append(self.current_lr)
        
        # 检查是否有性能提升
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 如果性能停滞，降低学习率
        if self.patience_counter >= self.patience and self.current_lr > self.min_lr:
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            self.patience_counter = 0
            
        return self.current_lr


class ExplorationScheduler:
    """
    探索策略调度器
    动态调整探索参数以平衡探索与利用
    """
    
    def __init__(self, initial_epsilon: float = 0.3, min_epsilon: float = 0.05,
                 decay_rate: float = 0.995, entropy_coef_initial: float = 0.01):
        """
        初始化探索调度器
        
        Args:
            initial_epsilon: 初始探索率
            min_epsilon: 最小探索率
            decay_rate: 衰减率
            entropy_coef_initial: 初始熵系数
        """
        self.initial_epsilon = initial_epsilon
        self.current_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.entropy_coef = entropy_coef_initial
        self.exploration_history = []
        
    def update(self, episode: int, performance_variance: float) -> Dict[str, float]:
        """
        更新探索参数
        
        Args:
            episode: 当前轮数
            performance_variance: 性能方差
            
        Returns:
            更新后的探索参数
        """
        # 基于轮数的衰减
        self.current_epsilon = max(
            self.current_epsilon * self.decay_rate,
            self.min_epsilon
        )
        
        # 基于性能方差调整熵系数
        if performance_variance > 0.2:  # 高方差时增加探索
            self.entropy_coef = min(self.entropy_coef * 1.05, 0.05)
        else:  # 低方差时减少探索
            self.entropy_coef = max(self.entropy_coef * 0.98, 0.001)
        
        exploration_params = {
            'epsilon': self.current_epsilon,
            'entropy_coef': self.entropy_coef
        }
        
        self.exploration_history.append(exploration_params.copy())
        return exploration_params
    
    def get_current_params(self) -> Dict[str, float]:
        """
        获取当前探索参数
        
        Returns:
            当前探索参数字典
        """
        return {
            'epsilon': self.current_epsilon,
            'entropy_coef': self.entropy_coef
        }


class RewardSmoother:
    """
    奖励平滑器
    使用指数移动平均和异常值检测来平滑奖励信号
    """
    
    def __init__(self, alpha: float = 0.1, outlier_threshold: float = 2.0):
        """
        初始化奖励平滑器
        
        Args:
            alpha: 指数移动平均系数
            outlier_threshold: 异常值检测阈值（标准差倍数）
        """
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.ema_reward = None
        self.reward_history = []
        self.smoothed_history = []
        
    def smooth(self, reward: float) -> float:
        """
        平滑奖励值
        
        Args:
            reward: 原始奖励
            
        Returns:
            平滑后的奖励
        """
        self.reward_history.append(reward)
        
        # 异常值检测
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-10:]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            
            # 如果是异常值，使用均值替代
            if abs(reward - mean_reward) > self.outlier_threshold * std_reward:
                reward = mean_reward
        
        # 指数移动平均
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = self.alpha * reward + (1 - self.alpha) * self.ema_reward
        
        self.smoothed_history.append(self.ema_reward)
        return self.ema_reward


class ConvergenceDetector:
    """
    收敛检测器
    检测训练是否收敛或陷入局部最优
    """
    
    def __init__(self, window_size: int = 50, stability_threshold: float = 0.05,
                 improvement_threshold: float = 0.01):
        """
        初始化收敛检测器
        
        Args:
            window_size: 检测窗口大小（减少到50以更快检测收敛）
            stability_threshold: 稳定性阈值
            improvement_threshold: 改进阈值
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.improvement_threshold = improvement_threshold
        self.reward_history = []
        
    def check_convergence(self, reward: float) -> Dict[str, Any]:
        """
        检查收敛状态
        
        Args:
            reward: 当前奖励
            
        Returns:
            收敛状态信息
        """
        self.reward_history.append(reward)
        
        if len(self.reward_history) < self.window_size:
            return {
                'status': 'insufficient_data',
                'message': f'需要至少{self.window_size}轮数据',
                'confidence': 0.0
            }
        
        # 获取最近的奖励窗口
        recent_rewards = self.reward_history[-self.window_size:]
        
        # 计算统计指标
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        # 计算趋势
        if len(self.reward_history) >= 2 * self.window_size:
            early_rewards = self.reward_history[-2*self.window_size:-self.window_size]
            early_mean = np.mean(early_rewards)
            improvement_rate = (mean_reward - early_mean) / abs(early_mean) if early_mean != 0 else 0
        else:
            improvement_rate = 0
        
        # 判断收敛状态
        if cv < self.stability_threshold:
            if improvement_rate > self.improvement_threshold:
                status = 'converging_with_improvement'
                message = '训练收敛且性能持续改进'
                confidence = 0.9
            else:
                status = 'converged'
                message = '训练已收敛'
                confidence = 0.8
        elif improvement_rate < -self.improvement_threshold:
            status = 'degrading'
            message = '性能下降，可能需要调整参数'
            confidence = 0.7
        elif abs(improvement_rate) < self.improvement_threshold and len(self.reward_history) > 3 * self.window_size:
            status = 'local_optimum'
            message = '可能陷入局部最优'
            confidence = 0.6
        else:
            status = 'training'
            message = '训练进行中'
            confidence = 0.5
        
        return {
            'status': status,
            'message': message,
            'confidence': confidence,
            'statistics': {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'coefficient_of_variation': cv,
                'improvement_rate': improvement_rate
            }
        }


# 导入必要的模块
from truck_routing import TruckSchedulingEnv, train_marl
import config


class TrainingManager:
    """
    训练管理器
    管理MAPPO训练过程，包括监控、日志记录、性能评估和稳定性优化
    """
    
    def __init__(self, env: TruckSchedulingEnv, enable_optimization: bool = True):
        """
        初始化训练管理器
        
        Args:
            env: 训练环境
            enable_optimization: 是否启用训练稳定性优化
        """
        self.env = env
        self.start_time = time.time()
        self.training_log = []
        self.performance_metrics = {}
        self.enable_optimization = enable_optimization
        
        # 初始化稳定性优化组件
        if self.enable_optimization:
            self.lr_scheduler = AdaptiveLearningRateScheduler(
                initial_lr=config.LEARNING_RATE,
                min_lr=config.LEARNING_RATE * 0.1,
                patience=100,
                decay_factor=0.8
            )
            self.exploration_scheduler = ExplorationScheduler(
                initial_epsilon=0.3,
                min_epsilon=0.05,
                decay_rate=0.995
            )
            self.reward_smoother = RewardSmoother(alpha=0.1, outlier_threshold=2.0)
            self.convergence_detector = ConvergenceDetector(
                window_size=100,
                stability_threshold=0.05,
                improvement_threshold=0.01
            )
            print("✅ 训练稳定性优化已启用")
        
    def log_training_progress(self, episode: int, reward: float, metrics: Dict[str, Any]):
        """
        记录训练进度并应用稳定性优化
        
        Args:
            episode: 训练轮数
            reward: 奖励值
            metrics: 性能指标
        """
        # 应用奖励平滑
        if self.enable_optimization:
            smoothed_reward = self.reward_smoother.smooth(reward)
            
            # 更新学习率
            new_lr = self.lr_scheduler.update(smoothed_reward, episode)
            
            # 计算性能方差用于探索调度
            recent_rewards = [entry['reward'] for entry in self.training_log[-20:]]
            if len(recent_rewards) > 5:
                performance_variance = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)
            else:
                performance_variance = 0.2
            
            # 更新探索参数
            exploration_params = self.exploration_scheduler.update(episode, performance_variance)
            
            # 检查收敛状态
            convergence_info = self.convergence_detector.check_convergence(smoothed_reward)
            
            # 更新metrics
            metrics.update({
                'smoothed_reward': smoothed_reward,
                'learning_rate': new_lr,
                'exploration_epsilon': exploration_params['epsilon'],
                'entropy_coefficient': exploration_params['entropy_coef'],
                'convergence_status': convergence_info['status'],
                'convergence_confidence': convergence_info['confidence']
            })
        
        log_entry = {
            'episode': episode,
            'reward': reward,
            'timestamp': time.time() - self.start_time,
            'metrics': metrics
        }
        self.training_log.append(log_entry)
        
        # 进度信息由tqdm进度条显示，无需额外print输出
    
    def get_current_training_params(self) -> Dict[str, Any]:
        """
        获取当前训练参数（用于动态调整）
        
        Returns:
            当前训练参数
        """
        if not self.enable_optimization:
            return {}
        
        return {
            'learning_rate': self.lr_scheduler.current_lr,
            'epsilon': self.exploration_scheduler.current_epsilon,
            'entropy_coef': self.exploration_scheduler.entropy_coef
        }
    
    def evaluate_training_performance(self) -> Dict[str, Any]:
        """
        评估训练性能
        
        Returns:
            性能评估结果
        """
        if not self.training_log:
            return {"status": "no_data", "message": "没有训练数据"}
        
        # 计算性能指标
        rewards = [entry['reward'] for entry in self.training_log]
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        
        performance = {
            'total_episodes': len(self.training_log),
            'training_time': time.time() - self.start_time,
            'average_reward': np.mean(rewards),
            'recent_average_reward': np.mean(recent_rewards),
            'best_reward': max(rewards),
            'reward_improvement': recent_rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
            'convergence_status': self._check_convergence(rewards)
        }
        
        # 添加稳定性优化相关指标
        if self.enable_optimization and len(self.training_log) > 0:
            latest_metrics = self.training_log[-1]['metrics']
            performance.update({
                'final_learning_rate': latest_metrics.get('learning_rate', config.LEARNING_RATE),
                'final_exploration_rate': latest_metrics.get('exploration_epsilon', 0.1),
                'reward_stability': np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8),
                'optimization_enabled': True
            })
        
        return performance
    
    def _check_convergence(self, rewards: List[float]) -> str:
        """
        检查训练收敛状态
        
        Args:
            rewards: 奖励历史
            
        Returns:
            收敛状态描述
        """
        if self.enable_optimization and hasattr(self, 'convergence_detector'):
            if len(rewards) > 0:
                convergence_info = self.convergence_detector.check_convergence(rewards[-1])
                return convergence_info['message']
        
        # 回退到原始检查逻辑
        if len(rewards) < 100:
            return "训练数据不足"
        
        # 检查最近100轮的奖励稳定性
        recent_rewards = rewards[-100:]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        if reward_std / abs(reward_mean) < 0.1:  # 变异系数小于10%
            return "已收敛"
        elif len(rewards) >= 500:
            # 检查趋势
            early_avg = np.mean(rewards[:100])
            late_avg = np.mean(rewards[-100:])
            if late_avg > early_avg * 1.1:
                return "持续改进中"
            else:
                return "可能陷入局部最优"
        else:
            return "训练中"
    
    def save_training_report(self, filename: str = "training_report.json"):
        """
        保存训练报告
        
        Args:
            filename: 报告文件名
        """
        performance = self.evaluate_training_performance()
        
        report = {
            'training_summary': performance,
            'environment_config': {
                'num_trucks': self.env.num_trucks,
                'num_lockers': self.env.num_lockers,
                'truck_capacity': self.env.truck_capacity,
                'state_dimension': self.env.state_dim
            },
            'training_config': {
                'total_timesteps': config.TOTAL_TIMESTEPS,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'gamma': config.GAMMA,
                'optimization_enabled': self.enable_optimization
            },
            'training_log': self.training_log,  # 保存完整的训练日志
            'recommendations': self._generate_recommendations(performance)
        }
        
        # 添加优化历史数据
        if self.enable_optimization:
            report['optimization_history'] = {
                'learning_rate_history': getattr(self.lr_scheduler, 'lr_history', []),
                'exploration_history': getattr(self.exploration_scheduler, 'exploration_history', []),
                'smoothed_rewards': getattr(self.reward_smoother, 'smoothed_history', [])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        print(f"📄 训练报告已保存到: {filename}")
    
    def _generate_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """
        根据性能生成建议
        
        Args:
            performance: 性能评估结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 安全获取收敛状态
        convergence_status = performance.get('convergence_status', '未知')
        if convergence_status == "已收敛":
            recommendations.append("训练已收敛，可以进行模型部署测试")
        elif convergence_status == "持续改进中":
            recommendations.append("训练效果良好，建议继续训练以获得更好性能")
        elif convergence_status == "可能陷入局部最优":
            if self.enable_optimization:
                recommendations.append("已启用自适应优化，系统将自动调整学习率和探索策略")
            else:
                recommendations.append("建议启用训练稳定性优化或手动调整学习率")
        else:
            recommendations.append("训练数据不足，建议增加训练轮数")
        
        # 安全获取奖励数据
        recent_avg = performance.get('recent_average_reward', 0)
        avg_reward = performance.get('average_reward', 0)
        
        if recent_avg > avg_reward * 1.1:
            recommendations.append("近期表现优秀，训练效果良好")
        elif recent_avg < avg_reward * 0.9:
            recommendations.append("近期表现下降，建议检查训练参数")
        
        # 稳定性相关建议
        if self.enable_optimization:
            reward_stability = performance.get('reward_stability', 0)
            if reward_stability > 0.3:
                recommendations.append("奖励波动较大，已启用奖励平滑机制")
            elif reward_stability < 0.1:
                recommendations.append("奖励稳定性良好，训练收敛效果佳")
        
        return recommendations

    def generate_training_plots(self, save_dir: str = "."):
        """
        生成训练过程的可视化图表
        
        Args:
            save_dir: 图表保存目录
        """
        if not self.training_log:
            print("⚠️ 没有训练数据，无法生成图表")
            return
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 提取数据
        episodes = [entry['episode'] for entry in self.training_log]
        rewards = [entry['reward'] for entry in self.training_log]
        timestamps = [entry['timestamp'] for entry in self.training_log]
        
        # 设置英文字体，避免字体显示问题
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建单个图表，只显示平均奖励曲线
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Average Reward Convergence Curve', fontsize=16, fontweight='bold')
        
        # 裁剪掉最后50个episode的数据
        if len(rewards) > 50:
            trimmed_rewards = rewards[:-50]
            trimmed_episodes = episodes[:-50]
        else:
            trimmed_rewards = rewards
            trimmed_episodes = episodes
        
        # 计算移动平均奖励
        if len(trimmed_rewards) > 30:
            window_size = 30  # 使用30个episode的移动平均
            moving_avg = np.convolve(trimmed_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = trimmed_episodes[window_size-1:]
            
            # 减去最低值进行归一化
            min_avg_reward = np.min(moving_avg)
            normalized_avg = moving_avg - min_avg_reward
            
            # 绘制归一化后的平均奖励曲线
            ax.plot(moving_episodes, normalized_avg, 'b-', linewidth=3, label=f'Average Reward (30-episode window)')
            
            # 计算并绘制趋势线
            if len(moving_episodes) > 10:  # 确保有足够的数据点
                # 使用线性回归计算趋势线
                X = np.array(moving_episodes).reshape(-1, 1)
                y = np.array(normalized_avg)
                
                lr_model = LinearRegression()
                lr_model.fit(X, y)
                
                # 计算趋势线的预测值
                trend_line = lr_model.predict(X)
                
                # 绘制趋势线
                ax.plot(moving_episodes, trend_line, 'r--', linewidth=2, alpha=0.8, 
                       label=f'Trend Line (slope: {lr_model.coef_[0]:.3f})')
                
                # 计算趋势强度 (R²)
                from sklearn.metrics import r2_score
                r2 = r2_score(y, trend_line)
                
                # 在图上显示趋势信息
                trend_direction = "Up" if lr_model.coef_[0] > 0 else "Down" if lr_model.coef_[0] < 0 else "Flat"
                ax.text(0.02, 0.85, f'Trend: {trend_direction}\nSlope: {lr_model.coef_[0]:.4f}\nR^2: {r2:.3f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax.set_xlabel('Training Episodes', fontsize=12)
            ax.set_ylabel('Normalized Average Reward (minus minimum)', fontsize=12)
            ax.set_title(f'Average Reward Curve with Trend (Min: {min_avg_reward:.2f}, Max: {np.max(moving_avg):.2f})', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # 添加统计信息
            trimmed_info = f" (trimmed last 50)" if len(rewards) > 50 else ""
            ax.text(0.02, 0.98, f'Episodes: {len(trimmed_episodes)}{trimmed_info}\nMin Avg: {min_avg_reward:.2f}\nMax Avg: {np.max(moving_avg):.2f}\nRange: {np.max(moving_avg) - min_avg_reward:.2f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\n(Need >30 episodes for average)', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Average Reward Curve')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(save_dir, 'training_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练分析图表已保存到: {plot_path}")
        
        # 生成损失曲线图（如果有损失数据）
        self._generate_loss_plots(save_dir)
        
        # 生成奖励分解曲线图
        self._generate_component_plots(save_dir)

        # 生成学习过程监控图（学习率、探索率、熵）
        self._generate_learning_process_plots(save_dir)

    def _generate_learning_process_plots(self, save_dir: str):
        """
        生成学习过程监控图（学习率、探索率、熵）
        """
        # 提取数据
        episodes = []
        learning_rates = []
        exploration_epsilons = []
        entropies = []
        
        for entry in self.training_log:
            if 'metrics' in entry and entry['metrics']:
                metrics = entry['metrics']
                # 收集存在的指标
                if 'current_learning_rate' in metrics:
                    episodes.append(entry['episode'])
                    learning_rates.append(metrics['current_learning_rate'])
                    exploration_epsilons.append(metrics.get('exploration_epsilon', 0))
                    entropies.append(metrics.get('exploration_entropy', 0))
        
        if not episodes:
            return

        # 创建图表 - 3个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('Learning Process Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning Rate
        ax1.plot(episodes, learning_rates, 'b-', linewidth=2)
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Schedule')
        ax1.grid(True, alpha=0.3)
        
        # 2. Exploration Rate (Epsilon)
        ax2.plot(episodes, exploration_epsilons, 'g-', linewidth=2)
        ax2.set_ylabel('Exploration Rate (Epsilon)')
        ax2.set_title('Exploration Decay')
        ax2.grid(True, alpha=0.3)
        
        # 3. Policy Entropy
        ax3.plot(episodes, entropies, 'r-', linewidth=2)
        ax3.set_ylabel('Policy Entropy')
        ax3.set_title('Policy Entropy (Randomness)')
        ax3.set_xlabel('Episode')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'learning_process.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"📊 学习过程监控图表已保存到: {plot_path}")

    
    def _generate_component_plots(self, save_dir: str):
        """
        生成奖励分解曲线图
        """
        # 提取数据
        episodes = []
        service_rewards = []
        efficiency_rewards = []
        cost_penalties = []
        
        for entry in self.training_log:
            if 'metrics' in entry and entry['metrics']:
                metrics = entry['metrics']
                # 检查是否有分解奖励数据
                if 'reward_service' in metrics:
                    episodes.append(entry['episode'])
                    service_rewards.append(metrics['reward_service'])
                    efficiency_rewards.append(metrics.get('reward_efficiency', 0))
                    cost_penalties.append(metrics.get('reward_cost', 0))
        
        if not episodes:
            return

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('Reward Components Analysis', fontsize=16, fontweight='bold')
        
        window_size = min(50, len(episodes) // 10) if len(episodes) > 50 else 1
        
        # 1. Service Reward
        ax1.plot(episodes, service_rewards, 'g-', alpha=0.3, label='Raw')
        if len(episodes) > window_size:
            ma = np.convolve(service_rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(episodes[window_size-1:], ma, 'g-', linewidth=2, label=f'MA({window_size})')
        ax1.set_ylabel('Service Reward')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Efficiency Reward
        ax2.plot(episodes, efficiency_rewards, 'b-', alpha=0.3, label='Raw')
        if len(episodes) > window_size:
            ma = np.convolve(efficiency_rewards, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(episodes[window_size-1:], ma, 'b-', linewidth=2, label=f'MA({window_size})')
        ax2.set_ylabel('Efficiency Reward')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cost Penalty
        ax3.plot(episodes, cost_penalties, 'r-', alpha=0.3, label='Raw')
        if len(episodes) > window_size:
            ma = np.convolve(cost_penalties, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(episodes[window_size-1:], ma, 'r-', linewidth=2, label=f'MA({window_size})')
        ax3.set_ylabel('Cost Penalty')
        ax3.set_xlabel('Episode')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'reward_components.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"📊 奖励分解图表已保存到: {plot_path}")

    def _generate_loss_plots(self, save_dir: str):
        """
        生成损失函数曲线图
        
        Args:
            save_dir: 图表保存目录
        """
        # 提取损失数据
        policy_losses = []
        value_losses = []
        episodes_with_loss = []
        
        for entry in self.training_log:
            if 'metrics' in entry and entry['metrics']:
                metrics = entry['metrics']
                if 'policy_loss' in metrics and 'value_loss' in metrics:
                    policy_losses.append(metrics['policy_loss'])
                    value_losses.append(metrics['value_loss'])
                    episodes_with_loss.append(entry['episode'])
        
        if not policy_losses:
            print("⚠️ 没有损失数据，跳过损失曲线图生成")
            return
        
        # 创建损失曲线图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('MAPPO Loss Function Analysis', fontsize=14, fontweight='bold')
        
        # 策略损失
        ax1.plot(episodes_with_loss, policy_losses, 'b-', linewidth=1, alpha=0.7)
        if len(policy_losses) > 10:
            window_size = min(20, len(policy_losses) // 5)
            moving_avg = np.convolve(policy_losses, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = episodes_with_loss[window_size-1:]
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average({window_size})')
            ax1.legend()
        
        ax1.set_xlabel('Training Episodes')
        ax1.set_ylabel('Policy Loss')
        ax1.set_title('Policy Loss Curve')
        ax1.grid(True, alpha=0.3)
        
        # 价值损失
        ax2.plot(episodes_with_loss, value_losses, 'g-', linewidth=1, alpha=0.7)
        if len(value_losses) > 10:
            window_size = min(20, len(value_losses) // 5)
            moving_avg = np.convolve(value_losses, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = episodes_with_loss[window_size-1:]
            ax2.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average({window_size})')
            ax2.legend()
        
        ax2.set_xlabel('Training Episodes')
        ax2.set_ylabel('Value Loss')
        ax2.set_title('Value Loss Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存损失图表
        loss_plot_path = os.path.join(save_dir, 'loss_analysis.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 损失分析图表已保存到: {loss_plot_path}")


def run_training_session(num_episodes: int = 1000, enable_optimization: bool = True) -> Dict[str, Any]:
    """
    运行训练会话
    
    Args:
        num_episodes: 训练轮数
        enable_optimization: 是否启用训练稳定性优化
        
    Returns:
        训练结果
    """
    print("🚀 开始MAPPO多智能体强化学习训练")
    print("=" * 60)
    
    try:
        # 创建环境
        print("🔧 初始化训练环境...")
        env = TruckSchedulingEnv(verbose=True)
        
        # 设置环境配置（使用config中的值）
        env_config = {
            'num_lockers': config.num_lockers,  # 快递柜数量（从config读取）
            'num_trucks': None,  # 卡车数量（动态计算）
            'boundary': config.boundary,    # 边界范围（从config读取）
            'demand_variance': config.demand_variance,  # 需求方差（从config读取）
            'time_pressure': config.time_pressure     # 时间压力（从config读取）
        }
        
        # 应用环境配置
        if hasattr(env, 'update_curriculum_config'):
            env.update_curriculum_config(env_config)
        
        # 创建训练管理器（启用优化功能）
        training_manager = TrainingManager(env, enable_optimization=enable_optimization)
        
        print(f"📊 环境配置:")
        print(f"   - 卡车数量: {env.num_trucks}")
        print(f"   - 快递柜数量: {env.num_lockers}")
        print(f"   - 边界范围: ±{env_config['boundary']}")
        print(f"   - 卡车容量: {env.truck_capacity}")
        print(f"   - 状态维度: {env.state_dim}")
        print(f"   - 无人机航程: {config.DRONE_MAX_RANGE} (单程{config.DRONE_MAX_RANGE//2})")
        
        print(f"\n🎯 训练配置:")
        print(f"   - 训练轮数: {num_episodes}")
        print(f"   - 初始学习率: {config.LEARNING_RATE}")
        print(f"   - 批次大小: {config.BATCH_SIZE}")
        print(f"   - 折扣因子: {config.GAMMA}")
        print(f"   - 稳定性优化: {'✅ 已启用' if enable_optimization else '❌ 未启用'}")
        
        # 开始训练
        print(f"\n🏃 开始训练...")
        trained_policy = train_marl(env, num_episodes=num_episodes, training_manager=training_manager, curriculum_manager=None)
        
        # 保存模型
        model_path = "trained_mappo_policy.pth"
        if hasattr(trained_policy, 'policy_net') and hasattr(trained_policy, 'value_net'):
            # 保存策略网络和价值网络的状态字典
            model_state = {
                'policy_net_state_dict': trained_policy.policy_net.state_dict(),
                'value_net_state_dict': trained_policy.value_net.state_dict(),
                'num_trucks': trained_policy.num_trucks,
                'policy_optimizer_state_dict': trained_policy.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': trained_policy.value_optimizer.state_dict(),
                'best_performance': trained_policy.best_performance,
                'training_metadata': {
                    'clip_ratio': trained_policy.clip_ratio,
                    'entropy_coef': trained_policy.entropy_coef,
                    'value_loss_coef': trained_policy.value_loss_coef,
                    'max_grad_norm': trained_policy.max_grad_norm
                }
            }
            torch.save(model_state, model_path)
            print(f"💾 模型已保存到: {model_path}")
        else:
            print("⚠️ 无法保存模型：MAPPO对象缺少必要属性")
            print(f"   可用属性: {[attr for attr in dir(trained_policy) if not attr.startswith('_')]}")
        
        # 评估性能
        performance = training_manager.evaluate_training_performance()
        
        # 生成训练图表
        print("\n📊 生成训练分析图表...")
        training_manager.generate_training_plots()
        
        # 保存训练报告
        training_manager.save_training_report()
        
        print("\n✅ 训练完成!")
        print(f"📈 训练性能总结:")
        print(f"   - 总轮数: {performance.get('total_episodes', 0)}")
        print(f"   - 训练时间: {performance.get('training_time', 0):.1f}秒")
        print(f"   - 平均奖励: {performance.get('average_reward', 0):.2f}")
        print(f"   - 最佳奖励: {performance.get('best_reward', 0):.2f}")
        print(f"   - 收敛状态: {performance.get('convergence_status', '未知')}")
        
        # 显示优化相关信息
        if enable_optimization and performance.get('optimization_enabled', False):
            print(f"\n🔧 训练优化总结:")
            print(f"   - 最终学习率: {performance.get('final_learning_rate', config.LEARNING_RATE):.2e}")
            print(f"   - 最终探索率: {performance.get('final_exploration_rate', 0.1):.3f}")
            print(f"   - 奖励稳定性: {performance.get('reward_stability', 0):.3f}")
            
            # 显示优化效果
            if performance.get('reward_stability', 1) < 0.2:
                print(f"   - 优化效果: ✅ 奖励稳定性良好")
            elif performance.get('reward_stability', 1) < 0.4:
                print(f"   - 优化效果: ⚠️ 奖励波动适中")
            else:
                print(f"   - 优化效果: ❌ 奖励波动较大")
        
        return {
            'status': 'success',
            'performance': performance,
            'model_path': model_path,
            'trained_policy': trained_policy,
            'optimization_enabled': enable_optimization
        }
        
    except Exception as e:
        error_msg = f"训练过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error_message': error_msg,
            'traceback': traceback.format_exc()
        }


def quick_training_test() -> bool:
    """
    快速训练测试
    运行少量轮数验证训练流程是否正常
    
    Returns:
        测试是否成功
    """
    print("🧪 开始快速训练测试...")
    
    try:
        # 运行1500轮训练测试，足够验证所有功能
        result = run_training_session(num_episodes=1500)  # 快速测试所有功能
        
        if result['status'] == 'success':
            print("✅ 快速训练测试通过")
            return True
        else:
            print(f"❌ 快速训练测试失败: {result.get('error_message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 快速训练测试异常: {str(e)}")
        return False


def main():
    """主函数"""
    try:
        print("🎯 开始智能卡车路径规划训练")
        print("=" * 50)
        
        # 快速测试
        print("🧪 执行快速功能测试...")
        if not quick_training_test():
            print("❌ 快速测试失败，请检查环境配置")
            return False
        
        print("✅ 快速测试通过")
        print("\n" + "=" * 50)
        
        # 正式训练 - 使用原始环境配置
        print("🚀 开始原始环境配置训练...")
        results = run_training_session(num_episodes=1000)
        
        if results['status'] == 'success':
            print("🎉 原始环境配置训练成功完成!")
            performance = results.get('performance', {})
            print(f"📈 最终性能: {performance.get('recent_average_reward', 0):.4f}")
            print(f"📊 收敛状态: {performance.get('convergence_status', '未知')}")
            return True
        else:
            print("❌ 训练失败")
            return False
            
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {str(e)}")
        traceback.print_exc()
        return False


def generate_plots_from_report(report_path: str, save_dir: str = "."):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    training_log = report.get("training_log", [])
    if not training_log:
        print("⚠️ 报告中没有训练日志，无法生成图表")
        return
    env = TruckSchedulingEnv(verbose=False)
    optimization_enabled = True
    cfg = report.get("training_config", {})
    if isinstance(cfg, dict) and "optimization_enabled" in cfg:
        optimization_enabled = cfg.get("optimization_enabled", True)
    tm = TrainingManager(env, enable_optimization=optimization_enabled)
    tm.training_log = training_log
    tm.generate_training_plots(save_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plots-from-report":
        rp = sys.argv[2] if len(sys.argv) > 2 else "training_report.json"
        sd = sys.argv[3] if len(sys.argv) > 3 else "."
        generate_plots_from_report(rp, sd)
    else:
        result = run_training_session(num_episodes=15000, enable_optimization=True)

        if result['status'] == 'success':
            performance = result['performance']
            print("\n✅ 训练完成！")
            print(f"📊 最终奖励: {performance.get('best_reward', 0):.2f}")
            print(f"⏱️ 训练时间: {performance.get('training_time', 0):.2f}秒")
            print(f"💾 模型保存路径: {result['model_path']}")
            
            # 显示优化状态
            if result.get('optimization_enabled', False):
                print(f"🔧 训练优化: ✅ 已启用")
            
            sys.exit(0)
        else:
            print("\n❌ 训练失败！")
            print(f"错误信息: {result.get('error_message', '未知错误')}")
            sys.exit(1)
