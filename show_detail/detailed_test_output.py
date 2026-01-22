#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试输出管理器
作者: Dionysus
联系方式: wechat:gzw1546484791

功能:
- 记录测试过程中的详细操作
- 生成文本和JSON格式的测试报告
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class DetailedTestOutputManager:
    """
    详细测试输出管理器
    
    功能:
    - 记录卡车操作步骤
    - 记录无人机服务详情
    - 生成测试报告
    """
    
    def __init__(self, output_dir: str = "test_results"):
        """
        初始化输出管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储记录的数据
        self.truck_operations = []
        self.drone_services = []
        self.step_summaries = []
        
    def record_truck_step_operation(
        self,
        step: int,
        truck_id: int,
        action: Dict[str, Any],
        truck_state: Dict[str, Any],
        env_state: Dict[str, Any],
        reward: float,
        time_cost: float
    ):
        """
        记录卡车步骤操作
        
        Args:
            step: 步骤编号
            truck_id: 卡车ID
            action: 动作信息
            truck_state: 卡车状态
            env_state: 环境状态
            reward: 奖励值
            time_cost: 时间成本
        """
        operation = {
            'step': step,
            'truck_id': truck_id,
            'action': action,
            'truck_state': truck_state.copy() if isinstance(truck_state, dict) else truck_state,
            'env_state': env_state.copy() if isinstance(env_state, dict) else env_state,
            'reward': float(reward),
            'time_cost': float(time_cost),
            'timestamp': datetime.now().isoformat()
        }
        self.truck_operations.append(operation)
        
    def record_drone_service_details(
        self,
        step: int,
        truck_id: int,
        drone_services: List[Dict[str, Any]]
    ):
        """
        记录无人机服务详情
        
        Args:
            step: 步骤编号
            truck_id: 卡车ID
            drone_services: 无人机服务列表
        """
        service_record = {
            'step': step,
            'truck_id': truck_id,
            'services': drone_services.copy() if isinstance(drone_services, list) else drone_services,
            'timestamp': datetime.now().isoformat()
        }
        self.drone_services.append(service_record)
        
    def generate_text_report(self) -> str:
        """
        生成文本格式的测试报告
        
        Returns:
            文本报告字符串
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("详细测试报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 卡车操作统计
        report_lines.append("卡车操作统计:")
        report_lines.append("-" * 80)
        if self.truck_operations:
            total_steps = len(set(op['step'] for op in self.truck_operations))
            total_operations = len(self.truck_operations)
            total_reward = sum(op['reward'] for op in self.truck_operations)
            avg_reward = total_reward / total_operations if total_operations > 0 else 0
            
            report_lines.append(f"总步骤数: {total_steps}")
            report_lines.append(f"总操作数: {total_operations}")
            report_lines.append(f"总奖励: {total_reward:.2f}")
            report_lines.append(f"平均奖励: {avg_reward:.2f}")
        else:
            report_lines.append("无卡车操作记录")
        report_lines.append("")
        
        # 无人机服务统计
        report_lines.append("无人机服务统计:")
        report_lines.append("-" * 80)
        if self.drone_services:
            total_services = sum(len(sr['services']) for sr in self.drone_services)
            report_lines.append(f"总服务记录数: {len(self.drone_services)}")
            report_lines.append(f"总服务次数: {total_services}")
        else:
            report_lines.append("无无人机服务记录")
        report_lines.append("")
        
        # 详细操作记录（前10条）
        report_lines.append("详细操作记录（前10条）:")
        report_lines.append("-" * 80)
        for i, op in enumerate(self.truck_operations[:10]):
            report_lines.append(f"步骤 {op['step']}, 卡车 {op['truck_id']}:")
            report_lines.append(f"  动作: {op['action']}")
            report_lines.append(f"  奖励: {op['reward']:.2f}")
            report_lines.append(f"  时间成本: {op['time_cost']:.2f}")
            report_lines.append("")
        
        if len(self.truck_operations) > 10:
            report_lines.append(f"... 还有 {len(self.truck_operations) - 10} 条记录未显示")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        生成详细报告（JSON格式）
        
        Returns:
            详细报告字典
        """
        # 生成 truck_summaries 格式（用于动画生成器）
        truck_summaries = {}
        # 记录每个卡车上一步的位置，用于构建连续轨迹
        truck_last_positions = {}
        
        for op in self.truck_operations:
            truck_id = str(op['truck_id'])
            if truck_id not in truck_summaries:
                truck_summaries[truck_id] = {
                    'operations': []
                }
                # 初始化：卡车从仓库 (0, 0) 开始
                truck_last_positions[truck_id] = [0, 0]
            
            # 从 truck_state 中提取位置信息
            truck_state = op.get('truck_state', {})
            
            # 获取 action（确保在所有情况下都有值）
            action = op.get('action', {})
            if not isinstance(action, dict):
                action = {}
            else:
                # 处理 numpy 类型，转换为 Python 原生类型以便 JSON 序列化
                processed_action = {}
                for k, v in action.items():
                    # 处理 numpy 类型
                    if hasattr(v, 'item'):  # numpy scalar
                        processed_action[k] = v.item()
                    elif isinstance(v, (list, tuple)):
                        processed_action[k] = [int(x.item()) if hasattr(x, 'item') else int(x) for x in v]
                    elif isinstance(v, (np.integer, np.int64, np.int32)):
                        processed_action[k] = int(v)
                    else:
                        processed_action[k] = v
                action = processed_action
            
            # 当前位置：使用上一步的目标位置（构建连续轨迹）
            current_pos = truck_last_positions[truck_id].copy()
            
            # 目标位置：执行动作后的位置（position）
            target_pos = truck_state.get('position', [0, 0])
            if isinstance(target_pos, tuple):
                target_pos = list(target_pos)
            elif not isinstance(target_pos, list) or len(target_pos) < 2:
                # 如果 position 无效，尝试从 action 和 env_state 中获取
                select_stop = action.get('select_stop', 0)
                # 处理字符串类型的 select_stop
                if isinstance(select_stop, str):
                    try:
                        select_stop = int(select_stop)
                    except (ValueError, TypeError):
                        select_stop = 0
                elif isinstance(select_stop, (np.integer, np.int64)):
                    select_stop = int(select_stop)
                
                # 如果 select_stop 是 0，目标是仓库 (0, 0)
                # 否则需要从 env_state 中获取快递柜位置
                if select_stop > 0:
                    env_state = op.get('env_state', {})
                    lockers_state = env_state.get('lockers_state', [])
                    # select_stop 是从 1 开始的索引，需要减 1
                    locker_idx = select_stop - 1
                    if 0 <= locker_idx < len(lockers_state):
                        locker = lockers_state[locker_idx]
                        target_pos = locker.get('location', [0, 0])
                        if isinstance(target_pos, tuple):
                            target_pos = list(target_pos)
                        elif not isinstance(target_pos, list) or len(target_pos) < 2:
                            target_pos = [0, 0]
                else:
                    # select_stop 为 0，目标是仓库
                    target_pos = [0, 0]
            
            # 更新上一步位置为当前目标位置，用于下一步
            truck_last_positions[truck_id] = target_pos.copy()
            
            # 计算移动距离
            move_distance = 0.0
            if current_pos and target_pos:
                import math
                move_distance = math.sqrt(
                    (current_pos[0] - target_pos[0])**2 + 
                    (current_pos[1] - target_pos[1])**2
                )
            
            operation = {
                'step': op['step'],
                'current_position': current_pos,
                'target_position': target_pos,
                'move_distance': move_distance,
                'action': action,
                'reward': op['reward'],
                'time_cost': op['time_cost']
            }
            truck_summaries[truck_id]['operations'].append(operation)
        
        report = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_truck_operations': len(self.truck_operations),
                'total_drone_services': len(self.drone_services)
            },
            'truck_operations': self.truck_operations,  # 保留原始格式
            'truck_summaries': truck_summaries,  # 添加动画生成器需要的格式
            'drone_services': self.drone_services,
            'statistics': self._calculate_statistics()
        }
        
        return report
        
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        if self.truck_operations:
            rewards = [op['reward'] for op in self.truck_operations]
            time_costs = [op['time_cost'] for op in self.truck_operations]
            
            stats['truck_operations'] = {
                'total_count': len(self.truck_operations),
                'unique_steps': len(set(op['step'] for op in self.truck_operations)),
                'reward': {
                    'total': sum(rewards),
                    'average': sum(rewards) / len(rewards) if rewards else 0,
                    'min': min(rewards) if rewards else 0,
                    'max': max(rewards) if rewards else 0
                },
                'time_cost': {
                    'total': sum(time_costs),
                    'average': sum(time_costs) / len(time_costs) if time_costs else 0,
                    'min': min(time_costs) if time_costs else 0,
                    'max': max(time_costs) if time_costs else 0
                }
            }
        
        if self.drone_services:
            total_service_count = sum(len(sr['services']) for sr in self.drone_services)
            stats['drone_services'] = {
                'total_records': len(self.drone_services),
                'total_service_count': total_service_count,
                'average_per_record': total_service_count / len(self.drone_services) if self.drone_services else 0
            }
        
        return stats

