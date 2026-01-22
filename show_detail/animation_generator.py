#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动画生成器 - 展示卡车和无人机的移动情况
作者: Dionysus
联系方式: wechat:gzw1546484791
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
import os
from datetime import datetime
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class MovementAnimationGenerator:
    """
    卡车和无人机移动动画生成器
    
    功能：
    - 读取测试结果数据
    - 生成卡车移动轨迹动画
    - 生成无人机飞行路径动画
    - 保存为GIF动图
    """
    
    def __init__(self, test_results_dir):
        """
        初始化动画生成器
        
        输入:
            test_results_dir (str): 测试结果目录路径
        """
        self.test_results_dir = test_results_dir
        self.truck_data = {}
        self.locker_positions = {}
        self.drone_data = {}
        self.animation_frames = []
        
        # 动画配置
        self.fig_size = (15, 10)
        self.truck_colors = ['red', 'blue', 'green']
        self.drone_colors = ['orange', 'purple', 'brown']
        self.locker_color = 'gray'
        
    def load_data(self):
        """
        加载测试结果数据
        
        输出:
            bool: 数据加载是否成功
        """
        try:
            # 加载详细测试数据
            with open(os.path.join(self.test_results_dir, 'detailed_test_data.json'), 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
            
            # 加载环境初始化数据
            with open(os.path.join(self.test_results_dir, 'environment_initialization.json'), 'r', encoding='utf-8') as f:
                env_data = json.load(f)
            
            # 提取快递柜位置
            for locker in env_data['locker_positions']:
                self.locker_positions[locker['locker_id']] = locker['position']
            
            # 提取卡车移动数据
            self._extract_truck_data(detailed_data)
            
            # 提取无人机数据
            self._extract_drone_data()
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def _extract_truck_data(self, detailed_data):
        """
        提取卡车移动数据
        
        输入:
            detailed_data (dict): 详细测试数据
        """
        truck_summaries = detailed_data.get('truck_summaries', {})
        
        for truck_id, truck_info in truck_summaries.items():
            truck_id = int(truck_id)
            self.truck_data[truck_id] = {
                'positions': [],
                'targets': [],
                'steps': [],
                'distances': []
            }
            
            operations = truck_info.get('operations', [])
            # 构建完整的轨迹序列
            # 轨迹应该是：起始位置 -> 第一个目标 -> 第二个目标 -> ...
            trajectory_positions = []
            
            for i, op in enumerate(operations):
                step = op.get('step', 0)
                current_pos = op.get('current_position', [0, 0])
                target_pos = op.get('target_position', [0, 0])
                distance = op.get('move_distance', 0)
                
                # 确保位置是列表格式，且有两个元素
                if not isinstance(current_pos, list) or len(current_pos) < 2:
                    current_pos = [0, 0]
                if not isinstance(target_pos, list) or len(target_pos) < 2:
                    target_pos = [0, 0]
                
                # 第一步：添加起始位置（current_position）和目标位置
                if i == 0:
                    trajectory_positions.append(current_pos)  # 起始位置
                
                # 添加目标位置（这是卡车移动到的位置）
                trajectory_positions.append(target_pos)
                
                # 保存步骤信息
                self.truck_data[truck_id]['steps'].append(step)
                self.truck_data[truck_id]['distances'].append(distance)
                self.truck_data[truck_id]['targets'].append(target_pos)
            
            # 将构建的轨迹位置序列保存
            self.truck_data[truck_id]['positions'] = trajectory_positions
    
    def _extract_drone_data(self):
        """
        从测试报告中提取无人机数据
        """
        try:
            report_path = os.path.join(self.test_results_dir, 'detailed_test_report.txt')
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析无人机信息
            lines = content.split('\n')
            current_step = 0
            current_truck = 0
            
            for line in lines:
                line = line.strip()
                if '步骤' in line and '卡车' in line and '无人机' in line:
                    # 解析步骤、卡车和无人机信息
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        step_info = parts[0].replace('步骤 ', '')
                        truck_info = parts[1].replace('卡车 ', '')
                        drone_info = parts[2].replace('无人机 ', '').replace(':', '')
                        
                        try:
                            step = int(step_info)
                            truck_id = int(truck_info)
                            drone_id = int(drone_info)
                            
                            if step not in self.drone_data:
                                self.drone_data[step] = {}
                            if truck_id not in self.drone_data[step]:
                                self.drone_data[step][truck_id] = []
                                
                        except ValueError:
                            continue
                            
                elif '目标快递柜:' in line:
                    target_locker = line.split(':')[1].strip()
                elif '飞行距离:' in line:
                    flight_distance = float(line.split(':')[1].strip())
                elif '总时间:' in line:
                    total_time = float(line.split(':')[1].strip())
                    
                    # 保存无人机数据
                    if (step in self.drone_data and 
                        truck_id in self.drone_data[step]):
                        
                        drone_info = {
                            'drone_id': drone_id,
                            'target_locker': int(target_locker),
                            'flight_distance': flight_distance,
                            'total_time': total_time
                        }
                        self.drone_data[step][truck_id].append(drone_info)
                        
        except Exception as e:
            print(f"无人机数据提取失败: {e}")
    
    def create_animation(self, save_path=None, interval=1000):
        """
        创建移动动画
        
        输入:
            save_path (str): 保存路径，默认为None
            interval (int): 帧间隔（毫秒）
            
        输出:
            matplotlib.animation.FuncAnimation: 动画对象
        """
        # 检查是否有数据
        if not self.truck_data:
            raise ValueError("没有卡车数据，无法生成动画。请先运行测试生成数据。")
        
        # 检查是否有有效的步骤数据
        all_steps = []
        for truck_info in self.truck_data.values():
            if truck_info.get('steps'):
                all_steps.extend(truck_info['steps'])
        
        if not all_steps:
            raise ValueError("没有有效的步骤数据，无法生成动画。")
        
        # 设置图形
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 计算坐标范围
        all_positions = []
        for locker_pos in self.locker_positions.values():
            all_positions.append(locker_pos)
        
        for truck_id, truck_info in self.truck_data.items():
            if truck_info.get('positions'):
                all_positions.extend(truck_info['positions'])
            if truck_info.get('targets'):
                all_positions.extend(truck_info['targets'])
        
        if all_positions:
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 添加边距
            margin = 20
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            # 如果没有位置数据，设置默认范围
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
        
        # 绘制快递柜
        for locker_id, pos in self.locker_positions.items():
            circle = Circle(pos, 3, color=self.locker_color, alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(locker_id), ha='center', va='center', fontsize=8)
        
        # 初始化卡车和轨迹
        truck_points = {}
        truck_trails = {}
        truck_texts = {}
        
        for truck_id in self.truck_data.keys():
            color = self.truck_colors[truck_id % len(self.truck_colors)]
            truck_points[truck_id], = ax.plot([], [], 'o', color=color, markersize=10, label=f'Truck {truck_id}')
            truck_trails[truck_id], = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=2)
            truck_texts[truck_id] = ax.text(0, 0, f'T{truck_id}', ha='center', va='center', fontweight='bold')
        
        # 初始化无人机
        drone_points = {}
        drone_lines = {}
        
        ax.set_title('Truck and Drone Movement Animation', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        if truck_points:  # 只有当有卡车数据时才显示图例
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 获取最大步数和构建步骤到位置的映射
        max_step = 0
        step_to_pos_map = {}  # {truck_id: {step: position_index}}
        
        for truck_id, truck_info in self.truck_data.items():
            steps = truck_info.get('steps', [])
            positions = truck_info.get('positions', [])
            if steps:
                max_step = max(max_step, max(steps))
                # 构建步骤到位置索引的映射
                step_map = {}
                for i, step in enumerate(steps):
                    # 每个步骤对应一个位置（目标位置）
                    # 位置索引 = i + 1（因为第一个位置是起始位置）
                    if i + 1 < len(positions):
                        step_map[step] = i + 1
                step_to_pos_map[truck_id] = step_map
        
        if max_step == 0:
            max_steps = 1
        else:
            max_steps = max_step + 1
        
        def animate(frame):
            """
            动画帧更新函数
            
            输入:
                frame (int): 当前帧数（对应步骤）
            """
            current_step = frame
            
            # 更新卡车位置和轨迹
            for truck_id, truck_info in self.truck_data.items():
                positions = truck_info.get('positions', [])
                steps = truck_info.get('steps', [])
                
                if not positions:
                    continue
                
                # 找到当前步骤对应的位置索引
                pos_index = 0  # 默认显示起始位置
                if current_step in step_to_pos_map.get(truck_id, {}):
                    pos_index = step_to_pos_map[truck_id][current_step]
                elif current_step > 0 and steps:
                    # 如果当前步骤不在映射中，使用最后一个位置
                    pos_index = len(positions) - 1
                
                # 确保索引有效
                if pos_index >= len(positions):
                    pos_index = len(positions) - 1
                
                pos = positions[pos_index]
                
                # 更新卡车位置
                truck_points[truck_id].set_data([pos[0]], [pos[1]])
                truck_texts[truck_id].set_position((pos[0], pos[1]))
                
                # 更新轨迹：显示到当前位置的所有路径点
                trail_x = [positions[i][0] for i in range(min(pos_index + 1, len(positions)))]
                trail_y = [positions[i][1] for i in range(min(pos_index + 1, len(positions)))]
                truck_trails[truck_id].set_data(trail_x, trail_y)
            
            # 清除之前的无人机
            for key in list(drone_points.keys()):
                drone_points[key].remove()
                del drone_points[key]
            for key in list(drone_lines.keys()):
                drone_lines[key].remove()
                del drone_lines[key]
            
            # 绘制当前步骤的无人机
            if current_step in self.drone_data:
                for truck_id, drones in self.drone_data[current_step].items():
                    if truck_id in self.truck_data and current_step < len(self.truck_data[truck_id]['positions']):
                        truck_pos = self.truck_data[truck_id]['positions'][current_step]
                        
                        for i, drone in enumerate(drones):
                            target_locker = drone['target_locker']
                            if target_locker in self.locker_positions:
                                target_pos = self.locker_positions[target_locker]
                                
                                # 绘制无人机飞行路径
                                drone_color = self.drone_colors[drone['drone_id'] % len(self.drone_colors)]
                                line_key = f"{truck_id}_{drone['drone_id']}_{i}"
                                
                                drone_lines[line_key], = ax.plot([truck_pos[0], target_pos[0]], 
                                                                [truck_pos[1], target_pos[1]], 
                                                                '--', color=drone_color, alpha=0.7, linewidth=1.5)
                                
                                # 绘制无人机位置（在路径中点）
                                mid_x = (truck_pos[0] + target_pos[0]) / 2
                                mid_y = (truck_pos[1] + target_pos[1]) / 2
                                drone_points[line_key], = ax.plot([mid_x], [mid_y], '^', 
                                                                 color=drone_color, markersize=8, alpha=0.8)
            
            # Update title to show current step
            ax.set_title(f'Truck and Drone Movement Animation - Step {current_step}', fontsize=16, fontweight='bold')
            
            return list(truck_points.values()) + list(truck_trails.values()) + list(truck_texts.values())
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                     interval=interval, blit=False, repeat=True)
        
        # Save animation
        if save_path:
            try:
                print(f"Saving animation to: {save_path}")
                anim.save(save_path, writer='pillow', fps=1)
                print("Animation saved successfully!")
            except Exception as e:
                print(f"Animation save failed: {e}")
        
        return anim
    
    def generate_static_overview(self, save_path=None):
        """
        生成静态总览图
        
        输入:
            save_path (str): 保存路径
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 计算坐标范围
        all_positions = []
        for locker_pos in self.locker_positions.values():
            all_positions.append(locker_pos)
        
        for truck_id, truck_info in self.truck_data.items():
            if truck_info.get('positions'):
                all_positions.extend(truck_info['positions'])
            if truck_info.get('targets'):
                all_positions.extend(truck_info['targets'])
        
        if all_positions:
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 添加边距
            margin = 20
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            # 如果没有位置数据，设置默认范围
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
        
        # 绘制快递柜
        for locker_id, pos in self.locker_positions.items():
            circle = Circle(pos, 3, color=self.locker_color, alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(locker_id), ha='center', va='center', fontsize=8)
        
        # 绘制卡车完整轨迹
        for truck_id, truck_info in self.truck_data.items():
            color = self.truck_colors[truck_id % len(self.truck_colors)]
            positions = truck_info.get('positions', [])
            
            if positions and len(positions) > 0:
                # 确保所有位置都是有效的
                valid_positions = []
                for pos in positions:
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        # 检查位置是否有效（不是 [0, 0] 或者有实际坐标）
                        if pos[0] != 0 or pos[1] != 0 or len(valid_positions) == 0:
                            valid_positions.append([float(pos[0]), float(pos[1])])
                
                if len(valid_positions) > 1:
                    x_coords = [pos[0] for pos in valid_positions]
                    y_coords = [pos[1] for pos in valid_positions]
                    
                    # Draw trajectory
                    ax.plot(x_coords, y_coords, '-', color=color, linewidth=2, alpha=0.7, label=f'Truck {truck_id} Trajectory')
                    
                    # 标记起点和终点
                    if len(valid_positions) > 0:
                        ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=12, markeredgecolor='black', label=f'Truck {truck_id} Start')
                        if len(valid_positions) > 1:
                            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=12, markeredgecolor='black', label=f'Truck {truck_id} End')
                ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=12, markeredgecolor='black')
                
                # 添加步骤标记
                for i, (x, y) in enumerate(positions[::2]):  # 每隔一个步骤标记
                    ax.text(x, y, str(i*2), ha='center', va='center', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))
        
        ax.set_title('Truck Movement Trajectory Overview', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Static overview saved to: {save_path}")
        
        return fig


def main():
    """
    Main function - Generate animation
    """
    # Set test results directory
    test_results_dir = "test_results"
    
    if not os.path.exists(test_results_dir):
        print(f"Test results directory does not exist: {test_results_dir}")
        return
    
    # Create animation generator
    generator = MovementAnimationGenerator(test_results_dir)
    
    # Load data
    if not generator.load_data():
        print("Data loading failed, cannot generate animation")
        return
    
    print("Data loaded successfully, starting animation generation...")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate static overview
    overview_path = f"movement_overview_{timestamp}.png"
    generator.generate_static_overview(overview_path)
    
    # Generate animation
    animation_path = f"movement_animation_{timestamp}.gif"
    anim = generator.create_animation(animation_path, interval=1500)
    
    # Show animation (optional)
    plt.show()


if __name__ == "__main__":
    main()