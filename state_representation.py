"""
状态表示增强模块

实现增强的状态空间表示，包括：
- 全局信息特征
- 时间特征
- 时间窗约束信息
- 动态特征工程
- 需求不确定性表示
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import math


class StateRepresentation:
    """
    增强的状态表示类
    
    提供多层次的状态特征提取，包括局部状态、全局状态、
    时间特征、空间特征和动态特征
    """
    
    def __init__(self, num_trucks: int, num_lockers: int, truck_capacity: int, 
                 depot_location: Tuple[float, float], max_timesteps: int = 100):
        """
        初始化状态表示器
        
        Args:
            num_trucks: 卡车数量
            num_lockers: 快递柜数量
            truck_capacity: 卡车容量
            depot_location: 仓库位置
            max_timesteps: 最大时间步数
        """
        self.num_trucks = num_trucks
        self.num_lockers = num_lockers
        self.truck_capacity = truck_capacity
        self.depot_location = depot_location
        self.max_timesteps = max_timesteps
        
        # 特征维度配置
        self.truck_features_dim = 17  # 每个卡车的基础特征数（8个基础特征 + 9个协调特征）
        self.locker_features_dim = 10  # 每个快递柜的基础特征数（包含需求密度特征）
        self.global_features_dim = 30  # 全局特征数（12基础+6区域密集度+8需求聚合+4覆盖效率）
        self.time_features_dim = 6  # 时间特征数
        self.spatial_features_dim = 8  # 空间特征数
        self.dynamic_features_dim = 10  # 动态特征数
        
        # 路线规划增强特征维度
        self.route_planning_features_dim = 12  # 路线规划特征数
        self.path_history_features_dim = 8  # 历史路径特征数
        self.future_demand_features_dim = 6  # 未来需求预测特征数
        self.coordination_features_dim = 10  # 多卡车协调特征数
        
        # 历史状态缓存（用于动态特征）
        self.state_history = []
        self.max_history_length = 10
        
    def get_enhanced_state(self, env_state: Dict[str, Any], truck_id: int = None) -> np.ndarray:
        """
        获取增强的状态表示
        
        Args:
            env_state: 环境状态字典
            truck_id: 特定卡车ID（如果为None则返回全局状态）
            
        Returns:
            增强的状态向量
        """
        features = []
        
        # 1. 基础状态特征
        if truck_id is not None:
            # 单个卡车的状态
            truck_features = self._extract_truck_features(env_state, truck_id)
            features.extend(truck_features)
        else:
            # 所有卡车的状态
            for i in range(self.num_trucks):
                truck_features = self._extract_truck_features(env_state, i)
                features.extend(truck_features)
        
        # 2. 快递柜状态特征
        locker_features = self._extract_locker_features(env_state)
        features.extend(locker_features)
        
        # 3. 全局状态特征
        global_features = self._extract_global_features(env_state)
        features.extend(global_features)
        
        # 4. 时间特征
        time_features = self._extract_time_features(env_state)
        features.extend(time_features)
        
        # 5. 空间特征
        spatial_features = self._extract_spatial_features(env_state, truck_id)
        features.extend(spatial_features)
        
        # 6. 动态特征
        dynamic_features = self._extract_dynamic_features(env_state)
        features.extend(dynamic_features)
        
        # 7. 路线规划特征
        route_planning_features = self._extract_route_planning_features(env_state, truck_id)
        features.extend(route_planning_features)
        
        # 8. 历史路径特征
        path_history_features = self._extract_path_history_features(env_state, truck_id)
        features.extend(path_history_features)
        
        # 9. 未来需求预测特征
        future_demand_features = self._extract_future_demand_features(env_state)
        features.extend(future_demand_features)
        
        # 10. 多卡车协调特征
        coordination_features = self._extract_coordination_features(env_state, truck_id)
        features.extend(coordination_features)
        
        # 更新历史状态
        self._update_state_history(env_state)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_truck_features(self, env_state: Dict[str, Any], truck_id: int) -> List[float]:
        """
        提取单个卡车的特征
        
        Args:
            env_state: 环境状态
            truck_id: 卡车ID
            
        Returns:
            卡车特征列表
        """
        truck = env_state['trucks'][truck_id]
        
        # 基础特征
        features = [
            truck['current_location'] / self.num_lockers,  # 归一化位置
            truck['current_delivery_load'] / self.truck_capacity,  # 归一化配送负载
            truck['current_return_load'] / self.truck_capacity,  # 归一化退货负载
            truck['remaining_space'] / self.truck_capacity,  # 归一化剩余空间
            float(truck['current_location'] == 0),  # 是否在仓库
        ]
        
        # 负载利用率
        total_load = truck['current_delivery_load'] + truck['current_return_load']
        load_utilization = total_load / self.truck_capacity
        features.append(load_utilization)
        
        # 访问过的站点数量（归一化）
        visited_count = len(truck.get('visited_stops', [])) / self.num_lockers
        features.append(visited_count)
        
        # 距离仓库的距离（归一化）
        if truck['current_location'] == 0:
            distance_to_depot = 0.0
        else:
            current_pos = self._get_location_coordinates(env_state, truck['current_location'])
            distance_to_depot = self._euclidean_distance(current_pos, self.depot_location)
            # 假设最大距离为对角线长度进行归一化
            max_distance = math.sqrt(2) * 100  # 假设地图大小为100x100
            distance_to_depot = min(distance_to_depot / max_distance, 1.0)
        features.append(distance_to_depot)
        
        # 其他卡车的协调信息
        other_trucks = [t for i, t in enumerate(env_state['trucks']) if i != truck_id]
        if other_trucks:
            # 其他卡车的平均负载
            avg_other_load = np.mean([
                (t['current_delivery_load'] + t['current_return_load']) / self.truck_capacity
                for t in other_trucks
            ])
            features.append(avg_other_load)
            
            # 其他卡车是否在相同区域（距离小于阈值）
            current_pos = self._get_location_coordinates(env_state, truck['current_location'])
            nearby_trucks = 0
            min_distance_to_other = float('inf')
            
            for other_truck in other_trucks:
                other_pos = self._get_location_coordinates(env_state, other_truck['current_location'])
                distance = self._euclidean_distance(current_pos, other_pos)
                min_distance_to_other = min(min_distance_to_other, distance)
                
                if distance < 30.0:  # 30单位距离内
                    nearby_trucks += 1
            
            features.append(nearby_trucks / len(other_trucks))
            
            # 到最近其他卡车的距离（归一化）
            max_distance = math.sqrt(2) * 100  # 假设地图大小为100x100
            if min_distance_to_other != float('inf'):
                features.append(min(min_distance_to_other / max_distance, 1.0))
            else:
                features.append(1.0)
            
            # 其他卡车的平均剩余容量
            avg_other_capacity = np.mean([
                t['remaining_space'] / self.truck_capacity
                for t in other_trucks
            ])
            features.append(avg_other_capacity)
            
            # 其他卡车的位置分布（相对于当前卡车）
            # 计算其他卡车在四个象限的分布
            quadrant_counts = [0, 0, 0, 0]  # 右上、左上、左下、右下
            for other_truck in other_trucks:
                other_pos = self._get_location_coordinates(env_state, other_truck['current_location'])
                dx = other_pos[0] - current_pos[0]
                dy = other_pos[1] - current_pos[1]
                
                if dx >= 0 and dy >= 0:
                    quadrant_counts[0] += 1  # 右上
                elif dx < 0 and dy >= 0:
                    quadrant_counts[1] += 1  # 左上
                elif dx < 0 and dy < 0:
                    quadrant_counts[2] += 1  # 左下
                else:
                    quadrant_counts[3] += 1  # 右下
            
            # 归一化象限分布
            total_other_trucks = len(other_trucks)
            features.extend([count / total_other_trucks for count in quadrant_counts])
            
            # 其他卡车是否在同一个快递柜
            same_location_trucks = sum(1 for t in other_trucks 
                                     if t['current_location'] == truck['current_location'])
            features.append(same_location_trucks / len(other_trucks))
            
        else:
            # 如果没有其他卡车，填充默认值
            features.extend([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _extract_locker_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        提取快递柜特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            快递柜特征列表
        """
        features = []
        
        for locker in env_state['lockers']:
            locker_features = [
                locker['location'][0] / 100.0,  # 归一化x坐标
                locker['location'][1] / 100.0,  # 归一化y坐标
                locker['demand_del'] / 50.0,  # 归一化配送需求 (适配新范围15-50)
                locker['demand_ret'] / 50.0,  # 归一化退货需求 (适配新范围10-40)
                locker.get('lambda_del', locker['demand_del']) / 50.0,  # 归一化配送需求率
                locker.get('lambda_ret', locker['demand_ret']) / 50.0,  # 归一化退货需求率
                float(locker['served']),  # 是否已服务
            ]
            
            # 总需求量
            total_demand = locker['demand_del'] + locker['demand_ret']
            locker_features.append(total_demand / 100.0)  # 归一化总需求 (适配新范围25-90)
            
            # 总需求率
            total_lambda = locker.get('lambda_del', locker['demand_del']) + locker.get('lambda_ret', locker['demand_ret'])
            locker_features.append(total_lambda / 100.0)  # 归一化总需求率
            
            # 需求密度（需求量/距离仓库距离）
            distance_to_depot = self._euclidean_distance(locker['location'], self.depot_location)
            demand_density = total_demand / (distance_to_depot + 1e-6)
            locker_features.append(min(demand_density / 10.0, 1.0))  # 归一化需求密度
            
            features.extend(locker_features)
        
        return features
    
    def _extract_global_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        提取增强的全局特征，包括区域密集度和需求聚合信息
        
        Args:
            env_state: 环境状态
            
        Returns:
            全局特征列表
        """
        features = []
        
        # 服务完成率
        served_count = sum(1 for locker in env_state['lockers'] if locker['served'])
        completion_rate = served_count / self.num_lockers
        features.append(completion_rate)
        
        # 卡车在仓库的比例
        at_depot_count = sum(1 for truck in env_state['trucks'] if truck['current_location'] == 0)
        at_depot_rate = at_depot_count / self.num_trucks
        features.append(at_depot_rate)
        
        # 总配送需求完成率
        total_delivery_demand = sum(locker['demand_del'] for locker in env_state['lockers'])
        served_delivery = env_state.get('served_delivery', 0)
        delivery_completion = served_delivery / (total_delivery_demand + 1e-6)
        features.append(min(delivery_completion, 1.0))
        
        # 总退货需求完成率
        total_return_demand = sum(locker['demand_ret'] for locker in env_state['lockers'])
        served_return = env_state.get('served_return', 0)
        return_completion = served_return / (total_return_demand + 1e-6)
        features.append(min(return_completion, 1.0))
        
        # 平均卡车负载利用率
        total_utilization = 0
        for truck in env_state['trucks']:
            total_load = truck['current_delivery_load'] + truck['current_return_load']
            utilization = total_load / self.truck_capacity
            total_utilization += utilization
        avg_utilization = total_utilization / self.num_trucks
        features.append(avg_utilization)
        
        # 剩余未服务快递柜的平均距离
        unserved_lockers = [l for l in env_state['lockers'] if not l['served']]
        if unserved_lockers:
            avg_distance = np.mean([
                self._euclidean_distance(locker['location'], self.depot_location)
                for locker in unserved_lockers
            ])
            features.append(min(avg_distance / 100.0, 1.0))  # 归一化
        else:
            features.append(0.0)
        
        # 剩余总需求量
        remaining_delivery = sum(l['demand_del'] for l in unserved_lockers)
        remaining_return = sum(l['demand_ret'] for l in unserved_lockers)
        features.append(remaining_delivery / (total_delivery_demand + 1e-6))
        features.append(remaining_return / (total_return_demand + 1e-6))
        
        # 卡车总行驶距离（归一化）
        total_distance = env_state.get('total_truck_distance', 0)
        max_possible_distance = self.num_trucks * self.num_lockers * 200  # 估算最大可能距离
        features.append(min(total_distance / max_possible_distance, 1.0))
        
        # 无人机总成本（归一化）
        total_drone_cost = env_state.get('total_drone_cost', 0)
        max_possible_cost = self.num_lockers * 100  # 估算最大可能成本
        features.append(min(total_drone_cost / max_possible_cost, 1.0))
        
        # 效率指标：服务数量/总距离
        if total_distance > 0:
            efficiency = served_count / total_distance
            features.append(min(efficiency * 1000, 1.0))  # 归一化效率
        else:
            features.append(0.0)
        
        # 负载平衡指标
        load_variance = np.var([
            (truck['current_delivery_load'] + truck['current_return_load']) / self.truck_capacity
            for truck in env_state['trucks']
        ])
        features.append(min(load_variance, 1.0))
        
        # 新增：区域密集度特征
        regional_features = self._calculate_regional_density_features(env_state)
        features.extend(regional_features)
        
        # 新增：需求聚合特征
        demand_aggregation_features = self._calculate_demand_aggregation_features(env_state)
        features.extend(demand_aggregation_features)
        
        # 新增：覆盖效率特征
        coverage_features = self._calculate_coverage_efficiency_features(env_state)
        features.extend(coverage_features)
        
        return features
    
    def _calculate_regional_density_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        计算区域密集度特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            区域密集度特征列表
        """
        features = []
        unserved_lockers = [l for l in env_state['lockers'] if not l['served']]
        
        if not unserved_lockers:
            return [0.0] * 6  # 返回6个零特征
        
        # 计算不同半径下的密集度
        radii = [15, 25, 35]
        for radius in radii:
            max_density = 0
            avg_density = 0
            density_count = 0
            
            for locker in unserved_lockers:
                nearby_count = 0
                for other_locker in unserved_lockers:
                    if locker != other_locker:
                        distance = self._euclidean_distance(locker['location'], other_locker['location'])
                        if distance <= radius:
                            nearby_count += 1
                
                density = nearby_count / (math.pi * radius * radius) * 1000  # 归一化密度
                max_density = max(max_density, density)
                avg_density += density
                density_count += 1
            
            if density_count > 0:
                avg_density /= density_count
            
            features.append(min(max_density, 1.0))
            features.append(min(avg_density, 1.0))
        
        return features
    
    def _calculate_demand_aggregation_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        计算需求聚合特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            需求聚合特征列表
        """
        features = []
        unserved_lockers = [l for l in env_state['lockers'] if not l['served']]
        
        if not unserved_lockers:
            return [0.0] * 8  # 返回8个零特征
        
        # 计算需求热点区域
        grid_size = 20  # 网格大小
        demand_grid = {}
        
        for locker in unserved_lockers:
            x, y = locker['location']
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in demand_grid:
                demand_grid[grid_key] = {'delivery': 0, 'return': 0, 'count': 0}
            
            demand_grid[grid_key]['delivery'] += locker['demand_del']
            demand_grid[grid_key]['return'] += locker['demand_ret']
            demand_grid[grid_key]['count'] += 1
        
        if demand_grid:
            # 最高需求密度网格
            max_delivery_density = max(grid['delivery'] / grid['count'] for grid in demand_grid.values())
            max_return_density = max(grid['return'] / grid['count'] for grid in demand_grid.values())
            
            # 平均需求密度
            avg_delivery_density = np.mean([grid['delivery'] / grid['count'] for grid in demand_grid.values()])
            avg_return_density = np.mean([grid['return'] / grid['count'] for grid in demand_grid.values()])
            
            # 需求分布方差
            delivery_densities = [grid['delivery'] / grid['count'] for grid in demand_grid.values()]
            return_densities = [grid['return'] / grid['count'] for grid in demand_grid.values()]
            delivery_variance = np.var(delivery_densities)
            return_variance = np.var(return_densities)
            
            # 高需求网格比例
            high_demand_threshold = avg_delivery_density + avg_return_density
            high_demand_grids = sum(1 for grid in demand_grid.values() 
                                  if (grid['delivery'] + grid['return']) / grid['count'] > high_demand_threshold)
            high_demand_ratio = high_demand_grids / len(demand_grid)
            
            # 网格覆盖范围
            grid_coverage = len(demand_grid)
            
            features.extend([
                min(max_delivery_density / 50.0, 1.0),  # 归一化最大配送密度
                min(max_return_density / 50.0, 1.0),   # 归一化最大退货密度
                min(avg_delivery_density / 30.0, 1.0), # 归一化平均配送密度
                min(avg_return_density / 30.0, 1.0),   # 归一化平均退货密度
                min(delivery_variance / 100.0, 1.0),   # 归一化配送方差
                min(return_variance / 100.0, 1.0),     # 归一化退货方差
                min(high_demand_ratio, 1.0),           # 高需求网格比例
                min(grid_coverage / 50.0, 1.0)         # 归一化网格覆盖
            ])
        else:
            features.extend([0.0] * 8)
        
        return features
    
    def _calculate_coverage_efficiency_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        计算覆盖效率特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            覆盖效率特征列表
        """
        features = []
        unserved_lockers = [l for l in env_state['lockers'] if not l['served']]
        
        if not unserved_lockers:
            return [0.0] * 4  # 返回4个零特征
        
        # 计算每个位置的覆盖潜力
        coverage_potentials = []
        for locker in unserved_lockers:
            # 计算在该位置停靠能覆盖的快递柜数量
            coverage_count = 0
            total_covered_demand = 0
            
            for other_locker in unserved_lockers:
                distance = self._euclidean_distance(locker['location'], other_locker['location'])
                if distance <= 30:  # 无人机覆盖半径
                    coverage_count += 1
                    total_covered_demand += other_locker['demand_del'] + other_locker['demand_ret']
            
            # 覆盖效率 = 覆盖数量 * 需求密度
            if coverage_count > 0:
                coverage_efficiency = coverage_count * (total_covered_demand / coverage_count)
                coverage_potentials.append(coverage_efficiency)
        
        if coverage_potentials:
            # 最大覆盖效率
            max_coverage_efficiency = max(coverage_potentials)
            features.append(min(max_coverage_efficiency / 500.0, 1.0))
            
            # 平均覆盖效率
            avg_coverage_efficiency = np.mean(coverage_potentials)
            features.append(min(avg_coverage_efficiency / 300.0, 1.0))
            
            # 覆盖效率方差（衡量分布均匀性）
            coverage_variance = np.var(coverage_potentials)
            features.append(min(coverage_variance / 10000.0, 1.0))
            
            # 高效覆盖点比例
            high_efficiency_threshold = avg_coverage_efficiency * 1.5
            high_efficiency_count = sum(1 for eff in coverage_potentials if eff > high_efficiency_threshold)
            high_efficiency_ratio = high_efficiency_count / len(coverage_potentials)
            features.append(min(high_efficiency_ratio, 1.0))
        else:
            features.extend([0.0] * 4)
        
        return features
    
    def _extract_time_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        提取时间特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            时间特征列表
        """
        current_time = env_state['time_step']
        max_time = env_state['max_timesteps']
        
        features = [
            current_time / max_time,  # 归一化时间进度
            math.sin(2 * math.pi * current_time / max_time),  # 周期性时间特征
            math.cos(2 * math.pi * current_time / max_time),  # 周期性时间特征
        ]
        
        # 剩余时间比例
        remaining_time_ratio = (max_time - current_time) / max_time
        features.append(remaining_time_ratio)
        
        # 时间紧迫度（基于剩余任务和剩余时间）
        unserved_count = sum(1 for locker in env_state['lockers'] if not locker['served'])
        if remaining_time_ratio > 0:
            urgency = unserved_count / (remaining_time_ratio * max_time + 1e-6)
            features.append(min(urgency, 1.0))
        else:
            features.append(1.0)
        
        # 时间效率（已完成任务数/已用时间）
        if current_time > 0:
            served_count = sum(1 for locker in env_state['lockers'] if locker['served'])
            time_efficiency = served_count / current_time
            features.append(min(time_efficiency, 1.0))
        else:
            features.append(0.0)
        
        return features
    
    def _extract_spatial_features(self, env_state: Dict[str, Any], truck_id: int = None) -> List[float]:
        """
        提取空间特征
        
        Args:
            env_state: 环境状态
            truck_id: 特定卡车ID
            
        Returns:
            空间特征列表
        """
        features = []
        
        if truck_id is not None:
            truck = env_state['trucks'][truck_id]
            current_pos = self._get_location_coordinates(env_state, truck['current_location'])
            
            # 到最近未服务快递柜的距离
            unserved_lockers = [l for l in env_state['lockers'] if not l['served']]
            if unserved_lockers:
                min_distance = min([
                    self._euclidean_distance(current_pos, locker['location'])
                    for locker in unserved_lockers
                ])
                features.append(min(min_distance / 100.0, 1.0))
            else:
                features.append(0.0)
            
            # 到最远未服务快递柜的距离
            if unserved_lockers:
                max_distance = max([
                    self._euclidean_distance(current_pos, locker['location'])
                    for locker in unserved_lockers
                ])
                features.append(min(max_distance / 100.0, 1.0))
            else:
                features.append(0.0)
            
            # 周围快递柜密度（半径内的快递柜数量）
            radius = 20.0
            nearby_count = sum(1 for locker in unserved_lockers
                             if self._euclidean_distance(current_pos, locker['location']) <= radius)
            density = nearby_count / len(unserved_lockers) if unserved_lockers else 0
            features.append(density)
            
            # 到其他卡车的最小距离
            other_trucks = [t for i, t in enumerate(env_state['trucks']) if i != truck_id and not t['returned']]
            if other_trucks:
                min_truck_distance = min([
                    self._euclidean_distance(current_pos, self._get_location_coordinates(env_state, t['current_location']))
                    for t in other_trucks
                ])
                features.append(min(min_truck_distance / 100.0, 1.0))
            else:
                features.append(1.0)
        else:
            # 全局空间特征
            features.extend([0.0] * 4)  # 占位符
        
        # 快递柜分布特征
        locker_positions = [locker['location'] for locker in env_state['lockers']]
        if locker_positions:
            # 快递柜重心
            centroid_x = np.mean([pos[0] for pos in locker_positions])
            centroid_y = np.mean([pos[1] for pos in locker_positions])
            features.extend([centroid_x / 100.0, centroid_y / 100.0])
            
            # 快递柜分散度
            distances_to_centroid = [
                self._euclidean_distance(pos, (centroid_x, centroid_y))
                for pos in locker_positions
            ]
            dispersion = np.std(distances_to_centroid)
            features.append(min(dispersion / 50.0, 1.0))
            
            # 快递柜覆盖面积（凸包面积的近似）
            x_range = max(pos[0] for pos in locker_positions) - min(pos[0] for pos in locker_positions)
            y_range = max(pos[1] for pos in locker_positions) - min(pos[1] for pos in locker_positions)
            coverage_area = x_range * y_range
            features.append(min(coverage_area / 10000.0, 1.0))  # 归一化到100x100区域
        else:
            features.extend([0.0] * 4)
        
        return features
    
    def _extract_dynamic_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        提取动态特征（基于历史状态变化）
        
        Args:
            env_state: 环境状态
            
        Returns:
            动态特征列表
        """
        features = []
        
        if len(self.state_history) < 2:
            # 历史不足，返回零特征
            return [0.0] * self.dynamic_features_dim
        
        # 服务速度（最近几步的服务数量变化）
        current_served = sum(1 for locker in env_state['lockers'] if locker['served'])
        prev_served = sum(1 for locker in self.state_history[-1]['lockers'] if locker['served'])
        service_rate = (current_served - prev_served) / 1.0  # 每步服务速度
        features.append(max(min(service_rate, 1.0), -1.0))
        
        # 距离变化率
        current_distance = env_state.get('total_truck_distance', 0)
        prev_distance = self.state_history[-1].get('total_truck_distance', 0)
        distance_rate = (current_distance - prev_distance) / 100.0  # 归一化距离变化
        features.append(max(min(distance_rate, 1.0), 0.0))
        
        # 负载变化趋势
        current_avg_load = np.mean([
            (truck['current_delivery_load'] + truck['current_return_load']) / self.truck_capacity
            for truck in env_state['trucks']
        ])
        prev_avg_load = np.mean([
            (truck['current_delivery_load'] + truck['current_return_load']) / self.truck_capacity
            for truck in self.state_history[-1]['trucks']
        ])
        load_change = current_avg_load - prev_avg_load
        features.append(max(min(load_change, 1.0), -1.0))
        
        # 效率趋势（最近几步的效率变化）
        if len(self.state_history) >= 3:
            recent_service_rates = []
            for i in range(-3, 0):
                if i == -1:
                    served_now = current_served
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                else:
                    served_now = sum(1 for locker in self.state_history[i+1]['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                recent_service_rates.append(served_now - served_prev)
            
            efficiency_trend = np.mean(recent_service_rates)
            features.append(max(min(efficiency_trend, 1.0), -1.0))
        else:
            features.append(0.0)
        
        # 卡车位置变化方差（移动活跃度）
        position_changes = []
        for i, truck in enumerate(env_state['trucks']):
            if i < len(self.state_history[-1]['trucks']):
                prev_truck = self.state_history[-1]['trucks'][i]
                if truck['current_location'] != prev_truck['current_location']:
                    position_changes.append(1.0)
                else:
                    position_changes.append(0.0)
        
        mobility = np.mean(position_changes) if position_changes else 0.0
        features.append(mobility)
        
        # 剩余特征填充
        while len(features) < self.dynamic_features_dim:
            features.append(0.0)
        
        return features[:self.dynamic_features_dim]
    
    def _extract_route_planning_features(self, env_state: Dict[str, Any], truck_id: int = None) -> List[float]:
        """
        提取路线规划相关特征
        
        Args:
            env_state: 环境状态
            truck_id: 卡车ID
            
        Returns:
            路线规划特征列表
        """
        features = []
        
        # 计算未服务的快递柜（在方法开始就定义，避免作用域问题）
        unserved_lockers = [locker for locker in env_state['lockers'] if not locker['served']]
        
        # 1. 路径效率指标
        if truck_id is not None and truck_id < len(env_state['trucks']):
            truck = env_state['trucks'][truck_id]
            current_location = truck['current_location']
            
            # 计算到最近未服务快递柜的距离
            if unserved_lockers:
                min_distance = min(
                    self._euclidean_distance(
                        self._get_location_coordinates(env_state, current_location),
                        self._get_location_coordinates(env_state, locker['id'])
                    ) for locker in unserved_lockers
                )
                features.append(min(min_distance / 100.0, 1.0))  # 归一化距离
                
                # 平均距离到未服务快递柜
                avg_distance = np.mean([
                    self._euclidean_distance(
                        self._get_location_coordinates(env_state, current_location),
                        self._get_location_coordinates(env_state, locker['id'])
                    ) for locker in unserved_lockers
                ])
                features.append(min(avg_distance / 200.0, 1.0))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # 2. 路径优化潜力
        total_unserved = sum(1 for locker in env_state['lockers'] if not locker['served'])
        total_lockers = len(env_state['lockers'])
        completion_ratio = 1.0 - (total_unserved / total_lockers) if total_lockers > 0 else 1.0
        features.append(completion_ratio)
        
        # 3. 时间压力指标
        current_time = env_state['time_step']
        max_time = env_state.get('max_timesteps', self.max_timesteps)
        time_pressure = current_time / max_time
        features.append(time_pressure)
        
        # 4. 负载平衡指标
        if env_state['trucks']:
            loads = [truck['current_delivery_load'] + truck['current_return_load'] for truck in env_state['trucks']]
            load_variance = np.var(loads) / (self.truck_capacity ** 2) if self.truck_capacity > 0 else 0.0
            features.append(min(load_variance, 1.0))
            
            # 平均负载率
            avg_load_ratio = np.mean([load / self.truck_capacity for load in loads])
            features.append(avg_load_ratio)
        else:
            features.extend([0.0, 0.0])
        
        # 5. 区域覆盖效率
        if unserved_lockers:
            # 计算未服务区域的分散程度
            unserved_coords = [
                self._get_location_coordinates(env_state, locker['id']) 
                for locker in unserved_lockers
            ]
            if len(unserved_coords) > 1:
                distances = []
                for i in range(len(unserved_coords)):
                    for j in range(i+1, len(unserved_coords)):
                        distances.append(self._euclidean_distance(unserved_coords[i], unserved_coords[j]))
                avg_spread = np.mean(distances) / 100.0  # 归一化
                features.append(min(avg_spread, 1.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 6. 服务密度指标
        served_count = sum(1 for locker in env_state['lockers'] if locker['served'])
        service_density = served_count / len(env_state['lockers']) if env_state['lockers'] else 0.0
        features.append(service_density)
        
        # 7. 路径连续性指标
        if len(self.state_history) >= 2 and truck_id is not None:
            current_loc = env_state['trucks'][truck_id]['current_location']
            prev_loc = self.state_history[-1]['trucks'][truck_id]['current_location']
            
            # 检查是否在连续服务相邻区域
            continuity_score = 1.0 if abs(current_loc - prev_loc) <= 1 else 0.0
            features.append(continuity_score)
        else:
            features.append(0.0)
        
        # 8. 剩余工作量评估
        remaining_work = total_unserved / total_lockers if total_lockers > 0 else 0.0
        remaining_time = (max_time - current_time) / max_time if max_time > 0 else 0.0
        work_time_ratio = remaining_work / remaining_time if remaining_time > 0 else 0.0
        features.append(min(work_time_ratio, 2.0) / 2.0)  # 归一化到[0,1]
        
        # 9. 协调需求指标
        if len(env_state['trucks']) > 1:
            truck_positions = [truck['current_location'] for truck in env_state['trucks']]
            position_variance = np.var(truck_positions)
            coordination_need = min(position_variance / 100.0, 1.0)
            features.append(coordination_need)
        else:
            features.append(0.0)
        
        # 确保特征数量正确
        while len(features) < self.route_planning_features_dim:
            features.append(0.0)
        
        return features[:self.route_planning_features_dim]
    
    def _extract_path_history_features(self, env_state: Dict[str, Any], truck_id: int = None) -> List[float]:
        """
        提取历史路径特征
        
        Args:
            env_state: 环境状态
            truck_id: 卡车ID
            
        Returns:
            历史路径特征列表
        """
        features = []
        
        if truck_id is not None and len(self.state_history) >= 3:
            # 1. 路径长度趋势
            path_lengths = []
            for i in range(-3, 0):
                if i == -1:
                    current_loc = env_state['trucks'][truck_id]['current_location']
                    prev_loc = self.state_history[i]['trucks'][truck_id]['current_location']
                else:
                    current_loc = self.state_history[i+1]['trucks'][truck_id]['current_location']
                    prev_loc = self.state_history[i]['trucks'][truck_id]['current_location']
                
                distance = abs(current_loc - prev_loc)
                path_lengths.append(distance)
            
            # 路径长度变化趋势
            if len(path_lengths) >= 2:
                length_trend = (path_lengths[-1] - path_lengths[0]) / max(path_lengths[0], 1)
                features.append(max(min(length_trend, 1.0), -1.0))
            else:
                features.append(0.0)
            
            # 2. 移动频率
            movement_count = sum(1 for length in path_lengths if length > 0)
            movement_frequency = movement_count / len(path_lengths)
            features.append(movement_frequency)
            
            # 3. 路径效率历史
            efficiency_scores = []
            for i in range(-3, 0):
                if i == -1:
                    served_now = sum(1 for locker in env_state['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                else:
                    served_now = sum(1 for locker in self.state_history[i+1]['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                
                service_gain = served_now - served_prev
                path_cost = path_lengths[i + 3] if i + 3 < len(path_lengths) else 0
                efficiency = service_gain / max(path_cost, 1) if path_cost > 0 else service_gain
                efficiency_scores.append(efficiency)
            
            avg_efficiency = np.mean(efficiency_scores)
            features.append(max(min(avg_efficiency, 2.0), 0.0) / 2.0)  # 归一化
            
            # 4. 回程频率（返回仓库的频率）
            depot_visits = sum(1 for i in range(-3, 0) 
                             if self.state_history[i]['trucks'][truck_id]['current_location'] == 0)
            depot_frequency = depot_visits / 3
            features.append(depot_frequency)
            
        else:
            features.extend([0.0] * 4)
        
        # 5. 服务连续性
        if len(self.state_history) >= 2:
            consecutive_services = 0
            for i in range(-min(len(self.state_history), 5), 0):
                if i == -1:
                    served_now = sum(1 for locker in env_state['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                else:
                    served_now = sum(1 for locker in self.state_history[i+1]['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                
                if served_now > served_prev:
                    consecutive_services += 1
                else:
                    break
            
            service_continuity = consecutive_services / 5.0
            features.append(service_continuity)
        else:
            features.append(0.0)
        
        # 6. 区域专注度
        if truck_id is not None and len(self.state_history) >= 3:
            recent_locations = []
            for i in range(-3, 0):
                recent_locations.append(self.state_history[i]['trucks'][truck_id]['current_location'])
            recent_locations.append(env_state['trucks'][truck_id]['current_location'])
            
            unique_locations = len(set(recent_locations))
            focus_score = 1.0 - (unique_locations - 1) / 3.0  # 越专注分数越高
            features.append(max(focus_score, 0.0))
        else:
            features.append(0.0)
        
        # 7. 负载变化模式
        if truck_id is not None and len(self.state_history) >= 2:
            load_changes = []
            for i in range(-2, 0):
                if i == -1:
                    current_load = env_state['trucks'][truck_id]['current_delivery_load'] + env_state['trucks'][truck_id]['current_return_load']
                    prev_load = self.state_history[i]['trucks'][truck_id]['current_delivery_load'] + self.state_history[i]['trucks'][truck_id]['current_return_load']
                else:
                    current_load = self.state_history[i+1]['trucks'][truck_id]['current_delivery_load'] + self.state_history[i+1]['trucks'][truck_id]['current_return_load']
                    prev_load = self.state_history[i]['trucks'][truck_id]['current_delivery_load'] + self.state_history[i]['trucks'][truck_id]['current_return_load']
                
                load_change = current_load - prev_load
                load_changes.append(load_change)
            
            load_stability = 1.0 - (np.var(load_changes) / (self.truck_capacity ** 2))
            features.append(max(load_stability, 0.0))
        else:
            features.append(0.0)
        
        # 8. 时间利用效率
        if len(self.state_history) >= 1:
            time_span = min(len(self.state_history), 5)
            total_services = 0
            
            for i in range(-time_span, 0):
                if i == -1:
                    served_now = sum(1 for locker in env_state['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                else:
                    served_now = sum(1 for locker in self.state_history[i+1]['lockers'] if locker['served'])
                    served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                
                total_services += max(served_now - served_prev, 0)
            
            time_efficiency = total_services / time_span
            features.append(min(time_efficiency, 1.0))
        else:
            features.append(0.0)
        
        # 确保特征数量正确
        while len(features) < self.path_history_features_dim:
            features.append(0.0)
        
        return features[:self.path_history_features_dim]
    
    def _extract_future_demand_features(self, env_state: Dict[str, Any]) -> List[float]:
        """
        提取未来需求预测特征
        
        Args:
            env_state: 环境状态
            
        Returns:
            未来需求预测特征列表
        """
        features = []
        
        # 1. 剩余需求分布
        unserved_lockers = [locker for locker in env_state['lockers'] if not locker['served']]
        total_lockers = len(env_state['lockers'])
        
        if unserved_lockers:
            # 剩余需求比例
            remaining_ratio = len(unserved_lockers) / total_lockers
            features.append(remaining_ratio)
            
            # 剩余需求的空间分布
            unserved_locations = [locker['id'] for locker in unserved_lockers]
            location_spread = np.var(unserved_locations) / (total_lockers ** 2) if total_lockers > 0 else 0.0
            features.append(min(location_spread, 1.0))
        else:
            features.extend([0.0, 0.0])
        
        # 2. 时间窗压力预测
        current_time = env_state['time_step']
        max_time = env_state.get('max_timesteps', self.max_timesteps)
        remaining_time = max_time - current_time
        
        if remaining_time > 0 and unserved_lockers:
            # 平均每个时间步需要完成的服务数
            required_service_rate = len(unserved_lockers) / remaining_time
            features.append(min(required_service_rate, 2.0) / 2.0)  # 归一化
        else:
            features.append(0.0)
        
        # 3. 容量需求预测
        if env_state['trucks']:
            total_current_load = sum(truck['current_delivery_load'] + truck['current_return_load'] for truck in env_state['trucks'])
            total_capacity = len(env_state['trucks']) * self.truck_capacity
            capacity_utilization = total_current_load / total_capacity if total_capacity > 0 else 0.0
            
            # 预测未来容量需求
            if unserved_lockers:
                estimated_future_load = len(unserved_lockers)  # 假设每个快递柜需要1单位容量
                future_capacity_need = estimated_future_load / total_capacity
                features.append(min(future_capacity_need, 1.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 4. 服务难度预测
        if unserved_lockers and env_state['trucks']:
            # 计算剩余快递柜的平均距离（从当前卡车位置）
            total_distance = 0
            distance_count = 0
            
            for truck in env_state['trucks']:
                truck_pos = self._get_location_coordinates(env_state, truck['current_location'])
                for locker in unserved_lockers:
                    locker_pos = self._get_location_coordinates(env_state, locker['id'])
                    distance = self._euclidean_distance(truck_pos, locker_pos)
                    total_distance += distance
                    distance_count += 1
            
            if distance_count > 0:
                avg_service_difficulty = (total_distance / distance_count) / 100.0  # 归一化
                features.append(min(avg_service_difficulty, 1.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 5. 协调复杂度预测
        if len(env_state['trucks']) > 1 and unserved_lockers:
            # 计算需要多卡车协调的区域数量
            truck_positions = [truck['current_location'] for truck in env_state['trucks']]
            unserved_positions = [locker['id'] for locker in unserved_lockers]
            
            # 估算需要协调的复杂度
            coordination_zones = len(set(unserved_positions)) / len(env_state['trucks'])
            coordination_complexity = min(coordination_zones / 5.0, 1.0)  # 归一化
            features.append(coordination_complexity)
        else:
            features.append(0.0)
        
        # 6. 完成时间预测
        if unserved_lockers and remaining_time > 0:
            # 基于当前服务速度预测完成时间
            if len(self.state_history) >= 2:
                recent_service_rate = 0
                for i in range(-min(len(self.state_history), 3), 0):
                    if i == -1:
                        served_now = sum(1 for locker in env_state['lockers'] if locker['served'])
                        served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                    else:
                        served_now = sum(1 for locker in self.state_history[i+1]['lockers'] if locker['served'])
                        served_prev = sum(1 for locker in self.state_history[i]['lockers'] if locker['served'])
                    
                    recent_service_rate += max(served_now - served_prev, 0)
                
                avg_service_rate = recent_service_rate / min(len(self.state_history), 3)
                
                if avg_service_rate > 0:
                    estimated_completion_time = len(unserved_lockers) / avg_service_rate
                    completion_pressure = estimated_completion_time / remaining_time
                    features.append(min(completion_pressure, 2.0) / 2.0)  # 归一化
                else:
                    features.append(1.0)  # 高压力
            else:
                features.append(0.5)  # 中等压力
        else:
            features.append(0.0)
        
        # 确保特征数量正确
        while len(features) < self.future_demand_features_dim:
            features.append(0.0)
        
        return features[:self.future_demand_features_dim]
    
    def _extract_coordination_features(self, env_state: Dict[str, Any], truck_id: int = None) -> List[float]:
        """
        提取多卡车协调特征
        
        Args:
            env_state: 环境状态
            truck_id: 卡车ID
            
        Returns:
            多卡车协调特征列表
        """
        features = []
        
        if len(env_state['trucks']) <= 1:
            # 单卡车情况，返回零特征
            return [0.0] * self.coordination_features_dim
        
        # 1. 卡车间距离分布
        truck_positions = []
        for truck in env_state['trucks']:
            pos = self._get_location_coordinates(env_state, truck['current_location'])
            truck_positions.append(pos)
        
        # 计算卡车间的平均距离
        inter_truck_distances = []
        for i in range(len(truck_positions)):
            for j in range(i+1, len(truck_positions)):
                distance = self._euclidean_distance(truck_positions[i], truck_positions[j])
                inter_truck_distances.append(distance)
        
        if inter_truck_distances:
            avg_inter_distance = np.mean(inter_truck_distances) / 100.0  # 归一化
            features.append(min(avg_inter_distance, 1.0))
            
            # 距离方差（分散程度）
            distance_variance = np.var(inter_truck_distances) / (100.0 ** 2)
            features.append(min(distance_variance, 1.0))
        else:
            features.extend([0.0, 0.0])
        
        # 2. 负载平衡度
        truck_loads = [truck['current_delivery_load'] + truck['current_return_load'] for truck in env_state['trucks']]
        load_mean = np.mean(truck_loads)
        load_variance = np.var(truck_loads) / (self.truck_capacity ** 2) if self.truck_capacity > 0 else 0.0
        
        load_balance = 1.0 - min(load_variance, 1.0)  # 方差越小，平衡度越高
        features.append(load_balance)
        
        # 3. 工作分配均匀度
        if truck_id is not None:
            # 当前卡车相对于其他卡车的负载
            current_load = env_state['trucks'][truck_id]['current_delivery_load'] + env_state['trucks'][truck_id]['current_return_load']
            relative_load = current_load / load_mean if load_mean > 0 else 1.0
            features.append(min(relative_load, 2.0) / 2.0)  # 归一化
        else:
            features.append(0.5)  # 中性值
        
        # 4. 区域覆盖协调
        truck_locations = [truck['current_location'] for truck in env_state['trucks']]
        location_spread = np.var(truck_locations)
        coverage_coordination = min(location_spread / 100.0, 1.0)
        features.append(coverage_coordination)
        
        # 5. 服务效率协调
        if len(self.state_history) >= 1:
            # 计算各卡车的服务贡献
            truck_contributions = []
            for i, truck in enumerate(env_state['trucks']):
                # 简化计算：基于位置变化推断服务活跃度
                if len(self.state_history) > 0:
                    prev_location = self.state_history[-1]['trucks'][i]['current_location']
                    current_location = truck['current_location']
                    activity = 1.0 if current_location != prev_location else 0.0
                    truck_contributions.append(activity)
                else:
                    truck_contributions.append(0.0)
            
            # 服务贡献的均匀度
            if truck_contributions:
                contribution_variance = np.var(truck_contributions)
                service_coordination = 1.0 - min(contribution_variance, 1.0)
                features.append(service_coordination)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 6. 冲突避免指标
        # 检查是否有卡车在相同或相邻位置
        location_conflicts = 0
        for i in range(len(truck_locations)):
            for j in range(i+1, len(truck_locations)):
                if abs(truck_locations[i] - truck_locations[j]) <= 1:
                    location_conflicts += 1
        
        conflict_ratio = location_conflicts / max(len(truck_locations) * (len(truck_locations) - 1) / 2, 1)
        conflict_avoidance = 1.0 - conflict_ratio
        features.append(conflict_avoidance)
        
        # 7. 协调响应性
        if truck_id is not None and len(self.state_history) >= 2:
            # 检查当前卡车是否响应了其他卡车的行动
            current_location = env_state['trucks'][truck_id]['current_location']
            prev_location = self.state_history[-1]['trucks'][truck_id]['current_location']
            
            # 检查其他卡车的移动模式
            other_movements = []
            for i, truck in enumerate(env_state['trucks']):
                if i != truck_id:
                    other_prev = self.state_history[-1]['trucks'][i]['current_location']
                    other_current = truck['current_location']
                    other_movements.append(other_current - other_prev)
            
            # 简化的协调响应性计算
            if other_movements and any(move != 0 for move in other_movements):
                current_movement = current_location - prev_location
                # 检查是否与其他卡车的移动方向协调
                coordination_response = 1.0 if current_movement != 0 else 0.5
                features.append(coordination_response)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 8. 全局优化贡献
        unserved_lockers = [locker for locker in env_state['lockers'] if not locker['served']]
        if truck_id is not None and unserved_lockers:
            # 计算当前卡车对全局目标的贡献潜力
            truck_pos = self._get_location_coordinates(env_state, env_state['trucks'][truck_id]['current_location'])
            
            # 到最近未服务快递柜的距离
            min_distance_to_unserved = min(
                self._euclidean_distance(truck_pos, self._get_location_coordinates(env_state, locker['id']))
                for locker in unserved_lockers
            )
            
            # 相对于其他卡车的优势
            other_min_distances = []
            for i, truck in enumerate(env_state['trucks']):
                if i != truck_id:
                    other_pos = self._get_location_coordinates(env_state, truck['current_location'])
                    other_min_dist = min(
                        self._euclidean_distance(other_pos, self._get_location_coordinates(env_state, locker['id']))
                        for locker in unserved_lockers
                    )
                    other_min_distances.append(other_min_dist)
            
            if other_min_distances:
                avg_other_distance = np.mean(other_min_distances)
                relative_advantage = avg_other_distance / max(min_distance_to_unserved, 1.0)
                global_contribution = min(relative_advantage / 2.0, 1.0)  # 归一化
                features.append(global_contribution)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 9. 动态协调适应性
        if len(self.state_history) >= 3:
            # 分析最近几步的协调模式变化
            recent_spreads = []
            for i in range(-3, 0):
                if i == -1:
                    locations = [truck['current_location'] for truck in env_state['trucks']]
                else:
                    locations = [truck['current_location'] for truck in self.state_history[i]['trucks']]
                spread = np.var(locations)
                recent_spreads.append(spread)
            
            # 协调适应性：分散度变化的稳定性
            spread_stability = 1.0 - (np.var(recent_spreads) / max(np.mean(recent_spreads), 1.0))
            features.append(max(spread_stability, 0.0))
        else:
            features.append(0.5)
        
        # 10. 团队效率指标
        if len(self.state_history) >= 1:
            # 团队整体服务速度
            total_served_now = sum(1 for locker in env_state['lockers'] if locker['served'])
            total_served_prev = sum(1 for locker in self.state_history[-1]['lockers'] if locker['served'])
            team_service_rate = max(total_served_now - total_served_prev, 0)
            
            # 相对于卡车数量的效率
            team_efficiency = team_service_rate / len(env_state['trucks'])
            features.append(min(team_efficiency, 1.0))
        else:
            features.append(0.0)
        
        # 确保特征数量正确
        while len(features) < self.coordination_features_dim:
            features.append(0.0)
        
        return features[:self.coordination_features_dim]
    
    def get_state_vector(self, trucks: List[Dict], lockers: List[Dict], 
                        time_step: int, total_distance: float = 0, 
                        total_drone_cost: float = 0) -> np.ndarray:
        """
        获取状态向量（兼容truck_routing.py的调用接口）
        
        Args:
            trucks: 卡车状态列表
            lockers: 快递柜状态列表
            time_step: 当前时间步
            total_distance: 总行驶距离
            total_drone_cost: 总无人机成本
            
        Returns:
            状态向量
        """
        # 构建环境状态字典
        env_state = {
            'trucks': trucks,
            'lockers': lockers,
            'time_step': time_step,
            'max_timesteps': self.max_timesteps,
            'total_truck_distance': total_distance,
            'total_drone_cost': total_drone_cost,
            'served_delivery': sum(1 for locker in lockers if locker.get('served', False)),
            'served_return': sum(1 for locker in lockers if locker.get('served', False))
        }
        
        # 调用增强状态表示方法
        return self.get_enhanced_state(env_state)

    def _update_state_history(self, env_state: Dict[str, Any]):
        """
        更新状态历史
        
        Args:
            env_state: 当前环境状态
        """
        # 深拷贝状态以避免引用问题
        state_copy = {
            'time_step': env_state['time_step'],
            'trucks': [truck.copy() for truck in env_state['trucks']],
            'lockers': [locker.copy() for locker in env_state['lockers']],
            'total_truck_distance': env_state.get('total_truck_distance', 0),
            'total_drone_cost': env_state.get('total_drone_cost', 0),
            'served_delivery': env_state.get('served_delivery', 0),
            'served_return': env_state.get('served_return', 0)
        }
        
        self.state_history.append(state_copy)
        
        # 限制历史长度
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    def _get_location_coordinates(self, env_state: Dict[str, Any], location_id: int) -> Tuple[float, float]:
        """
        根据位置ID获取坐标
        
        Args:
            env_state: 环境状态
            location_id: 位置ID（0为仓库，其他为快递柜ID）
            
        Returns:
            位置坐标
        """
        if location_id == 0:
            return self.depot_location
        
        for locker in env_state['lockers']:
            if locker.get('id', 0) == location_id:
                return locker['location']
        
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
    
    def get_state_dimension(self) -> int:
        """
        获取状态向量的总维度
        
        Returns:
            状态向量维度
        """
        total_dim = (
            self.truck_features_dim * self.num_trucks +
            self.locker_features_dim * self.num_lockers +
            self.global_features_dim +
            self.time_features_dim +
            self.spatial_features_dim +
            self.dynamic_features_dim +
            self.route_planning_features_dim +
            self.path_history_features_dim +
            self.future_demand_features_dim +
            self.coordination_features_dim
        )
        return total_dim
    
    def reset(self):
        """
        重置状态表示器（清空历史）
        """
        self.state_history.clear()


class TimeWindowConstraints:
    """
    时间窗约束处理类
    
    实现软时间窗约束，提供时间窗违规的渐进式惩罚
    """
    
    def __init__(self, num_lockers: int, soft_penalty_factor: float = 0.1):
        """
        初始化时间窗约束
        
        Args:
            num_lockers: 快递柜数量
            soft_penalty_factor: 软时间窗惩罚因子
        """
        self.num_lockers = num_lockers
        self.soft_penalty_factor = soft_penalty_factor
        
        # 为每个快递柜生成时间窗
        self.time_windows = self._generate_time_windows()
    
    def _generate_time_windows(self) -> List[Tuple[int, int]]:
        """
        生成快递柜的时间窗
        
        Returns:
            时间窗列表，每个元素为(开始时间, 结束时间)
        """
        time_windows = []
        for i in range(self.num_lockers):
            # 随机生成时间窗，确保有一定的重叠
            start_time = np.random.randint(0, 50)
            window_length = np.random.randint(20, 60)
            end_time = start_time + window_length
            time_windows.append((start_time, end_time))
        
        return time_windows