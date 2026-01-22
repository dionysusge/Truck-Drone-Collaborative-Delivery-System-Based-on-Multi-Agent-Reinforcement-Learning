import pandas as pd
import numpy as np
import math
import config
import os
from tqdm import tqdm
from drone_delivery_simulator import (
    Drone,
    LockersManager,
    euclidean_distance,
    process_forced_deliveries,
    process_optimized_deliveries,
    get_next_task,
    find_best_combo_task
)


# ========== 特征计算函数 ==========
def calculate_fdcr(scene_data, center=(0, 0)):
    """
    计算FDCR指标（强制配送完成后的剩余需求距离乘积）

    公式：FDCR = 2 × Σ [|取货需求 - 退货需求| × 距离(快递柜位置, 中心点)]

    参数:
    scene_data: 包含场景数据的Series或字典
    center: 中心点坐标，默认为(0, 0)

    返回:
    fdcr_value: FDCR指标值
    """
    fdcr_sum = 0.0

    # 遍历所有可能的快递柜
    for i in range(1, config.MAX_LOCKERS + 1):
        # 获取快递柜数据
        x = scene_data.get(f'locker_{i}_x')
        y = scene_data.get(f'locker_{i}_y')
        delivery = scene_data.get(f'locker_{i}_delivery', 0)
        return_d = scene_data.get(f'locker_{i}_return', 0)

        # 检查位置是否有效
        if pd.isnull(x) or pd.isnull(y):
            continue

        # 计算剩余需求（取货和退货的绝对差值）
        remaining_demand = abs(delivery - return_d)

        # 计算到中心点的距离
        distance = euclidean_distance((x, y), center)

        # 累加乘积
        fdcr_sum += remaining_demand * distance

    # 乘以2得到最终结果
    fdcr_value = 2 * fdcr_sum
    return fdcr_value

def calculate_cp(lockers, center):
    """计算中心近度值 (Centrality Proximity)"""
    total_distance = 0
    count = 0
    for locker in lockers:
        pos = locker[:2]
        total_distance += euclidean_distance(pos, center)
        count += 1
    return total_distance / count if count > 0 else 0


def calculate_di(lockers):
    """计算需求失衡度 (Demand Imbalance)"""
    total_delivery = 0
    total_return = 0
    for locker in lockers:
        total_delivery += locker[2]  # 取货需求
        total_return += locker[3]  # 退货需求

    total_demand = total_delivery + total_return
    if total_demand == 0:
        return 0
    return abs(total_delivery - total_return) / total_demand


def calculate_sd(lockers):
    """计算空间分散度 (Spatial Dispersion)"""
    n = len(lockers)
    if n < 2:
        return 0

    total_distance = 0
    count = 0
    for i in range(n):
        pos_i = lockers[i][:2]
        for j in range(i + 1, n):
            pos_j = lockers[j][:2]
            total_distance += euclidean_distance(pos_i, pos_j)
            count += 1
    return total_distance / count if count > 0 else 0


def calculate_ci(lockers):
    """计算簇聚指数 (Cluster Index)"""
    if len(lockers) < 3:
        return 1.0

    # 计算凸包面积
    points = np.array([locker[:2] for locker in lockers])
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    diameter = euclidean_distance((min_x, min_y), (max_x, max_y))
    circle_area = math.pi * (diameter / 2) ** 2

    # 计算点集凸包面积
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        convex_area = hull.volume
    except:
        # 如果无法计算凸包，使用边界矩形面积
        convex_area = (max_x - min_x) * (max_y - min_y)

    return convex_area / circle_area if circle_area > 0 else 1.0


def calculate_ldp(lockers, center):
    """计算负载-距离乘积 (Load-Distance Product)"""
    ldp = 0
    for locker in lockers:
        pos = locker[:2]
        delivery = locker[2]
        return_d = locker[3]
        demand = delivery + return_d
        dist = euclidean_distance(pos, center)
        ldp += demand * dist
    return ldp


def calculate_ic(lockers):
    """计算点间关联度 (Inter-point Connectivity)"""
    n = len(lockers)
    if n < 2:
        return 0

    total_min_dist = 0
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = euclidean_distance(lockers[i][:2], lockers[j][:2])
                if dist < min_dist:
                    min_dist = dist
        total_min_dist += min_dist
    return total_min_dist / n


def calculate_dle(lockers, center):
    """计算距离-负载效率 (Distance-Load Efficiency)"""
    total_demand = sum(locker[2] + locker[3] for locker in lockers)
    ldp = calculate_ldp(lockers, center)
    return ldp / total_demand if total_demand > 0 else 0


def calculate_dii(lockers):
    """计算分散-失衡指数 (Dispersion-Imbalance Index)"""
    sd = calculate_sd(lockers)
    di = calculate_di(lockers)
    return sd * di


def calculate_cci(lockers, center):
    """计算中心-关联指数 (Central-Connectivity Index)"""
    cp = calculate_cp(lockers, center)
    ic = calculate_ic(lockers)
    return cp / ic if ic > 0 else cp


def calculate_cli(lockers, center):
    """计算簇聚-负载指数 (Cluster-Load Index)"""
    ci = calculate_ci(lockers)
    ldp = calculate_ldp(lockers, center)
    return ci * ldp


def calculate_features_for_scene(scene_data, center=(0, 0)):
    """计算单个场景的所有特征"""
    # 提取快递柜数据
    lockers = []
    for i in range(1, config.MAX_LOCKERS + 1):
        x = scene_data[f'locker_{i}_x']
        y = scene_data[f'locker_{i}_y']
        delivery = scene_data[f'locker_{i}_delivery']
        return_d = scene_data[f'locker_{i}_return']

        if not pd.isnull(x) and not pd.isnull(y):
            lockers.append((x, y, delivery, return_d))

    # 如果没有快递柜，返回空特征
    if not lockers:
        return {}

    # 计算所有特征
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

    # 计算FDCR指标并添加到特征字典
    features['FDCR'] = calculate_fdcr(scene_data, center)

    return features

def calculate_optimized_cost_for_scene(scene_data, center=(0, 0), drone_num=4, max_range=30, speed=1.0):
    """计算单个场景的优化配送成本"""
    # 创建快递柜管理器
    manager = LockersManager()

    # 添加快递柜
    for i in range(1, config.MAX_LOCKERS + 1):
        x = scene_data[f'locker_{i}_x']
        y = scene_data[f'locker_{i}_y']
        delivery = scene_data[f'locker_{i}_delivery']
        return_d = scene_data[f'locker_{i}_return']

        if not pd.isnull(x) and not pd.isnull(y):
            position = (x, y)
            manager.add_locker(position, delivery, return_d)

    # 如果没有快递柜，返回0
    if not manager.lockers:
        return 0.0

    # 分类需求
    manager.classify_demands()

    # 初始化无人机
    drones = [Drone(i, center, max_range=max_range) for i in range(drone_num)]

    # 处理强制配送阶段
    forced_cost = process_forced_deliveries(drones, manager, center, speed)

    # 处理优化配送阶段
    optimized_cost = process_optimized_deliveries(drones, manager, center, speed)

    return optimized_cost


def main():
    # 加载数据集
    print(f"加载数据集: {config.DATASET_FILE}")
    df = pd.read_excel(config.DATASET_FILE)

    # 创建结果数据框 - 添加FDCR列
    result_columns = ['scene_id', 'optimized_cost', 'FDCR',
                      'CP', 'DI', 'SD', 'CI', 'LDP', 'IC', 'DLE', 'DII', 'CCI', 'CLI']
    result_df = pd.DataFrame(columns=result_columns)

    # 处理每个场景
    print(f"开始处理 {len(df)} 个场景...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        scene_id = row['scene_id']

        # 计算优化配送成本
        optimized_cost = calculate_optimized_cost_for_scene(
            row,
            center=config.CENTER,
            drone_num=config.DRONE_NUM,
            max_range=config.DRONE_MAX_RANGE,
            speed=config.SPEED
        )

        # 计算特征 - 包含FDCR
        features = calculate_features_for_scene(row, center=config.CENTER)

        # 添加到结果数据框
        result_row = {
            'scene_id': scene_id,
            'optimized_cost': optimized_cost,
            **features
        }
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)

    # 保存结果
    output_file = config.OUTPUT_FILE
    result_df.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 打印统计信息
    print("\n统计信息:")
    print(f"- 场景数量: {len(result_df)}")
    print(f"- 平均优化成本: {result_df['optimized_cost'].mean():.2f}")
    print(f"- 最小优化成本: {result_df['optimized_cost'].min():.2f}")
    print(f"- 最大优化成本: {result_df['optimized_cost'].max():.2f}")
    print(f"- 平均FDCR: {result_df['FDCR'].mean():.2f}")


if __name__ == "__main__":
    # 配置参数
    config.DRONE_NUM = 4  # 无人机数量
    config.DRONE_MAX_RANGE = 30  # 无人机最大续航
    config.SPEED = 1.0  # 无人机速度
    config.CENTER = (0, 0)  # 中心点坐标
    config.MAX_LOCKERS = 10  # 最大快递柜数量
    config.DATASET_FILE = "locker_dataset.xlsx"  # 输入数据集文件
    config.OUTPUT_FILE = "features_and_costs.xlsx"  # 输出文件

    main()