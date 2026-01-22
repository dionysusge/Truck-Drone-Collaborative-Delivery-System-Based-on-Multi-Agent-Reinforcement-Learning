import numpy as np
import pandas as pd
import math
import config
import os
import random
from openpyxl import Workbook


def euclidean_distance(p1, p2):
    """计算两点间欧氏距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_random_locker(max_range=config.DRONE_MAX_RANGE):
    """
    生成随机快递柜位置，确保往返距离在续航范围内
    """
    while True:
        # 在半径为 max_range/2 的圆内生成位置
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, max_range / 2)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pos = (x, y)

        # 检查距离中心点的距离是否满足续航约束
        dist_to_center = euclidean_distance(pos, config.CENTER)
        if 2 * dist_to_center <= max_range:
            return pos


def generate_dataset():
    """生成随机数据集"""
    # 创建空列表存储所有场景数据
    all_scene_data = []

    # 生成每个场景
    for scene_id in range(config.DATASET_SIZE):
        # 中心点坐标（固定为(0,0)）
        center_x, center_y = config.CENTER

        # 随机确定快递柜数量
        num_lockers = np.random.randint(config.MIN_LOCKERS, config.MAX_LOCKERS + 1)

        # 创建场景数据字典
        scene_data = {
            'scene_id': scene_id,
            'center_x': center_x,
            'center_y': center_y
        }

        # 生成快递柜位置和需求
        for i in range(config.MAX_LOCKERS):
            if i < num_lockers:
                # 生成位置
                pos = generate_random_locker()

                # 生成需求
                delivery_demand = np.random.randint(config.DEMAND_MIN, config.DEMAND_MAX + 1)
                return_demand = np.random.randint(config.DEMAND_MIN, config.DEMAND_MAX + 1)

                # 添加到场景数据
                scene_data[f'locker_{i + 1}_x'] = pos[0]
                scene_data[f'locker_{i + 1}_y'] = pos[1]
                scene_data[f'locker_{i + 1}_delivery'] = delivery_demand
                scene_data[f'locker_{i + 1}_return'] = return_demand
            else:
                # 没有快递柜的位置，设为None
                scene_data[f'locker_{i + 1}_x'] = None
                scene_data[f'locker_{i + 1}_y'] = None
                scene_data[f'locker_{i + 1}_delivery'] = None
                scene_data[f'locker_{i + 1}_return'] = None

        # 添加到场景列表
        all_scene_data.append(scene_data)

    # 一次性创建DataFrame
    df = pd.DataFrame(all_scene_data)
    return df


def save_dataset(df, filename):
    """保存数据集到Excel文件"""
    # 获取目录路径
    dir_path = os.path.dirname(filename)

    # 如果目录路径不为空，确保目录存在
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 创建Excel工作簿
    wb = Workbook()

    # 使用活动工作表
    ws = wb.active
    ws.title = "场景数据"

    # 添加数据标题行
    headers = list(df.columns)
    ws.append(headers)

    # 添加数据行
    for _, row in df.iterrows():
        ws.append([row[col] for col in headers])

    # 保存工作簿
    wb.save(filename)
    print(f"数据集已保存到 {filename}")


def main():
    print("开始生成数据集...")
    print(f"配置参数:")
    print(f"- 数据集大小: {config.DATASET_SIZE}")
    print(f"- 快递柜数量范围: {config.MIN_LOCKERS} 到 {config.MAX_LOCKERS}")
    print(f"- 无人机最大续航: {config.DRONE_MAX_RANGE}")
    print(f"- 中心点坐标: {config.CENTER}")
    print(f"- 需求范围: {config.DEMAND_MIN} 到 {config.DEMAND_MAX}")

    # 生成数据集
    df = generate_dataset()

    # 保存数据集
    save_dataset(df, config.OUTPUT_FILE)

    print("\n数据集生成完成!")
    print(f"共生成 {len(df)} 个场景")

    # 计算平均快递柜数量
    locker_cols = [col for col in df.columns if col.startswith('locker_') and col.endswith('_x')]
    avg_lockers = df[locker_cols].notnull().sum(axis=1).mean()
    print(f"平均每个场景快递柜数量: {avg_lockers:.2f}")


if __name__ == "__main__":
    main()