# config.py - 所有可配置参数
import numpy as np
import random
import pandas as pd
import time

# 使用时间戳作为动态种子，确保每次运行都有不同的随机性
current_time = int(time.time())
random.seed(current_time)
np.random.seed(current_time)

# 快递柜信息生成参数
num_lockers = 30  # 快递柜数量（训练使用30个）
boundary = 100   # 地图边界调整为200x200（-100到100）
min_lambda = 1  # 需求lambda最小值（泊松分布参数）
max_lambda = 4  # 需求lambda最大值（泊松分布参数）

# 环境配置参数
demand_variance = 0.5  # 需求方差
time_pressure = 1.0     # 时间压力

lockers_x, lockers_y = {}, {}
for i in range(1, num_lockers + 1):
    x = round(random.uniform(-boundary, boundary), 2)  # 保留 2 位小数，可按需调整
    y = round(random.uniform(-boundary, boundary), 2)
    lockers_x[str(i)] = x
    lockers_y[str(i)] = y

lambda_del, lambda_ret = {}, {}
for i in range(1, num_lockers + 1):
    x = round(random.uniform(min_lambda, max_lambda), 2)  # 保留 2 位小数，可按需调整
    y = round(random.uniform(min_lambda, max_lambda), 2)
    lambda_del[str(i)] = x
    lambda_ret[str(i)] = y

demand_del = {
    k: int(np.random.poisson(lam=v))   # lam=v 表示以 v 为 λ 的泊松分布
    for k, v in lambda_del.items()}
demand_ret = {
    k: int(np.random.poisson(lam=v))   # lam=v 表示以 v 为 λ 的泊松分布
    for k, v in lambda_ret.items()}

df = pd.DataFrame({
    "序号": list(range(1, num_lockers + 1)),
    "lockers_x": [lockers_x[str(i)] for i in range(1, num_lockers + 1)],
    "lockers_y": [lockers_y[str(i)] for i in range(1, num_lockers + 1)],
    "lamda_del": [lambda_del[str(i)] for i in range(1, num_lockers + 1)],
    "lamda_ret": [lambda_ret[str(i)] for i in range(1, num_lockers + 1)],
    "demand_del": [demand_del[str(i)] for i in range(1, num_lockers + 1)],
    "demand_ret": [demand_ret[str(i)] for i in range(1, num_lockers + 1)]
})
df.to_csv("locker_data.csv", index=False, encoding="utf-8-sig")  # 保存为 CSV
df = pd.read_csv('locker_data.csv')
locker_info = [list(row) for row in zip(*df.iloc[:, 1:].values.tolist())]

def generate_locker_info():
    """
    重新生成快递柜信息，每次调用都使用新的随机种子确保数据多样性
    """
    global locker_info, lockers_x, lockers_y, lambda_del, lambda_ret, demand_del, demand_ret, df
    
    # 使用当前时间戳加上随机数作为种子，确保每次调用都不同
    new_seed = int(time.time() * 1000) % 2147483647 + random.randint(0, 1000)
    random.seed(new_seed)
    np.random.seed(new_seed)
    
    # 重新生成快递柜位置
    lockers_x, lockers_y = {}, {}
    for i in range(1, num_lockers + 1):
        x = round(random.uniform(-boundary, boundary), 2)
        y = round(random.uniform(-boundary, boundary), 2)
        lockers_x[str(i)] = x
        lockers_y[str(i)] = y

    # 重新生成lambda参数
    lambda_del, lambda_ret = {}, {}
    for i in range(1, num_lockers + 1):
        x = round(random.uniform(min_lambda, max_lambda), 2)
        y = round(random.uniform(min_lambda, max_lambda), 2)
        lambda_del[str(i)] = x
        lambda_ret[str(i)] = y

    # 重新生成需求
    demand_del = {
        k: int(np.random.poisson(lam=v))
        for k, v in lambda_del.items()}
    demand_ret = {
        k: int(np.random.poisson(lam=v))
        for k, v in lambda_ret.items()}

    # 重新生成DataFrame和locker_info
    df = pd.DataFrame({
        "序号": list(range(1, num_lockers + 1)),
        "lockers_x": [lockers_x[str(i)] for i in range(1, num_lockers + 1)],
        "lockers_y": [lockers_y[str(i)] for i in range(1, num_lockers + 1)],
        "lamda_del": [lambda_del[str(i)] for i in range(1, num_lockers + 1)],
        "lamda_ret": [lambda_ret[str(i)] for i in range(1, num_lockers + 1)],
        "demand_del": [demand_del[str(i)] for i in range(1, num_lockers + 1)],
        "demand_ret": [demand_ret[str(i)] for i in range(1, num_lockers + 1)]
    })
    df.to_csv("locker_data.csv", index=False, encoding="utf-8-sig")
    
    # 更新locker_info
    locker_info = [list(row) for row in zip(*df.iloc[:, 1:].values.tolist())]
    
    # 更新Config类中的locker_info
    Config.locker_info = locker_info

def generate_demand_only():
    """
    只重新生成需求分布，保持快递柜位置不变
    每个episode为每个快递柜生成lambda在1-4之间的泊松分布需求
    """
    global lambda_del, lambda_ret, demand_del, demand_ret
    
    # 使用当前时间戳加上随机数作为种子，确保每次调用都不同
    new_seed = int(time.time() * 1000) % 2147483647 + random.randint(0, 1000)
    random.seed(new_seed)
    np.random.seed(new_seed)
    
    # 重新生成lambda参数（1-4之间）
    lambda_del, lambda_ret = {}, {}
    for i in range(1, num_lockers + 1):
        x = round(random.uniform(min_lambda, max_lambda), 2)
        y = round(random.uniform(min_lambda, max_lambda), 2)
        lambda_del[str(i)] = x
        lambda_ret[str(i)] = y

    # 重新生成需求（基于新的lambda参数的泊松分布）
    demand_del = {
        k: int(np.random.poisson(lam=v))
        for k, v in lambda_del.items()}
    demand_ret = {
        k: int(np.random.poisson(lam=v))
        for k, v in lambda_ret.items()}

# 模块级别的配置参数（用于向后兼容）
DEPOT = (0, 0)
CENTER = (0, 0)  # 中心点坐标，与DEPOT相同
DRONE_MAX_RANGE = 50  # 无人机航程，单程50（以卡车为中心进行配送）
TRUCK_CAPACITY = 100
PENALTY_WEIGHT = 0.1
MAX_TIMESTEPS = 600  # 保持合理的最大步数，通过优化时间参数提高效率

# 数据集生成参数（模块级别，用于向后兼容）
DATASET_SIZE = 1000
MIN_LOCKERS = 2
MAX_LOCKERS = 10
DEMAND_MIN = 1
DEMAND_MAX = 10
OUTPUT_FILE = "locker_dataset.xlsx"

# MAPPO训练参数 - 优化GPU利用率
TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 3e-4
BATCH_SIZE = 256  # 增加批次大小，提升GPU利用率
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95

class Config:
    """配置类 - 包含所有环境运行时的配置参数"""
    
    # 无人机配送系统参数
    DRONE_NUM = 3   # 每辆卡车的无人机数量
    DRONE_MAX_RANGE = 50  # 无人机航程，单程50（以卡车为中心进行配送）
    DRONE_SPEED = 1.0  # 无人机飞行速度
    DRONE_SERVICE_TIME = 2  # 无人机每个需求服务时间（秒）
    DEPOT = (0, 0)  # 仓库位置

    # 卡车配送系统参数
    TRUCK_CAPACITY = 100  # 卡车容量
    TRUCK_SERVICE_TIME = 60  # 卡车服务时间（秒）
    TRUCK_SPEED = 20  # 卡车移动速度（单位/时间）

    # 环境参数
    PENALTY_WEIGHT = 0.1  # 惩罚权重
    MAX_TIMESTEPS = 600  # 最大时间步数
    
    # 快递柜数据（动态生成）
    locker_info = [list(row) for row in zip(*df.iloc[:, 1:].values.tolist())]
