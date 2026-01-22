# 无人机-卡车路径协同优化系统

基于多智能体强化学习（MAPPO）的无人机-卡车协同配送路径优化系统。

**作者**: Dionysus  
**核心算法**: MAPPO (Multi-Agent Proximal Policy Optimization)

## 📋 项目简介

本项目使用强化学习训练智能体，实现卡车和无人机的协同配送，优化最后一公里配送的路径规划和资源调度。

### 核心功能
- 多智能体协同决策（卡车+无人机）
- 动态路径规划与实时调度
- 智能无人机任务分配
- 自适应训练优化

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 生成训练数据集

```bash
python generate_training_dataset.py
```

这将生成 `locker_dataset.xlsx` 文件，包含随机生成的快递柜位置和需求数据。

### 3. 开始训练

```bash
python start_training.py
```

训练完成后，模型会保存为 `trained_mappo_policy.pth`，训练报告保存为 `training_report.json`。

## 📖 主要功能使用

### 训练模型

直接运行训练脚本：

```bash
python start_training.py
```

训练参数可在 `config.py` 中修改：
- `TOTAL_TIMESTEPS`: 总训练步数
- `LEARNING_RATE`: 学习率
- `BATCH_SIZE`: 批次大小
- `num_episodes`: 训练轮数（在 `start_training.py` 中设置）

### 生成训练数据

```bash
python generate_training_dataset.py
```

生成的数据集参数可在 `config.py` 中配置：
- `DATASET_SIZE`: 数据集大小
- `MIN_LOCKERS` / `MAX_LOCKERS`: 快递柜数量范围
- `DEMAND_MIN` / `DEMAND_MAX`: 需求范围

### 测试环境

```bash
python show_detail/test_environment.py
```

### 生成模型数据集

```bash
python generate_model_dataset.py
```

生成特征和成本数据，用于模型训练和分析。

### 步骤可视化

```bash
python show_detail/run_animation.py
```


## ⚙️ 配置说明

主要配置在 `config.py` 文件中：

### 环境参数
- `num_lockers`: 快递柜数量（默认50）
- `boundary`: 地图边界范围（默认±100）
- `DRONE_MAX_RANGE`: 无人机最大续航（默认50）
- `TRUCK_CAPACITY`: 卡车容量（默认100）

### 训练参数
- `TOTAL_TIMESTEPS`: 总训练步数（默认50000）
- `LEARNING_RATE`: 学习率（默认3e-4）
- `BATCH_SIZE`: 批次大小（默认256）
- `GAMMA`: 折扣因子（默认0.99）

### 奖励函数权重
在 `Config` 类中配置：
- `truck_routing_cost`: 卡车路径成本权重
- `drone_routing_cost`: 无人机路径成本权重
- `serve_reward`: 服务奖励
- `unserved_punishment`: 未服务惩罚

## 📁 主要文件说明

- `config.py`: 系统配置文件，包含所有可配置参数
- `start_training.py`: 训练启动脚本，包含训练管理和优化功能
- `truck_routing.py`: 核心算法模块，包含环境、MAPPO算法和智能体
- `generate_training_dataset.py`: 生成强化学习训练数据集
- `generate_model_dataset.py`: 生成模型训练数据集
- `reward_function.py`: 奖励函数实现
- `dynamic_drone_scheduler.py`: 动态无人机调度器
- `state_representation.py`: 状态表示和特征提取

## 📊 输出文件

训练完成后会生成：
- `trained_mappo_policy.pth`: 训练好的模型
- `training_report.json`: 训练报告（包含性能指标和日志）
- `training_analysis.png`: 训练过程可视化图表
- `loss_analysis.png`: 损失函数分析图表

## 🔧 常见问题

### 训练速度慢
- 减少 `TOTAL_TIMESTEPS` 或 `num_episodes`
- 调整 `BATCH_SIZE` 大小
- 在 `config.py` 中减少 `num_lockers` 数量

### 内存不足
- 减小 `BATCH_SIZE`
- 减少快递柜数量
- 减少训练步数

### 修改训练参数
所有训练相关参数都在 `config.py` 中，修改后重新运行训练即可。

## 📝 注意事项

1. 首次运行会自动生成 `locker_data.csv` 文件
2. 训练过程会显示进度条和实时指标
3. 建议使用 GPU 加速训练（如果可用）
4. 训练时间取决于硬件配置和参数设置

## 📧 联系方式

如有问题，请联系：wechat: gzw1546484791

