#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 加载训练好的模型进行测试
作者: Dionysus
联系方式: wechat:gzw1546484791

功能:
- 加载训练好的MAPPO模型
- 运行多个测试回合
- 显示测试结果和性能指标
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any
import argparse

# 导入必要的模块
from truck_routing import TruckSchedulingEnv, MAPPO
import config


def infer_env_config_from_model(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    从模型checkpoint中推断训练时的环境配置
    
    Args:
        checkpoint: 模型checkpoint字典
        
    Returns:
        环境配置字典，包含 num_lockers, state_dim, num_trucks 等信息
    """
    config = {}
    
    # 获取策略网络的状态字典
    policy_state_dict = None
    if 'policy_net_state_dict' in checkpoint:
        policy_state_dict = checkpoint['policy_net_state_dict']
    elif 'policy_net' in checkpoint:
        policy_state_dict = checkpoint['policy_net']
    
    if policy_state_dict is not None:
        # 从 stop_head 推断快递柜数量
        # stop_head 输出维度 = num_lockers + 1 (仓库)
        if 'stop_head.7.weight' in policy_state_dict:
            stop_output_dim = policy_state_dict['stop_head.7.weight'].shape[0]
            config['num_lockers'] = stop_output_dim - 1
            print(f"🔍 从模型结构推断: 快递柜数量 = {config['num_lockers']} (stop_head输出维度: {stop_output_dim})")
        
        # 从 service_head 验证快递柜数量
        if 'service_head.7.weight' in policy_state_dict:
            service_output_dim = policy_state_dict['service_head.7.weight'].shape[0]
            inferred_lockers = service_output_dim
            if 'num_lockers' in config:
                if config['num_lockers'] != inferred_lockers:
                    print(f"⚠️  警告: stop_head推断的快递柜数量({config['num_lockers']})与service_head推断的({inferred_lockers})不一致")
                    # 使用stop_head的结果（更准确，因为包含仓库）
            else:
                config['num_lockers'] = inferred_lockers
                print(f"🔍 从模型结构推断: 快递柜数量 = {config['num_lockers']} (service_head输出维度: {service_output_dim})")
        
        # 从 state_encoder 推断状态维度
        if 'state_encoder.0.weight' in policy_state_dict:
            state_dim = policy_state_dict['state_encoder.0.weight'].shape[1]
            config['state_dim'] = state_dim
            print(f"🔍 从模型结构推断: 状态维度 = {state_dim}")
    
    # 从checkpoint中读取保存的配置信息
    if 'num_trucks' in checkpoint:
        config['num_trucks'] = checkpoint['num_trucks']
        print(f"📋 从checkpoint读取: 卡车数量 = {config['num_trucks']}")
    
    return config


def create_env_from_config(env_config: Dict[str, Any]) -> TruckSchedulingEnv:
    """
    根据配置创建环境
    
    Args:
        env_config: 环境配置字典
        
    Returns:
        配置好的环境实例
    """
    # 创建环境
    env = TruckSchedulingEnv(verbose=False)
    
    # 如果配置中指定了快递柜数量，更新环境配置
    if 'num_lockers' in env_config or 'num_trucks' in env_config:
        num_lockers = env_config.get('num_lockers', None)
        num_trucks = env_config.get('num_trucks', None)
        
        if num_lockers is not None:
            print(f"🔧 配置环境: 快递柜数量 = {num_lockers}")
        if num_trucks is not None:
            print(f"🔧 配置环境: 卡车数量 = {num_trucks}")
        
        # 使用 update_curriculum_config 更新环境（使用config中的值）
        curriculum_config = {
            'boundary': config.boundary,
            'demand_variance': config.demand_variance,
            'time_pressure': config.time_pressure
        }
        
        if num_lockers is not None:
            curriculum_config['num_lockers'] = num_lockers
        if num_trucks is not None:
            curriculum_config['num_trucks'] = num_trucks
        
        if hasattr(env, 'update_curriculum_config'):
            env.update_curriculum_config(curriculum_config)
        else:
            # 如果方法不存在，直接修改配置
            # import config # Removed to avoid UnboundLocalError
            if num_lockers is not None:
                config.num_lockers = num_lockers
                config.generate_locker_info()
                env.num_lockers = num_lockers
                env.lockers_info = config.locker_info
            if num_trucks is not None:
                env.num_trucks = num_trucks
    
    # 重置环境以初始化所有内部状态（包括卡车、性能指标等）
    try:
        env.reset()
    except Exception as e:
        print(f"⚠️  环境重置时出现警告: {e}")
        # 继续执行，环境可能已经部分初始化
    
    return env


def load_model(model_path: str, env: TruckSchedulingEnv = None) -> tuple:
    """
    加载训练好的模型，自动推断环境配置
    
    Args:
        model_path: 模型文件路径
        env: 可选的环境实例，如果为None则从模型推断配置
        
    Returns:
        (mappo, env) 元组，包含加载好的MAPPO模型和配置好的环境
    """
    print(f"📂 正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载checkpoint
    try:
        # 兼容新版本PyTorch的weights_only参数
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"加载模型文件失败: {e}")
    
    # 从模型推断环境配置
    inferred_config = infer_env_config_from_model(checkpoint)
    
    # 创建或更新环境
    if env is None:
        # 从模型推断配置创建环境
        env = create_env_from_config(inferred_config)
    else:
        # 检查环境配置是否匹配
        need_update = False
        if 'num_lockers' in inferred_config:
            if env.num_lockers != inferred_config['num_lockers']:
                print(f"⚠️  环境配置不匹配: 当前环境有 {env.num_lockers} 个快递柜，但模型是为 {inferred_config['num_lockers']} 个快递柜训练的")
                need_update = True
        if 'num_trucks' in inferred_config:
            if env.num_trucks != inferred_config['num_trucks']:
                print(f"⚠️  环境配置不匹配: 当前环境有 {env.num_trucks} 辆卡车，但模型是为 {inferred_config['num_trucks']} 辆卡车训练的")
                need_update = True
        
        if need_update:
            print(f"   正在更新环境配置以匹配模型...")
            env = create_env_from_config(inferred_config)
    
    # 确保环境已正确初始化（重置环境）
    try:
        env.reset()
    except Exception as e:
        print(f"⚠️  环境重置时出现警告: {e}")
    
    # 获取环境参数
    num_trucks = env.num_trucks
    
    # 获取状态维度（需要先重置环境）
    try:
        truck_states = env.get_truck_specific_states()
        state_dim = len(truck_states[0]) if truck_states else env.state_dim
    except Exception as e:
        # 如果获取状态失败，尝试使用推断的状态维度
        print(f"⚠️  获取环境状态时出错: {e}")
        if 'state_dim' in inferred_config:
            state_dim = inferred_config['state_dim']
            print(f"   使用从模型推断的状态维度: {state_dim}")
        else:
            # 最后尝试使用环境的state_dim属性
            state_dim = getattr(env, 'state_dim', 422)
            print(f"   使用环境默认状态维度: {state_dim}")
    
    # 如果从模型推断出了状态维度，使用推断的值
    if 'state_dim' in inferred_config:
        inferred_state_dim = inferred_config['state_dim']
        if state_dim != inferred_state_dim:
            print(f"⚠️  状态维度不匹配: 当前环境状态维度为 {state_dim}，但模型期望 {inferred_state_dim}")
            print(f"   使用模型期望的状态维度: {inferred_state_dim}")
            state_dim = inferred_state_dim
    
    action_dim = {
        "select_stop": env.num_lockers + 1,  # 0:仓库, 1-n:快递柜
        "service_area": env.num_lockers  # 每个快递柜一个二进制选择
    }
    
    print(f"\n🔧 最终环境配置:")
    print(f"   - 卡车数量: {num_trucks}")
    print(f"   - 快递柜数量: {env.num_lockers}")
    print(f"   - 状态维度: {state_dim}")
    print(f"   - 动作维度: {action_dim}")
    
    # 创建MAPPO实例
    mappo = MAPPO(num_trucks, state_dim, action_dim, lr=config.LEARNING_RATE)
    
    # 加载模型权重
    try:
        # 检查checkpoint的结构
        if isinstance(checkpoint, dict):
            # 检查不同的键名格式
            if 'policy_net_state_dict' in checkpoint and 'value_net_state_dict' in checkpoint:
                mappo.policy_net.load_state_dict(checkpoint['policy_net_state_dict'], strict=False)
                mappo.value_net.load_state_dict(checkpoint['value_net_state_dict'], strict=False)
                print("✅ 成功加载模型权重 (policy_net_state_dict, value_net_state_dict)")
            elif 'policy_net' in checkpoint and 'value_net' in checkpoint:
                mappo.policy_net.load_state_dict(checkpoint['policy_net'], strict=False)
                mappo.value_net.load_state_dict(checkpoint['value_net'], strict=False)
                print("✅ 成功加载模型权重 (policy_net, value_net)")
            else:
                raise ValueError(f"模型文件格式不匹配，可用键: {list(checkpoint.keys())}")
        else:
            raise ValueError("模型文件格式不正确")
        
        # 设置模型为评估模式
        mappo.policy_net.eval()
        mappo.value_net.eval()
        
        print(f"✅ 模型加载成功!")
        
        # 显示模型信息（如果有）
        if 'num_trucks' in checkpoint:
            print(f"   - 模型训练时的卡车数量: {checkpoint['num_trucks']}")
        if 'episode' in checkpoint:
            print(f"   - 模型训练轮数: {checkpoint['episode']}")
        if 'best_avg_reward' in checkpoint:
            print(f"   - 模型最佳平均奖励: {checkpoint['best_avg_reward']:.2f}")
        if 'episode_reward' in checkpoint:
            print(f"   - 模型保存时的奖励: {checkpoint['episode_reward']:.2f}")
            
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")
    
    return mappo, env


def run_test_episode(env: TruckSchedulingEnv, mappo: MAPPO, episode_num: int, verbose: bool = True) -> Dict[str, Any]:
    """
    运行一个测试回合
    
    Args:
        env: 环境实例
        mappo: MAPPO模型
        episode_num: 回合编号
        verbose: 是否显示详细信息
        
    Returns:
        包含测试结果的字典
    """
    # 重置环境
    state, action_mask = env.reset()
    episode_reward = 0.0
    step_count = 0
    done = False
    
    # 记录每步的奖励
    step_rewards = []
    
    if verbose:
        print(f"\n🎮 开始测试回合 {episode_num + 1}")
    
    while not done and step_count < env.max_timesteps:
        # 获取动作掩码
        action_masks = env.get_action_masks()
        
        # 获取每个卡车的特定状态
        truck_states = env.get_truck_specific_states()
        
        # 使用模型选择动作（不添加探索噪声）
        with torch.no_grad():
            actions, log_probs, values = mappo.act(truck_states, action_masks, env)
        
        # 执行动作
        next_state, rewards, done, next_action_mask = env.step(actions)
        
        # 累积奖励
        if isinstance(rewards, list):
            step_reward = sum(rewards)
            episode_reward += step_reward
            step_rewards.append(step_reward)
        else:
            episode_reward += rewards
            step_rewards.append(rewards)
        
        # 更新状态
        state = next_state
        action_mask = next_action_mask
        step_count += 1
    
    # 计算性能指标
    completion_rate = env._calculate_completion_rate()
    path_efficiency = env._calculate_path_efficiency()
    capacity_utilization = env._calculate_capacity_utilization()
    
    # 计算总需求和服务情况
    total_demand_del = sum(locker.get('demand_del', 0) for locker in env.lockers_state)
    total_demand_ret = sum(locker.get('demand_ret', 0) for locker in env.lockers_state)
    total_demand = total_demand_del + total_demand_ret
    
    total_served_del = sum(locker.get('served_demand_del', 0) for locker in env.lockers_state)
    total_served_ret = sum(locker.get('served_demand_ret', 0) for locker in env.lockers_state)
    total_served = total_served_del + total_served_ret
    
    served_rate = (total_served / total_demand * 100) if total_demand > 0 else 0.0
    
    result = {
        'episode': episode_num + 1,
        'episode_reward': episode_reward,
        'step_count': step_count,
        'completion_rate': completion_rate,
        'path_efficiency': path_efficiency,
        'capacity_utilization': capacity_utilization,
        'total_demand': total_demand,
        'total_served': total_served,
        'served_rate': served_rate,
        'avg_step_reward': np.mean(step_rewards) if step_rewards else 0.0,
        'max_step_reward': np.max(step_rewards) if step_rewards else 0.0,
        'min_step_reward': np.min(step_rewards) if step_rewards else 0.0
    }
    
    if verbose:
        print(f"   ✅ 回合完成")
        print(f"      - 总奖励: {episode_reward:.2f}")
        print(f"      - 步数: {step_count}")
        print(f"      - 完成率: {completion_rate:.2f}%")
        print(f"      - 路径效率: {path_efficiency:.2f}%")
        print(f"      - 容量利用率: {capacity_utilization:.2f}%")
        print(f"      - 服务率: {served_rate:.2f}% ({total_served}/{total_demand})")
    
    return result


def run_tests(model_path: str = "trained_mappo_policy.pth", num_episodes: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """
    运行测试
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试回合数
        verbose: 是否显示详细信息
        
    Returns:
        测试结果汇总
    """
    print("=" * 60)
    print("🧪 MAPPO模型快速测试")
    print("=" * 60)
    
    # 加载模型（会自动推断环境配置）
    print("\n📂 加载模型并推断环境配置...")
    try:
        mappo, env = load_model(model_path, env=None)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}
    
    print(f"\n📊 最终环境配置:")
    print(f"   - 卡车数量: {env.num_trucks}")
    print(f"   - 快递柜数量: {env.num_lockers}")
    print(f"   - 卡车容量: {env.truck_capacity}")
    print(f"   - 最大步数: {env.max_timesteps}")
    print(f"   - 无人机航程: {config.DRONE_MAX_RANGE}")
    
    # 运行测试
    print(f"\n🚀 开始运行 {num_episodes} 个测试回合...")
    print("=" * 60)
    
    test_results = []
    for i in range(num_episodes):
        result = run_test_episode(env, mappo, i, verbose=verbose)
        test_results.append(result)
    
    # 计算统计信息
    episode_rewards = [r['episode_reward'] for r in test_results]
    completion_rates = [r['completion_rate'] for r in test_results]
    path_efficiencies = [r['path_efficiency'] for r in test_results]
    capacity_utilizations = [r['capacity_utilization'] for r in test_results]
    served_rates = [r['served_rate'] for r in test_results]
    step_counts = [r['step_count'] for r in test_results]
    
    summary = {
        'status': 'success',
        'num_episodes': num_episodes,
        'episode_rewards': {
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards)
        },
        'completion_rates': {
            'mean': np.mean(completion_rates),
            'std': np.std(completion_rates),
            'min': np.min(completion_rates),
            'max': np.max(completion_rates)
        },
        'path_efficiencies': {
            'mean': np.mean(path_efficiencies),
            'std': np.std(path_efficiencies),
            'min': np.min(path_efficiencies),
            'max': np.max(path_efficiencies)
        },
        'capacity_utilizations': {
            'mean': np.mean(capacity_utilizations),
            'std': np.std(capacity_utilizations),
            'min': np.min(capacity_utilizations),
            'max': np.max(capacity_utilizations)
        },
        'served_rates': {
            'mean': np.mean(served_rates),
            'std': np.std(served_rates),
            'min': np.min(served_rates),
            'max': np.max(served_rates)
        },
        'step_counts': {
            'mean': np.mean(step_counts),
            'std': np.std(step_counts),
            'min': np.min(step_counts),
            'max': np.max(step_counts)
        },
        'detailed_results': test_results
    }
    
    # 显示测试结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"\n🎯 奖励统计 ({num_episodes} 回合):")
    print(f"   - 平均奖励: {summary['episode_rewards']['mean']:.2f} ± {summary['episode_rewards']['std']:.2f}")
    print(f"   - 最佳奖励: {summary['episode_rewards']['max']:.2f}")
    print(f"   - 最差奖励: {summary['episode_rewards']['min']:.2f}")
    
    print(f"\n📈 性能指标:")
    print(f"   - 平均完成率: {summary['completion_rates']['mean']:.2f}% ± {summary['completion_rates']['std']:.2f}%")
    print(f"   - 平均路径效率: {summary['path_efficiencies']['mean']:.2f}% ± {summary['path_efficiencies']['std']:.2f}%")
    print(f"   - 平均容量利用率: {summary['capacity_utilizations']['mean']:.2f}% ± {summary['capacity_utilizations']['std']:.2f}%")
    print(f"   - 平均服务率: {summary['served_rates']['mean']:.2f}% ± {summary['served_rates']['std']:.2f}%")
    
    print(f"\n⏱️  步数统计:")
    print(f"   - 平均步数: {summary['step_counts']['mean']:.1f} ± {summary['step_counts']['std']:.1f}")
    print(f"   - 最少步数: {summary['step_counts']['min']}")
    print(f"   - 最多步数: {summary['step_counts']['max']}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)
    
    return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAPPO模型快速测试脚本')
    parser.add_argument('--model', type=str, default='trained_mappo_policy.pth',
                        help='模型文件路径 (默认: trained_mappo_policy.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试回合数 (默认: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，不显示详细信息')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print(f"   请先训练模型或指定正确的模型路径")
        print(f"   可用的模型文件:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"      - {file}")
        sys.exit(1)
    
    # 运行测试
    try:
        summary = run_tests(
            model_path=args.model,
            num_episodes=args.episodes,
            verbose=not args.quiet
        )
        
        if summary['status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

