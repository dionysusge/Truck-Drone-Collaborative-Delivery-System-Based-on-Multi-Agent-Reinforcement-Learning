#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行动画生成器
作者: Dionysus
联系方式: wechat:gzw1546484791
"""

import os
import sys
from animation_generator import MovementAnimationGenerator
from datetime import datetime

def main():
    """
    主函数 - 运行动画生成
    """
    print("=== 卡车和无人机移动动画生成器 ===")
    
    # 设置测试结果目录
    test_results_dir = "test_results"
    
    if not os.path.exists(test_results_dir):
        print(f"错误: 测试结果目录不存在: {test_results_dir}")
        print("请先运行测试生成测试结果数据")
        return False
    
    # 检查必要文件
    required_files = [
        'detailed_test_data.json',
        'environment_initialization.json',
        'detailed_test_report.txt'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(test_results_dir, file_name)
        if not os.path.exists(file_path):
            print(f"错误: 缺少必要文件: {file_path}")
            return False
    
    print("✓ 所有必要文件存在")
    
    # 创建动画生成器
    print("正在初始化动画生成器...")
    generator = MovementAnimationGenerator(test_results_dir)
    
    # 加载数据
    print("正在加载测试数据...")
    if not generator.load_data():
        print("错误: 数据加载失败")
        return False
    
    print("✓ 数据加载成功")
    print(f"  - 卡车数量: {len(generator.truck_data)}")
    print(f"  - 快递柜数量: {len(generator.locker_positions)}")
    print(f"  - 无人机步骤数据: {len(generator.drone_data)}")
    
    # 验证数据
    if not generator.truck_data:
        print("\n⚠️  警告: 没有卡车数据，无法生成动画")
        print("请先运行测试脚本生成测试数据:")
        print("  python show_detail/test_environment.py")
        return False
    
    # 检查是否有有效的步骤数据
    has_valid_data = False
    for truck_id, truck_info in generator.truck_data.items():
        if truck_info.get('positions') and len(truck_info['positions']) > 0:
            has_valid_data = True
            break
    
    if not has_valid_data:
        print("\n⚠️  警告: 没有有效的移动数据，无法生成动画")
        print("测试数据中可能没有记录卡车的移动轨迹")
        return False
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成静态总览图
    print("\n正在生成静态总览图...")
    overview_path = f"movement_overview_{timestamp}.png"
    try:
        generator.generate_static_overview(overview_path)
        print(f"✓ 静态总览图已保存: {overview_path}")
    except Exception as e:
        print(f"错误: 静态总览图生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成动画
    print("\n正在生成动画（这可能需要一些时间）...")
    animation_path = f"movement_animation_{timestamp}.gif"
    
    try:
        anim = generator.create_animation(animation_path, interval=1500)
        print(f"✓ 动画已保存: {animation_path}")
        
        # 显示动画信息
        print("\n=== 动画信息 ===")
        print(f"动画文件: {animation_path}")
        print(f"总览图: {overview_path}")
        print("动画包含:")
        for truck_id in generator.truck_data.keys():
            positions = generator.truck_data[truck_id].get('positions', [])
            steps = len(positions)
            print(f"  - 卡车{truck_id}: {steps}个移动步骤")
        
        return True
        
    except Exception as e:
        print(f"错误: 动画生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 动画生成完成!")
    else:
        print("\n❌ 动画生成失败")
        sys.exit(1)