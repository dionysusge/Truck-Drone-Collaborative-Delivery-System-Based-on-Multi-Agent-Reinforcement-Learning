import math
import sys


class Drone:
    def __init__(self, drone_id, center, max_range=30):
        self.id = drone_id
        self.position = center
        self.available_time = 0  # 当前可开始新任务的时间
        self.total_distance = 0.0  # 无人机累计飞行距离
        self.max_range = max_range  # 无人机最大飞行距离（续航限制）
        self.log = []  # 日志记录: (开始时间, 离开点, 到达时间, 到达点, 任务类型, 距离)


class LockersManager:
    def __init__(self):
        self.lockers = []  # (位置, 取货需求, 退货需求)
        self.deliver_only = []  # 仅取货需求点
        self.return_only = []  # 仅退货需求点
        self.double_demands = []  # 双需求点

    def add_locker(self, position, delivery_demand, return_demand):
        self.lockers.append((position, delivery_demand, return_demand))

    def classify_demands(self):
        self.deliver_only = []
        self.return_only = []
        self.double_demands = []
        for locker in self.lockers:
            pos, d, r = locker
            if d > 0 and r > 0:
                self.double_demands.append((pos, d, r))
            elif d > 0:
                self.deliver_only.append((pos, d, r))
            elif r > 0:
                self.return_only.append((pos, d, r))

    def update_demand(self, position, demand_type, change):
        for i, locker in enumerate(self.lockers):
            if locker[0] == position:
                pos, d, r = locker
                if demand_type == 'delivery':
                    self.lockers[i] = (pos, d + change, r)
                else:  # 'return'
                    self.lockers[i] = (pos, d, r + change)
                break
        self.classify_demands()


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def realtime_print(message):
    print(message)
    sys.stdout.flush()


def process_forced_deliveries(drones, manager, center, speed):
    """处理强制配送阶段的双需求点"""
    total_forced_cost = 0.0
    while manager.double_demands:
        # 选择当前可最早开始任务的无人机
        drone = min(drones, key=lambda d: d.available_time)

        # 选择最近的双需求点
        best_dist = float('inf')
        best_locker = None
        for locker in manager.double_demands:
            pos, _, _ = locker
            dist = euclidean_distance(center, pos)
            # 检查续航限制
            if dist * 2 <= drone.max_range and dist < best_dist:
                best_dist = dist
                best_locker = locker

        # 如果没有满足续航的点，转到优化阶段
        if best_locker is None:
            realtime_print("续航限制：没有满足续航的双需求点，转入优化配送阶段")
            break

        pos, d, r = best_locker
        combo_dist = best_dist * 2

        # 计算配送时间
        to_time = best_dist / speed
        back_time = best_dist / speed
        total_time = to_time + back_time

        # 记录日志
        start_time = drone.available_time
        leave_time = start_time
        arrive_time = start_time + to_time
        return_time = arrive_time + back_time

        # 更新无人机状态
        drone.available_time = return_time
        drone.total_distance += combo_dist
        drone.log.append((leave_time, center, arrive_time, pos, "强制配送（取货+退货）", best_dist))
        drone.log.append((arrive_time, pos, return_time, center, "返回中心", best_dist))

        # 输出实时信息
        realtime_print(f"无人机[{drone.id}]在时间 {leave_time:.2f} 从配送中心出发")
        realtime_print(f"无人机[{drone.id}]在时间 {arrive_time:.2f} 到达快递柜{pos} (距离:{best_dist:.2f})")
        realtime_print(f"无人机[{drone.id}]在时间 {return_time:.2f} 返回配送中心 (任务总距离:{combo_dist:.2f})")

        # 更新需求
        manager.update_demand(pos, 'delivery', -1)
        manager.update_demand(pos, 'return', -1)

        # 累加总成本
        total_forced_cost += combo_dist

    return total_forced_cost


def find_best_combo_task(drone, manager, center):
    """寻找最优组合任务（中心→A→B→中心）"""
    best_combo = None
    min_combo_dist = float('inf')

    # 遍历所有可能的取货点A和退货点B组合
    for del_locker in manager.deliver_only:
        del_pos, d, _ = del_locker
        if d <= 0: continue

        for ret_locker in manager.return_only:
            ret_pos, _, r = ret_locker
            if r <= 0: continue

            # 计算三段距离
            dist1 = euclidean_distance(center, del_pos)  # 中心 → 取货点
            dist2 = euclidean_distance(del_pos, ret_pos)  # 取货点 → 退货点
            dist3 = euclidean_distance(ret_pos, center)  # 退货点 → 中心
            combo_dist = dist1 + dist2 + dist3

            # 检查条件：
            # 1. 符合续航限制
            # 2. 成本比单独执行两个任务低
            # 3. A和B需求类型不同（已通过遍历保证）
            if (combo_dist <= drone.max_range and
                    combo_dist < (2 * dist1 + 2 * dist3)):

                # 更新最优组合
                if combo_dist < min_combo_dist:
                    min_combo_dist = combo_dist
                    best_combo = (del_pos, ret_pos, dist1, dist2, dist3)

    return best_combo, min_combo_dist


def get_next_task(drone, manager, center):
    """获取下一个任务：优先组合任务，否则按顺序分配单独任务"""
    # 1. 首先尝试寻找最优组合任务
    best_combo, combo_dist = find_best_combo_task(drone, manager, center)
    if best_combo:
        del_pos, ret_pos, d1, d2, d3 = best_combo
        return 'combo', (del_pos, ret_pos, d1, d2, d3), combo_dist

    # 2. 没有组合任务，按顺序分配单独任务
    # 先尝试取货任务
    for locker in manager.deliver_only:
        pos, d, _ = locker
        if d <= 0: continue
        dist = euclidean_distance(center, pos)
        task_dist = 2 * dist
        if task_dist <= drone.max_range:
            return 'deliver_only', (pos, dist), task_dist

    # 再尝试退货任务
    for locker in manager.return_only:
        pos, _, r = locker
        if r <= 0: continue
        dist = euclidean_distance(center, pos)
        task_dist = 2 * dist
        if task_dist <= drone.max_range:
            return 'return_only', (pos, dist), task_dist

    # 没有可用任务
    return None, None, 0


def process_optimized_deliveries(drones, manager, center, speed):
    """处理优化配送阶段的剩余需求，考虑续航限制"""
    total_optimized_cost = 0.0
    while manager.deliver_only or manager.return_only:
        # 选择当前可最早开始任务的无人机
        drones.sort(key=lambda d: d.available_time)
        drone = drones[0]

        # 获取下一个任务
        task_type, task_data, task_distance = get_next_task(drone, manager, center)
        if task_type is None:
            # 尝试其他无人机
            for d in drones[1:]:
                task_type, task_data, task_distance = get_next_task(d, manager, center)
                if task_type is not None:
                    drone = d
                    break

            # 如果还是没有任务，结束
            if task_type is None:
                realtime_print("所有需求已完成或超出续航限制!")
                break

        start_time = drone.available_time
        events = []

        # 处理不同任务类型
        if task_type == 'return_only':
            pos, dist = task_data
            to_time = dist / speed
            events.append((start_time, center, pos, to_time, "退货任务", dist))
            back_time = dist / speed
            events.append((start_time + to_time, pos, center, back_time, "返回中心", dist))
            drone.available_time = start_time + to_time + back_time
            manager.update_demand(pos, 'return', -1)

        elif task_type == 'deliver_only':
            pos, dist = task_data
            to_time = dist / speed
            events.append((start_time, center, pos, to_time, "取货任务", dist))
            back_time = dist / speed
            events.append((start_time + to_time, pos, center, back_time, "返回中心", dist))
            drone.available_time = start_time + to_time + back_time
            manager.update_demand(pos, 'delivery', -1)

        elif task_type == 'combo':
            del_pos, ret_pos, d1, d2, d3 = task_data
            t1 = d1 / speed  # 到取货点时间
            t2 = d2 / speed  # 取货点到退货点时间
            t3 = d3 / speed  # 退货点到中心时间

            events.append((start_time, center, del_pos, t1, "取货任务", d1))
            events.append((start_time + t1, del_pos, ret_pos, t2, "前往退货点", d2))
            events.append((start_time + t1 + t2, ret_pos, center, t3, "退货任务并返回", d3))

            drone.available_time = start_time + t1 + t2 + t3
            manager.update_demand(del_pos, 'delivery', -1)
            manager.update_demand(ret_pos, 'return', -1)

        # 记录日志并输出
        drone.total_distance += task_distance
        total_optimized_cost += task_distance

        for event in events:
            leave_time, from_pos, to_pos, duration, desc, dist = event
            arrive_time = leave_time + duration
            drone.log.append((leave_time, from_pos, arrive_time, to_pos, desc, dist))
            realtime_print(f"无人机[{drone.id}]在时间 {leave_time:.2f} 从{from_pos}出发: {desc} (距离:{dist:.2f})")
            realtime_print(f"无人机[{drone.id}]在时间 {arrive_time:.2f} 到达{to_pos}")

        realtime_print(f"无人机[{drone.id}]完成{task_type}任务，总飞行距离:{task_distance:.2f}")

    return total_optimized_cost


def main():
    # ===== 初始化参数 =====
    speed = 1.0  # 无人机速度 (单位/秒)
    center = (0, 0)  # 配送中心坐标
    total_cost = 0.0  # 总配送成本（总飞行距离）

    # 快递柜数据: (位置, 取货需求, 退货需求)
    lockers_manager = LockersManager()
    lockers_manager.add_locker((-1.07096336564449, 0.00357922571952681), 2, 4)  # 距离 √2≈1.41
    lockers_manager.add_locker((-11.8463248263775, 7.30341402934254), 6, 6)  # 距离 √13≈3.61

    # 初始化无人机
    drones = [
        Drone(0, center, max_range=30),
        Drone(1, center, max_range=30),
        Drone(2, center, max_range=30),
        Drone(3, center, max_range=15)  # 中等续航
    ]

    # ===== 配送流程 =====
    lockers_manager.classify_demands()

    realtime_print("\n======= 开始强制配送阶段 =======")
    forced_cost = process_forced_deliveries(drones, lockers_manager, center, speed)
    total_cost += forced_cost

    realtime_print("\n======= 开始优化配送阶段 =======")
    optimized_cost = process_optimized_deliveries(drones, lockers_manager, center, speed)
    total_cost += optimized_cost

    realtime_print("\n======= 配送完成 =======")

    # 打印成本信息
    realtime_print(f"\n总配送成本: {total_cost:.2f}")
    realtime_print(f"强制配送阶段成本: {forced_cost:.2f}")
    realtime_print(f"优化配送阶段成本: {optimized_cost:.2f}")

    # 打印无人机成本明细
    for drone in drones:
        realtime_print(f"无人机[{drone.id}]总飞行距离: {drone.total_distance:.2f}, 续航限制: {drone.max_range}")

    # 打印最终状态
    realtime_print("\n最终需求状态:")
    for locker in lockers_manager.lockers:
        print(f"快递柜{locker[0]}: 取货需求={locker[1]}, 退货需求={locker[2]}")

    realtime_print("\n无人机任务汇总:")
    for drone in drones:
        print(f"\n无人机[{drone.id}]的任务记录 (总距离:{drone.total_distance:.2f}):")
        for log in drone.log:
            print(f"- {log[1]} -> {log[3]} (开始:{log[0]:.2f}, 到达:{log[2]:.2f}, 距离:{log[5]:.2f}, 任务:{log[4]})")


if __name__ == "__main__":
    main()
