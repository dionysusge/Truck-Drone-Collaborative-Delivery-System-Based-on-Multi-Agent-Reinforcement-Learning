import json
import pandas as pd

def generate_truck1_table():
    with open('test_results/detailed_test_data.json', 'r') as f:
        data = json.load(f)

    lockers_state = data['truck_operations'][0]['env_state']['lockers_state']
    locker_locations = [locker['location'] for locker in lockers_state]

    truck_ops = [op for op in data['truck_operations'] if op['truck_id'] == 1]
    truck_ops.sort(key=lambda x: x['step'])

    rewards = {}
    truck_df = pd.read_csv('test_results/truck_states.csv')
    for _, row in truck_df.iterrows():
        if int(row['truck_id']) == 1:
            rewards[int(row['step'])] = float(row['reward'])

    print("| 步骤 (Step) | 卡车停靠坐标 (Truck Stop) | 无人机服务目标 (Drone Targets) | 单步奖励 (Reward) |")
    print("| :---: | :---: | :--- | :---: |")

    for op in truck_ops:
        step = op['step']
        pos = op['truck_state']['position']
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})"

        service_area = op['action']['service_area']
        targets = []
        for idx, served in enumerate(service_area):
            if served == 1 and idx < len(locker_locations):
                loc = locker_locations[idx]
                targets.append(f"L{idx+1}({loc[0]:.2f}, {loc[1]:.2f})")

        if len(targets) > 4:
            targets_str = ", ".join(targets[:4]) + ", ..."
        elif len(targets) > 0:
            targets_str = ", ".join(targets)
        else:
            targets_str = "None"

        reward = rewards.get(step, 0.0)
        print(f"| {step} | {pos_str} | {targets_str} | {reward:.2f} |")

if __name__ == '__main__':
    generate_truck1_table()
