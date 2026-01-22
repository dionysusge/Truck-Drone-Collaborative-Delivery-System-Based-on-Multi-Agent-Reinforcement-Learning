import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_truck_plot(json_path, truck_id, output_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract locker locations
    # Assuming lockers_state is consistent across steps, we take it from the first operation
    # or we can look for it in the first operation that has it.
    lockers_map = {}
    
    # Find the first operation to get lockers_state
    first_op = data['truck_operations'][0]
    lockers_state = first_op['env_state']['lockers_state']
    
    locker_locations = []
    for locker in lockers_state:
        # Assuming locker['location'] is [x, y]
        loc = locker['location']
        locker_locations.append(loc)
        lockers_map[locker['id']] = loc

    # Filter operations for the specific truck
    truck_ops = [op for op in data['truck_operations'] if op['truck_id'] == truck_id]
    truck_ops.sort(key=lambda x: x['step'])

    if not truck_ops:
        print(f"No operations found for truck {truck_id}")
        return

    # Prepare plotting
    plt.figure(figsize=(12, 10))
    
    # Plot all lockers as background
    locker_xs = [loc[0] for loc in locker_locations]
    locker_ys = [loc[1] for loc in locker_locations]
    plt.scatter(locker_xs, locker_ys, c='lightgray', s=30, label='Lockers', alpha=0.5)

    # Track truck path
    truck_path_x = []
    truck_path_y = []
    
    # Depot location (warehouse/starting point)
    depot = (0, 0)
    
    # Colors for steps to show progression? Or just one color.
    # Let's use a colormap for drone paths to distinguish steps if needed, 
    # but for simplicity, let's use consistent colors.
    
    truck_color = 'blue'
    drone_color = 'orange'
    depot_color = 'red'

    print(f"Processing {len(truck_ops)} steps for Truck {truck_id}...")

    # Plot depot (warehouse)
    plt.scatter(depot[0], depot[1], c=depot_color, s=150, marker='*', 
                zorder=6, label='Depot', edgecolors='black', linewidths=1.5)
    plt.text(depot[0]+2, depot[1]+2, 'Depot', fontsize=11, color=depot_color, 
             fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    for i, op in enumerate(truck_ops):
        step = op['step']
        truck_state = op['truck_state']
        action = op['action']
        
        # Truck position at this step
        # Note: In the log, 'position' seems to be the location *after* arrival or *at* the stop.
        current_pos = truck_state['position']
        truck_path_x.append(current_pos[0])
        truck_path_y.append(current_pos[1])
        
        # If this is the first stop, draw line from depot to first stop
        if i == 0:
            plt.plot([depot[0], current_pos[0]], [depot[1], current_pos[1]], 
                    c=truck_color, linewidth=2, alpha=0.8, linestyle='-')
            # Add arrow from depot to first stop
            plt.arrow(depot[0], depot[1], 
                     (current_pos[0] - depot[0])*0.5, 
                     (current_pos[1] - depot[1])*0.5, 
                     head_width=3, head_length=5, fc=truck_color, ec=truck_color, alpha=0.8)
        
        # Mark the stop
        plt.scatter(current_pos[0], current_pos[1], c=truck_color, s=100, marker='s', zorder=5)
        plt.text(current_pos[0]+2, current_pos[1]+2, f'S{step}', fontsize=10, color=truck_color, fontweight='bold')

        # Drone Operations
        # action['service_area'] is the drone dispatch vector
        service_area = action['service_area']
        
        for idx, is_served in enumerate(service_area):
            if is_served == 1:
                # Get locker location
                if idx < len(locker_locations):
                    target_loc = locker_locations[idx]
                    
                    # Draw drone path (dashed)
                    plt.plot([current_pos[0], target_loc[0]], [current_pos[1], target_loc[1]], 
                             c=drone_color, linestyle='--', alpha=0.6, linewidth=1)
                    
                    # Highlight served locker
                    plt.scatter(target_loc[0], target_loc[1], c='red', s=40, marker='^', zorder=4)

    # Draw truck route lines
    plt.plot(truck_path_x, truck_path_y, c=truck_color, linewidth=2, label='Truck Path', alpha=0.8)
    
    # Add arrows to truck path
    for i in range(len(truck_path_x) - 1):
        plt.arrow(truck_path_x[i], truck_path_y[i], 
                  (truck_path_x[i+1] - truck_path_x[i])*0.5, 
                  (truck_path_y[i+1] - truck_path_y[i])*0.5, 
                  head_width=3, head_length=5, fc=truck_color, ec=truck_color, alpha=0.8)

    # Custom legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=depot_color, 
               markeredgecolor='black', markeredgewidth=1.5, label='Depot', markersize=12),
        Line2D([0], [0], color=truck_color, lw=2, label='Truck Route'),
        Line2D([0], [0], color=drone_color, lw=1, linestyle='--', label='Drone Flight'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=truck_color, label='Truck Stop', markersize=10),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', label='Served by Drone', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', label='Other Lockers', markersize=8),
    ]

    plt.title(f'Route and Drone Operations Analysis - Truck {truck_id}', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    base_dir = "/home/gzw/bad_truck_drone"
    json_file = os.path.join(base_dir, "test_results/detailed_test_data.json")
    output_file = os.path.join(base_dir, "truck1_route_analysis.png")
    
    generate_truck_plot(json_file, 1, output_file)
