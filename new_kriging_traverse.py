import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import heapq
import ast
cell_size=0.00012
# Load existing entries from db.txt to avoid duplicates
def read_existing_robot_ids(filepath="db.txt"):
    existing_ids = set()
    try:
        with open(filepath, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if fields and fields[0].isdigit():
                    existing_ids.add(int(fields[0]))
    except FileNotFoundError:
        pass  # file will be created if doesn't exist
    return existing_ids

def add_or_update_robot_position(robot_id, x, y, filepath="db.txt"):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    found = False
    new_lines = []
    for line in lines:
        fields = line.strip().split(",")
        if fields and fields[0].isdigit() and int(fields[0]) == robot_id:
            # Update this robot's line
            new_lines.append(f"{robot_id},{x},{y}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        # Add new line if robot_id not found
        new_lines.append(f"{robot_id},{x},{y}\n")

    with open(filepath, "w") as f:
        f.writelines(new_lines)

def create_grid_around_robot1(robot_position, grid_size, cell_size):
    if grid_size % 2 == 0:
        raise ValueError("Grid size must be odd.")
    half = grid_size // 2
    x_vals = np.arange(-half, half + 1) * cell_size + robot_position[0]
    y_vals = np.arange(-half, half + 1) * cell_size + robot_position[1]
    return np.meshgrid(x_vals, y_vals)
def get_nearest_grid_cell(grid_X, grid_Y, robot_position):
    dx = grid_X - robot_position[0]
    dy = grid_Y - robot_position[1]
    dist_squared = dx**2 + dy**2
    flat_idx = np.argmin(dist_squared)
    return np.unravel_index(flat_idx, grid_X.shape)
def insert_vmc_into_map(vmc_map, grid_X, grid_Y, robot_position, vmc_value):
    row, col = get_nearest_grid_cell(grid_X, grid_Y, robot_position)
    vmc_map[row, col] = vmc_value
    return (row, col)
def visualize_robot_paths(grid_X, grid_Y, robot_paths, Zhat, step, cell_size):
    global plt_counter
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Zhat, origin="lower", cmap="YlOrRd", extent=[
        grid_X[0, 0] - cell_size / 2, grid_X[0, -1] + cell_size / 2,
        grid_Y[0, 0] - cell_size / 2, grid_Y[-1, 0] + cell_size / 2
    ])
    for i, path in enumerate(robot_paths):
        y_coords = [idx[0] for idx in path[:step + 1]]
        x_coords = [idx[1] for idx in path[:step + 1]]
        ax.plot(grid_X[y_coords, x_coords], grid_Y[y_coords, x_coords],
                marker="o", label=f"Robot {i+1}")
    rows, cols = grid_X.shape
    for i in range(rows):
        for j in range(cols):
            x = grid_X[i, j] - cell_size / 2
            y = grid_Y[i, j] - cell_size / 2
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                 fill=False, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
    ax.set_title(f"Robot Paths and Kriging Map - Step {step+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Kriging Zhat (Expected Value)")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"figures/{plt_counter}.png") 
    plt_counter+=1
    plt.close()
# Kriging function
def perform_kriging_from_vmc_points(robot_positions, robot_vmc_values, grid_X, grid_Y):
    x_known = np.array([p[0] for p in robot_positions])
    y_known = np.array([p[1] for p in robot_positions])
    v_known = np.array(robot_vmc_values)
    OK = OrdinaryKriging(x_known, y_known, v_known, variogram_model='exponential', exact_values=True)
    x_grid = grid_X[0, :]
    y_grid = grid_Y[:, 0]
    Zhat, Zvar = OK.execute("grid", x_grid, y_grid)
    return Zhat, Zvar
def compute_closeness_to_center_map(grid_X, grid_Y):
    center_x = grid_X[grid_X.shape[0] // 2, grid_X.shape[1] // 2]
    center_y = grid_Y[grid_Y.shape[0] // 2, grid_Y.shape[1] // 2]
    dist = np.sqrt((grid_X - center_x) ** 2 + (grid_Y - center_y) ** 2)
    return 1 - (dist / np.max(dist))
def compute_closeness_to_robot_map(grid_X, grid_Y, robot_position):
    dist = np.sqrt((grid_X - robot_position[0]) ** 2 + (grid_Y - robot_position[1]) ** 2)
    return 1 - (dist / np.max(dist))
def compute_score_map(Zhat, Zvar, closeness_center_map, closeness_robot_map, dist_to_goal_map):
    Zhat_n = (Zhat - np.min(Zhat)) / (np.max(Zhat) - np.min(Zhat) + 1e-9)
    Zvar_n = (Zvar - np.min(Zvar)) / (np.max(Zvar) - np.min(Zvar) + 1e-9)
    dist_goal_n = (dist_to_goal_map - np.min(dist_to_goal_map)) / (np.max(dist_to_goal_map) - np.min(dist_to_goal_map) + 1e-9)
    score = (
        1.0 * Zhat_n +
        10.0 * Zvar_n +
        0.1 * closeness_center_map +
        0.1 * closeness_robot_map +
        10.0 * (1 - dist_goal_n)
    )
    return score
def a_star_path(grid_X, grid_Y, score_map, start_idx, goal_idx, visited_cells):
    rows, cols = score_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    g_score = {start_idx: 0}
    def h(a, b):
        # Chebyshev distance for 8-connected grid
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for d_i in [-1, 0, 1]:
            for d_j in [-1, 0, 1]:
                if d_i == 0 and d_j == 0:
                    continue  # Skip the current cell itself
                neighbor = (current[0] + d_i, current[1] + d_j)
                if (0 <= neighbor[0] < rows and
                    0 <= neighbor[1] < cols and
                    neighbor not in visited_cells):  # Skip visited cells
                    step_cost = np.hypot(d_i, d_j)  # 1 for straight, √2 for diagonal
                    tentative_g = g_score[current] + step_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f = tentative_g + h(neighbor, goal_idx)
                        heapq.heappush(open_set, (f, neighbor))
                        came_from[neighbor] = current
    return []
def get_goal_from_score_map(score_map, visited_cells):
    score_copy = np.copy(score_map)
    for cell in visited_cells:
        score_copy[cell] = -1000000#-np.inf  # Exclude visited cells
    flat_idx = np.argmax(score_copy)
    return np.unravel_index(flat_idx, score_copy.shape)
def move_one_step_a_star(score_map, grid_X, grid_Y, current_idx, visited_cells):
    goal_idx = get_goal_from_score_map(score_map, visited_cells)
    path = a_star_path(grid_X, grid_Y, score_map, current_idx, goal_idx, visited_cells)
    if len(path) >= 2:
        return path[1]
    return current_idx
def visualize_initial_state(grid_X, grid_Y, robot_positions, Zhat, cell_size):
    global plt_counter
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Zhat, origin="lower", cmap="YlOrRd", extent=[
        grid_X[0, 0] - cell_size / 2, grid_X[0, -1] + cell_size / 2,
        grid_Y[0, 0] - cell_size / 2, grid_Y[-1, 0] + cell_size / 2
    ])

    # Add robot positions
    for i, pos in enumerate(robot_positions):
        ax.plot(pos[0], pos[1], 'o', label=f'Robot {i+1}')

    # Add grid cell boundaries
    rows, cols = grid_X.shape
    for i in range(rows):
        for j in range(cols):
            x = grid_X[i, j] - cell_size / 2
            y = grid_Y[i, j] - cell_size / 2
            rect = plt.Rectangle((x, y), cell_size, cell_size, fill=False, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)

    ax.set_title("Initial Robot Positions and Kriging Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Kriging Zhat (Expected Value)")
    plt.tight_layout()
    plt.savefig(f"figures/{plt_counter}.png")
    plt.show()
    plt_counter+=1
    plt.close()
plt_counter = 0;
def run_multi_robot_exploration_with_visualization(grid_X, grid_Y, robot_positions, robot_vmc_values,cell_size,robot_positions_1,robot_vmc_values_1,num_iterations=0):
    visited_cells = set() # Track all visited cells
    #reads all saved ones from file
    with open('points.txt', 'r') as f:
        points = {ast.literal_eval(line.strip()) for line in f}
        visited_cells.update(points)# adds all points from file to visited cells
    
    num_robots = len(robot_positions)
    robot_paths = [[get_nearest_grid_cell(grid_X, grid_Y, pos)] for pos in robot_positions]
    
    # Initialize visited_cells with starting positions
    for path in robot_paths:
        visited_cells.add(path[0])

    for step in range(num_iterations):
        print(f"\n=== Iteration {step + 1} ===")
        Zhat, Zvar = perform_kriging_from_vmc_points(robot_positions_1, robot_vmc_values_1, grid_X, grid_Y)
        closeness_center_map = compute_closeness_to_center_map(grid_X, grid_Y)
        for i in range(num_robots):
            curr_idx = robot_paths[i][-1]
            robot_pos = (grid_X[curr_idx], grid_Y[curr_idx])
            closeness_robot_map = compute_closeness_to_robot_map(grid_X, grid_Y, robot_pos)
            dist_to_robot_map = np.sqrt((grid_X - robot_pos[0])**2 + (grid_Y - robot_pos[1])**2)
            score_map = compute_score_map(Zhat, Zvar, closeness_center_map, closeness_robot_map, dist_to_robot_map)
            new_idx = move_one_step_a_star(score_map, grid_X, grid_Y, curr_idx, visited_cells)
            robot_paths[i].append(new_idx)
            visited_cells.add(new_idx)  # Mark as visited
            new_pos = (grid_X[new_idx], grid_Y[new_idx])
            print(f"Robot {i+1} moved to: {new_pos}")
            add_or_update_robot_position(i, new_pos[0], new_pos[1])

        with open('points.txt', 'w') as f:
            for point in visited_cells:
                f.write(repr(point) + '\n')
        visualize_robot_paths(grid_X, grid_Y, robot_paths, Zhat, step, cell_size=cell_size)
    return robot_paths, Zhat