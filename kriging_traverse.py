import heapq
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import os
import pickle
import logging

sensor_file = "sensor.txt"
grid_x_file = "grid_X.npy"
grid_y_file = "grid_Y.npy"

logger = logging.getLogger("__main__")

# --- Robot Class ---
class Robot:
    
    def __init__(self, robot_id:int, pos:int, vmc:int):
        self.id = robot_id
        self.pos = pos
        self.vmc = vmc
        self.goal = None
        self.goal_idx = None
        self.next_move = None
        self.path = []
        self.grid_idx = None
        self.closeness_map = None
        self.next_pos = None
    def __repr__(self):
         return (
        f"<Robot {self.id} | Pos (lon, lat): {self.pos} | VMC: {self.vmc} | "
        f"Grid Index: {self.grid_idx} | "
        f"Goal (lon, lat): {self.goal} | Goal Index: {self.goal_idx} | "
        f"Next Move (grid idx): {self.next_move} | Next Pos (lon, lat): {self.next_pos}>"
         )


    def compute_holistic_score_map(self, shared_score_map, grid_X, grid_Y, w_current_pos:float=10, w_goal:float=10.0, visualize:bool=True):
        """
        Combines shared score map with robot-specific closeness to current pos and goal stability.
        """
        dist_to_current = np.sqrt((grid_X - self.pos[0])**2 + (grid_Y - self.pos[1])**2)
        closeness_to_current = 1 - (dist_to_current / np.max(dist_to_current))

        if self.goal:
            goal_pos = self.goal
            dist_to_goal = np.sqrt((grid_X - goal_pos[0])**2 + (grid_Y - goal_pos[1])**2)
            norm_dist_to_goal = (dist_to_goal - np.min(dist_to_goal)) / (np.ptp(dist_to_goal) + 1e-9)
            prefer_goal_map = 1 - norm_dist_to_goal
        else:
            prefer_goal_map = np.ones_like(shared_score_map)  # in the initial stage, there is no goal , this part makes sure 
            # that lack of goal does not affect the holistic map

        score_map = (
            shared_score_map +
            w_current_pos * closeness_to_current +
            w_goal * prefer_goal_map
        )

        if visualize:
            plt.figure(figsize=(6, 5))
            plt.imshow(score_map, origin='lower', cmap='plasma', extent=[
                grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]
            ])
            plt.title(f"Robot {self.id} - Holistic Score Map")
            plt.colorbar(label="Score")
            plt.scatter(self.pos[0], self.pos[1], c='red', marker='x', label='Current Pos')
            if self.goal:
                plt.scatter(self.goal[0], self.goal[1], c='green', marker='o', label='Current Goal')
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return score_map
   
    

    def assign_to_grid(self, grid_X, grid_Y, vmc_map):
        row, col = get_nearest_cell(grid_X, grid_Y, self.pos)
        self.grid_idx = (row, col)
        self.path.append((row, col))
        vmc_map[row, col] = self.vmc
        return row, col

    def compute_closeness_map(self, grid_X, grid_Y, visualize=True):
        dist = np.sqrt((grid_X - self.pos[0])**2 + (grid_Y - self.pos[1])**2)
        closeness = 1 - (dist / np.max(dist))
        self.closeness_map = closeness

        if visualize:
            plt.figure(figsize=(6, 5))
            plt.imshow(closeness, origin='lower', cmap='Greens', extent=[
                grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]
            ])
            plt.colorbar(label="Closeness to Robot")
            plt.scatter(self.pos[0], self.pos[1], c='red', marker='x', label=f'Robot {self.id}')
            plt.text(self.pos[0], self.pos[1], f'R{self.id}', color='black', fontsize=12, ha='right')
            plt.title("Closeness to Robot Map")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return closeness
    



# --- Grid Functions ---
def a_star(start, goal, cost_map, allow_diagonal=True):
    rows, cols = cost_map.shape
    visited = np.zeros_like(cost_map, dtype=bool)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    
    open_set = []
    heapq.heappush(open_set, (f_score[start], start))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        visited[current] = True
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if visited[neighbor]:
                    continue
                tentative_g = g_score[current] + cost_map[neighbor]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # return empty path if no route found
 

def create_grid(center_pos, grid_size, cell_size_lat, visualize=False):
    center_lon, center_lat = center_pos
    cell_size_lon = cell_size_lat * np.cos(np.radians(center_lat))

    half = grid_size // 2
    x_vals = np.arange(-half, half + 1) * cell_size_lon + center_lon
    y_vals = np.arange(-half, half + 1) * cell_size_lat + center_lat
    grid_X, grid_Y = np.meshgrid(x_vals, y_vals)

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(center_lon, center_lat, 'ro', label="Center")
        for i in range(grid_size):
            for j in range(grid_size):
                x = grid_X[i, j] - cell_size_lon / 2
                y = grid_Y[i, j] - cell_size_lat / 2
                rect = plt.Rectangle((x, y), cell_size_lon, cell_size_lat,
                                     fill=False, edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
        ax.set_title("Grid around Center Position")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        ax.legend()
        plt.grid(False)
        plt.show()

    return grid_X, grid_Y

def get_nearest_cell(grid_X, grid_Y, pos):
    dx = grid_X - pos[0]
    dy = grid_Y - pos[1]
    dist_sq = dx**2 + dy**2
    idx_flat = np.argmin(dist_sq)
    return np.unravel_index(idx_flat, grid_X.shape)

def perform_kriging(known_positions, known_vmc, grid_X, grid_Y):
    x = np.array([p[0] for p in known_positions])
    y = np.array([p[1] for p in known_positions])
    v = np.array(known_vmc)

    OK = OrdinaryKriging(x, y, v, variogram_model='exponential', exact_values=True)
    Zhat, Zvar = OK.execute('grid', grid_X[0, :], grid_Y[:, 0])
    return Zhat, Zvar

def compute_closeness_to_center_map(grid_X, grid_Y):
    center_x = grid_X[grid_X.shape[0] // 2, grid_X.shape[1] // 2]
    center_y = grid_Y[grid_X.shape[0] // 2, grid_X.shape[1] // 2]
    dist = np.sqrt((grid_X - center_x) ** 2 + (grid_Y - center_y) ** 2)
    closeness = 1 - (dist / np.max(dist))
    return closeness
# --- Assign Goal and Path ---
def assign_goal_and_path_for_robot(robot, holistic_score_map, grid_X, grid_Y, visited_mask, visualize=True):
    penalized_map = np.copy(holistic_score_map)
    penalized_map[visited_mask] = -1e9

    best_idx = np.unravel_index(np.argmax(penalized_map), penalized_map.shape)
    robot.goal = (grid_X[best_idx], grid_Y[best_idx])
    robot.goal_idx = best_idx

    cost_map = 1 / (holistic_score_map + 1e-6)
    cost_map[visited_mask] += 1000

    robot.path = a_star(robot.grid_idx, best_idx, cost_map)
    robot.next_move = robot.path[1] if len(robot.path) > 1 else robot.path[0] if robot.path else None
    
    # Convert to lat/lon
    if robot.next_move:
        robot.next_pos = (grid_X[robot.next_move], grid_Y[robot.next_move])
    else:
        robot.next_pos = None

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(holistic_score_map, origin='lower', cmap='plasma',
                  extent=[grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]])
        ax.set_title(f"Robot {robot.id} Path to Goal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.plot(robot.pos[0], robot.pos[1], 'rx', label='Current Pos')
        ax.plot(robot.goal[0], robot.goal[1], 'go', label='Goal')
        for (r, c) in robot.path:
            ax.plot(grid_X[r, c], grid_Y[r, c], 'ko', markersize=2)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return robot.goal, robot.path, robot.next_move
def save_visited_cell(idx, file='visited_file'):
    visited = load_visited_cells(file)
    if idx not in visited:
        visited.append(idx)
        with open(file, 'wb') as f:
            pickle.dump(visited, f)

def load_visited_cells(file='visited_file'):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    return []

def save_robot_goals(goals_dict, file='robot_goals.pkl'):
    with open(file, 'wb') as f:
        pickle.dump(goals_dict, f)

def load_robot_goals(file='robot_goals.pkl'):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    return {}
def save_state(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_state(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return set()

def get_latest_robot_entries(filename):
    latest_entries = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue  # skip invalid lines
            try:
                vmc = float(parts[0])
                lon = float(parts[3])
                lat = float(parts[4])
                robot_id = int(parts[-1])
            except ValueError:
                continue  # skip lines with bad data

            # Always keep the last occurrence (later line) for each robot_id
            latest_entries[robot_id] = (robot_id, (lon, lat), vmc)

    return [Robot(robot_id, pos, vmc) for robot_id, pos, vmc in latest_entries.values()]

def compute_shared_score_map(Zhat, Zvar, closeness_to_center, 
                              w_exp=1.0, w_var=10.0, w_center=0.1):
    # Normalize each component to [0, 1]
    Zhat_norm = (Zhat - np.min(Zhat)) / (np.ptp(Zhat) + 1e-9)
    Zvar_norm = (Zvar - np.min(Zvar)) / (np.ptp(Zvar) + 1e-9)
    center_norm = (closeness_to_center - np.min(closeness_to_center)) / (np.ptp(closeness_to_center) + 1e-9)

    # Weighted sum
    score_map = (
        w_exp * Zhat_norm +
        w_var * Zvar_norm +
        w_center * center_norm
    )
    return score_map


def run_traverse():
    grid_X = np.load(grid_x_file)
    grid_Y = np.load(grid_y_file)
    
    logger.debug("Grid X ", grid_X)
    logger.debug("Grid Y ", grid_Y)
    vmc_map = np.full_like(grid_X, np.nan)
    all_positions = []
    all_vmc = []
    latest_robot_data = {}
    visited_cells = set()
    
    with open("sensor.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6 or parts[0].strip() == "":
                continue
            try:
                vmc = float(parts[0])
                lon = float(parts[3])
                lat = float(parts[4])
                robot_id = int(parts[-1])
    
                # For kriging
                all_positions.append((lon, lat))
                all_vmc.append(vmc)
    
                # Latest robot position
                latest_robot_data[robot_id] = Robot(robot_id, (lon, lat), vmc)
    
                # Visited cell
                visited_cells.add(get_nearest_cell(grid_X, grid_Y, (lon, lat)))
            except:
                continue
    # --- Step 7: Kriging and scoring ---
    Zhat, Zvar = perform_kriging(all_positions, all_vmc, grid_X, grid_Y)
    center_map = compute_closeness_to_center_map(grid_X, grid_Y)
    shared_score =compute_shared_score_map(Zhat, Zvar, center_map,w_exp=1.0, w_var=1.0, w_center=0.1)
    
    # --- Step 8: Load previous goals ---
    goals_file = "goals_file.txt"
    if not os.path.exists(goals_file):
        goal_dict = {}
    else:
        goal_dict = load_state(goals_file)
        if not isinstance(goal_dict, dict):
          goal_dict = {}
    
    
    # --- Step 9: Update robots with past goals ---
    robots = list(latest_robot_data.values())
    for robot in robots:
        if robot.id in goal_dict:
            robot.goal_idx = goal_dict[robot.id]
            robot.goal = (grid_X[robot.goal_idx], grid_Y[robot.goal_idx])
    
    # --- Step 10: Apply visited mask ---
    visited_mask = np.zeros_like(shared_score, dtype=bool)
    for idx in visited_cells:
        visited_mask[idx] = True
    for robot in robots:
        row, col = get_nearest_cell(grid_X, grid_Y, robot.pos)
        robot.grid_idx = (row, col)  # this sets the start point for A*
    # --- Step 11: Compute holistic map and assign goals ---
    for robot in robots:
        holistic_map = robot.compute_holistic_score_map(shared_score, grid_X, grid_Y)
        goal, path, next_move = assign_goal_and_path_for_robot(
            robot, holistic_map, grid_X, grid_Y, visited_mask
        )
        goal_dict[robot.id] = robot.goal_idx
    
    # --- Step 12: Save goals and visited cells ---
    save_state(goals_file, goal_dict)
    
    
    
    # --- Step 13: Output ---
    logger.debug("\n--- Robot Planning Results ---")
    for robot in robots:
        logger.debug(robot)
    logger.debug(visited_cells)