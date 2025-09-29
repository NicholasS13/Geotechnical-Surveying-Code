import heapq
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import os
import pickle
import logging

sensor_file = "sensor.txt"
grid_x_file = "grid_X.npy"
grid_y_file = "grid_Y.npy"

logger = logging.getLogger("__main__")
holistic_score_map_image_counter = 0
closeness_to_robot_image_counter = 0
grid_image_counter = 0
path_to_goal_image_counter = 0
perform_kriging_counter = 0


import heapq
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import os
import pickle

# --- Robot Class ---
class Robot:
    
    def __init__(self, robot_id, pos, vmc):
        self.id = robot_id  # it can be zero -1 -2 , depends on the data that we have
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


    def compute_holistic_score_map(self, shared_score_map, grid_X, grid_Y, w_current_pos=0.10, w_goal=1.0,visualize=True):
        global holistic_score_map_image_counter
        # # holistic code combines the shared map with weighted values of features that are unique to robots 
        # these two values are:    stick to existing goal and closeness to the goal 
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
            fig = plt.figure(figsize=(6, 5))
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
            #plt.show()

            filename = f"static/figures/Holistic Score Map/{self.id} {holistic_score_map_image_counter}.png"
            fig.savefig(filename)

            plt.close(fig)  # prevent memory leaks
            holistic_score_map_image_counter = holistic_score_map_image_counter + 1


        return score_map
   
    
    def assign_to_grid(self, grid_X, grid_Y, vmc_map):
        row, col = get_nearest_cell(grid_X, grid_Y, self.pos)
        self.grid_idx = (row, col)
        self.path.append((row, col))
        vmc_map[row, col] = self.vmc
        return row, col

    def compute_closeness_map(self, grid_X, grid_Y, visualize=True):
        global closeness_to_robot_image_counter
        dist = np.sqrt((grid_X - self.pos[0])**2 + (grid_Y - self.pos[1])**2)
        closeness = 1 - (dist / np.max(dist))
        self.closeness_map = closeness

        if visualize:
            fig =plt.figure(figsize=(6, 5))
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
            #plt.show()

            filename = f"static/figures/Closeness To Robot/{self.id} {closeness_to_robot_image_counter}.png"
            fig.savefig(filename)

            plt.close(fig)  # prevent memory leaks
            closeness_to_robot_image_counter = closeness_to_robot_image_counter + 1

        return closeness
    



# --- Grid Functions ---
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def visualize_shared_score(shared_score, grid_X, grid_Y,
                           robots=None, visited_mask=None, ownership_map=None,
                           title="Shared Score Map", cmap="plasma",
                           show=True, save_path=None):
    """
    Visualize the shared score map with optional overlays:
      - robot positions & goals
      - visited cells
      - Voronoi (ownership) boundaries

    Parameters
    ----------
    shared_score : 2D np.ndarray
    grid_X, grid_Y : 2D np.ndarray (same shape)
    robots : list[Robot] or None
    visited_mask : 2D bool array or set of (row,col) or None
    ownership_map : 2D int array (index of nearest robot per cell) or None
    title : str
    cmap : str
    show : bool
    save_path : str or None
    """
    # Normalize if needed (for nicer color range)
    s = np.array(shared_score, dtype=float)
    smin, sptp = np.nanmin(s), (np.nanmax(s) - np.nanmin(s) + 1e-9)
    s_norm = (s - smin) / sptp

    fig, ax = plt.subplots(figsize=(7, 6))

    # Base heatmap
    im = ax.imshow(
        s_norm, origin="lower", cmap=cmap, alpha=0.95,
        extent=[grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]]
    )
    fig.colorbar(im, ax=ax, label="Shared score (normalized)")

    # Voronoi boundaries (white contours)
    if ownership_map is not None:
        if ownership_map.shape != shared_score.shape:
            raise ValueError("ownership_map shape must match shared_score shape.")
        # Contour between integer regions: draw at half steps [-0.5, 0.5, 1.5, ...]
        levels = np.arange(-0.5, np.max(ownership_map) + 1.5, 1.0)
        ax.contour(grid_X, grid_Y, ownership_map, levels=levels,
                   colors="white", linewidths=1.0, alpha=0.75)

    # Visited cells overlay
    if visited_mask is not None:
        if isinstance(visited_mask, set):
            vm = np.zeros_like(s, dtype=bool)
            for r, c in visited_mask:
                vm[r, c] = True
        else:
            vm = visited_mask.astype(bool)
        vx = grid_X[vm]; vy = grid_Y[vm]
        if vx.size:
            ax.scatter(vx, vy, s=8, c="k", alpha=0.6, label="Visited")

    # Robots and goals
    if robots:
        for rb in robots:
            ax.scatter(rb.pos[0], rb.pos[1], c="cyan", edgecolor="k",
                       s=70, marker="o", zorder=3, label=f"R{rb.id} pos")
            ax.text(rb.pos[0], rb.pos[1], f"R{rb.id}", fontsize=10,
                    ha="left", va="bottom", color="k")
            if getattr(rb, "goal", None) is not None:
                ax.scatter(rb.goal[0], rb.goal[1], c="lime", edgecolor="k",
                           s=80, marker="*", zorder=3, label=f"R{rb.id} goal")
                ax.plot([rb.pos[0], rb.goal[0]], [rb.pos[1], rb.goal[1]],
                        linestyle="--", linewidth=1, color="white", alpha=0.6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def compute_ownership_map(robots, grid_X, grid_Y):
    if not robots:
        return None
    xs = np.array([rb.pos[0] for rb in robots])[:, None, None]
    ys = np.array([rb.pos[1] for rb in robots])[:, None, None]
    d = np.hypot(grid_X[None, ...] - xs, grid_Y[None, ...] - ys)  # (R,H,W)
    closest = np.argmin(d, axis=0)  # int array (H,W)
    return closest


def voronoi_masks(robots, grid_X, grid_Y):
    # stack distances: shape (R, H, W)
    R = len(robots)
    xs = np.array([rb.pos[0] for rb in robots])[:, None, None]
    ys = np.array([rb.pos[1] for rb in robots])[:, None, None]
    dist = np.sqrt((grid_X[None, ...] - xs)**2 + (grid_Y[None, ...] - ys)**2)
    closest = np.argmin(dist, axis=0)  # (H,W) robot index for each cell
    # map robot.id -> boolean mask where cell is NOT this robot's Voronoi region
    id_to_idx = {rb.id: i for i, rb in enumerate(robots)}
    return {rb.id: (closest != id_to_idx[rb.id]) for rb in robots}

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
    global grid_image_counter
    
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
        #plt.show()

        filename = f"static/figures/Grid around Center Position/{grid_image_counter}.png"
        plt.savefig(filename)

        plt.close(fig)  # prevent memory leaks
        grid_image_counter = grid_image_counter + 1

    return grid_X, grid_Y

def get_nearest_cell(grid_X, grid_Y, pos):
    dx = grid_X - pos[0]
    dy = grid_Y - pos[1]
    dist_sq = dx**2 + dy**2
    idx_flat = np.argmin(dist_sq)
    return np.unravel_index(idx_flat, grid_X.shape)

def perform_kriging(known_positions, known_vmc, grid_X, grid_Y):
    global perform_kriging_counter
     # Deduplicate exact coordinate repeats
    pts = np.array(known_positions, dtype=float)
    vals = np.array(known_vmc, dtype=float)
    if pts.size == 0:
        raise RuntimeError("No data points provided to kriging.")
    _, keep_idx = np.unique(pts, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    pts, vals = pts[keep_idx], vals[keep_idx]
    # Build Ordinary Kriging (turn off exact interpolation for stability)
    OK = OrdinaryKriging(
        pts[:, 0], pts[:, 1], vals,
        variogram_model="spherical",
        exact_values=False
    )
    # Use moving window + loop backend (required)
    Zhat, Zvar = OK.execute(
        "grid",
        grid_X[0, :],
        grid_Y[:, 0],
        backend="loop",                      # <-- key fix
        n_closest_points=min(20, len(vals))   # local neighborhood; tweak if you like
    )

    # --- Visualization of Zhat (Expected Value) ---
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(Zhat, origin='lower', cmap='viridis',
               extent=[grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]])
    plt.title("Kriging - Expected Value (Zhat)")
    plt.colorbar(label="VMC")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    #plt.show()
    filename = f"static/figures/Zhat/{perform_kriging_counter}.png"
    plt.savefig(filename)
    plt.close(fig)


    # --- Visualization of Zvar (Variance) ---
    fig2 = plt.figure(figsize=(6, 5))
    plt.imshow(Zvar, origin='lower', cmap='magma',
               extent=[grid_X[0, 0], grid_X[0, -1], grid_Y[0, 0], grid_Y[-1, 0]])
    plt.title("Kriging - Variance (Zvar)")
    plt.colorbar(label="Uncertainty")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    #plt.show()
    filename = f"static/figures/Zvar/{perform_kriging_counter}.png"
    plt.savefig(filename)

    perform_kriging_counter = perform_kriging_counter+1;

    plt.close(fig2)

    return Zhat, Zvar


def compute_closeness_to_center_map(grid_X, grid_Y):
    center_x = grid_X[grid_X.shape[0] // 2, grid_X.shape[1] // 2]
    center_y = grid_Y[grid_X.shape[0] // 2, grid_X.shape[1] // 2]
    dist = np.sqrt((grid_X - center_x) ** 2 + (grid_Y - center_y) ** 2)
    closeness = 1 - (dist / np.max(dist))
    return closeness
def assign_goal_and_path_for_robot(robot, holistic_score_map, grid_X, grid_Y, visited_mask,
                                   visualize=True, voronoi_block=None):
    global path_to_goal_image_counter

    penalized_map = np.copy(holistic_score_map)

    # 1) Voronoi: zero cells that belong to other robots
    if voronoi_block is not None:
        penalized_map[voronoi_block] = 0.0

    # 2) Visited penalty (keep your behavior)
    penalized_map[visited_mask] = -1e9

    # 3) Pick goal from the Voronoi-masked map
    best_idx = np.unravel_index(np.nanargmax(penalized_map), penalized_map.shape)
    robot.goal = (grid_X[best_idx], grid_Y[best_idx])
    robot.goal_idx = best_idx

    # 4) Build A* cost **from the same masked map**
    s = penalized_map
    smin = np.nanmin(s)
    sptp = np.nanmax(s) - smin + 1e-9
    s_norm = (s - smin) / sptp
    cost_map = (1.0 / np.clip(s_norm, 1e-6, None)) / 0.01  # matches Alg.1 routing step
    cost_map[visited_mask] += 1000

    robot.path = a_star(robot.grid_idx, best_idx, cost_map)
    robot.next_move = robot.path[1] if len(robot.path) > 1 else (robot.path[0] if robot.path else None)
    robot.next_pos = (grid_X[robot.next_move], grid_Y[robot.next_move]) if robot.next_move else None

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
        #plt.show()

        filename = f"static/figures/Path To Goal/{robot.id} {path_to_goal_image_counter}.png"
        plt.savefig(filename)

        plt.close(fig)  # prevent memory leaks
        path_to_goal_image_counter = path_to_goal_image_counter + 1

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
