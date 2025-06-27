import heapq
import numpy as np

def a_star_path(start, goal, cost_grid):
    """
    Weighted A* pathfinding using a cost grid and embedded multi-factor weights.

    Args:
        start (tuple): (row, col) start cell.
        goal (tuple): (row, col) goal cell.
        cost_grid (2D np.ndarray): assumed to reflect expected value or base cost.

    Returns:
        list of (row, col): path from start to goal.
    """
    # Hardcoded weights
    weight_expected_value = 1
    weight_uncertainty = 10
    weight_prefer_center = 0.1
    weight_prefer_closeness = 0.1
    weight_prefer_existing_goal = 10
    weight_step_cost = 0.01  # NEW: penalize each step slightly

    rows, cols = cost_grid.shape
    center = np.array([rows / 2, cols / 2])
    max_center_dist = np.linalg.norm(center)
    max_dist = np.linalg.norm(np.array([rows, cols]))

    # Normalize cost_grid as if it's expected value
    norm_grid = (cost_grid - np.min(cost_grid)) / (np.max(cost_grid) - np.min(cost_grid) + 1e-8)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b), 2)  # Euclidean

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Extract expected/uncertainty proxies
                ev = norm_grid[neighbor]
                unc = 1 - ev  # proxy for uncertainty

                dist_center = np.linalg.norm(np.array(neighbor) - center) / max_center_dist
                dist_start = np.linalg.norm(np.array(neighbor) - np.array(start)) / max_dist
                dist_goal = np.linalg.norm(np.array(neighbor) - np.array(goal)) / max_dist

                # Weighted cost of entering neighbor
                step_cost = (
                    weight_expected_value * ev +
                    weight_uncertainty * unc +
                    weight_prefer_center * (1 - dist_center) +
                    weight_prefer_closeness * (1 - dist_start) +
                    weight_prefer_existing_goal * (1 - dist_goal) +
                    weight_step_cost  # penalize extra steps slightly
                )

                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    raise ValueError(f"No path found from {start} to {goal}")
