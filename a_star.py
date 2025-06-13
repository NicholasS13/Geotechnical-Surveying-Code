import heapq
import numpy as np

def a_star_path(start, goal, cost_grid):
    """
    A* pathfinding on a 2D grid using a cost map.

    Args:
        start (tuple): (row, col) start cell.
        goal (tuple): (row, col) goal cell.
        cost_grid (2D np.ndarray): Lower values = preferred path (e.g., variance map).

    Returns:
        list of (row, col): path from start to goal.
    """
    rows, cols = cost_grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # reverse

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4 directions
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                tentative_g = g_score[current] + cost_grid[neighbor]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    # If we get here, no path found
    raise ValueError("No path found from {} to {}".format(start, goal))
