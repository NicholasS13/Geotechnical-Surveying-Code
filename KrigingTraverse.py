import numpy as np
import matplotlib.pyplot as plt
from variogramfitnew import variogramfit
from krigingnew import kriging
from a_star_alg import a_star_path
from scipy.spatial.distance import pdist


def build_experimental_variogram(x, y, values, bins=10, max_dist=None):
    coords = np.column_stack((x, y))
    dists = pdist(coords)
    diffs = pdist(values.reshape(-1, 1)) ** 2 / 2

    if max_dist is None:
        max_dist = np.max(dists) * 0.7

    bin_edges = np.linspace(0, max_dist, bins + 1)
    bin_indices = np.digitize(dists, bin_edges)

    h, gamma, numobs = [], [], []
    for i in range(1, bins + 1):
        mask = bin_indices == i
        if np.any(mask):
            h.append(np.mean(dists[mask]))
            gamma.append(np.mean(diffs[mask]))
            numobs.append(np.sum(mask))

    return np.array(h), np.array(gamma), np.array(numobs)


def load_sensor_data(filename):
    sensor_data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            vwc, temp, ec = map(float, parts[0:3])
            lon = float(parts[3])
            lat = float(parts[4])
            timestamp = parts[5]
            robot_id = parts[6]
            sensor_data.append((vwc, temp, ec, f"{lon} {lat}", timestamp, robot_id))
    return sensor_data


def grid_to_longlat(i, j, grid_shape, lon_min, lon_max, lat_min, lat_max):
    nx, ny = grid_shape
    lon = lon_min + (lon_max - lon_min) * (j / (nx - 1))
    lat = lat_min + (lat_max - lat_min) * (i / (ny - 1))
    return lon, lat


def kriging_traverse_maze(file: str, grid_shape: tuple = (50, 50)):
    sensor_data = load_sensor_data(file)

    x, y, vwc = [], [], []
    for v, temp, ec, coord_str, time_str, robot_id in sensor_data:
        lon, lat = map(float, coord_str.strip().split())
        x.append(lon)
        y.append(lat)
        vwc.append(float(v))

    x = np.array(x)
    y = np.array(y)
    vwc = np.array(vwc)

    lon_min, lon_max = np.min(x), np.max(x)
    lat_min, lat_max = np.min(y), np.max(y)

    h, gammaexp, numobs = build_experimental_variogram(x, y, vwc)
    a, c, nugget, vstruct = variogramfit(
        h, gammaexp, numobs=numobs, model="spherical", nugget=0.05, plotit=False
    )

    grid_x_vals = np.linspace(lon_min, lon_max, grid_shape[0])
    grid_y_vals = np.linspace(lat_min, lat_max, grid_shape[1])
    grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

    mean_map, var_map = kriging(vstruct, x, y, vwc, grid_x, grid_y)
    '''
    # Visualize Kriging Results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Kriging: Estimated VWC")
    plt.imshow(mean_map, origin='lower', cmap='viridis')
    plt.colorbar(label='VWC')
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.savefig("results/KrigingEstimatedVWC.jpg")

    plt.subplot(1, 2, 2)
    plt.title("Kriging Variance (Uncertainty)")
    plt.imshow(var_map, origin='lower', cmap='hot')
    plt.colorbar(label='Variance')
    plt.xlabel("X grid")
    plt.ylabel("Y grid")

    plt.tight_layout()
    plt.savefig("results/KrigingVarianceUncertainty.jpg")
    #plt.show()
    '''
    # Robot goal selection
    next_goals = {}
    robot_ids = list(set(entry[5] for entry in sensor_data))
    num_robots = len(robot_ids)

    flat_indices = np.argpartition(var_map.flatten(), -num_robots)[-num_robots:]
    top_goal_indices = [np.unravel_index(idx, var_map.shape) for idx in flat_indices]

    robot_ids.sort()
    top_goal_indices.sort(key=lambda t: -var_map[t])

    assigned_goals = []

    for robot_id in robot_ids:
        recent_entry = next(e for e in reversed(sensor_data) if e[5] == robot_id)
        lon, lat = map(float, recent_entry[3].strip().split())

        i = np.abs(grid_x_vals - lon).argmin()
        j = np.abs(grid_y_vals - lat).argmin()
        current = (j, i)

        goal_dists = [
            (np.linalg.norm(np.array(current) - np.array(g)), g)
            for g in top_goal_indices if g not in assigned_goals
        ]

        if not goal_dists:
            target = current
        else:
            _, target = min(goal_dists)
            assigned_goals.append(target)

        try:
            path = a_star_path(current, target, var_map)
            if len(path) > 1:
                next_goals[robot_id] = path[1]
            else:
                next_goals[robot_id] = target
        except Exception as e:
            print(f"Path planning failed for robot {robot_id}: {e}")
            next_goals[robot_id] = target

    return mean_map, var_map, next_goals, (lon_min, lon_max, lat_min, lat_max)
def run_kriging_traverse():
    mean_map, var_map, next_goals, bounds = kriging_traverse_maze("sensorData.txt")
    lon_min, lon_max, lat_min, lat_max = bounds
    grid_shape = var_map.shape

    print("\nNext Robot Goals:")
    for robot, (i, j) in next_goals.items():
        lon, lat = grid_to_longlat(i, j, grid_shape, lon_min, lon_max, lat_min, lat_max)
        print(f"Robot {robot} → Grid ({i}, {j}) → GPS ({lon:.6f}, {lat:.6f})")

        # Update db.txt
        with open("db.txt", 'r') as db:
            lines = db.readlines()

        robot_found = False
        for idx, line in enumerate(lines):
            if line.startswith(robot + ","):
                lines[idx] = f"{robot},{lon},{lat}\n"
                robot_found = True
                break

        if not robot_found:
            lines.append(f"{robot},{lon},{lat}\n")

        with open("db.txt", 'w') as db:
            db.writelines(lines)

    # Final variance heatmap
    plt.imshow(var_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Kriging Variance")
    plt.title("Kriging Variance Map")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.savefig("results/KrigingVariance.jpg")
    #plt.show()


if __name__ == "__main__":
    mean_map, var_map, next_goals, bounds = kriging_traverse_maze("sensorData.txt")
    lon_min, lon_max, lat_min, lat_max = bounds
    grid_shape = var_map.shape

    print("\nNext Robot Goals:")
    for robot, (i, j) in next_goals.items():
        lon, lat = grid_to_longlat(i, j, grid_shape, lon_min, lon_max, lat_min, lat_max)
        print(f"Robot {robot} → Grid ({i}, {j}) → GPS ({lon:.6f}, {lat:.6f})")

        # Update db.txt
        with open("db.txt", 'r') as db:
            lines = db.readlines()

        robot_found = False
        for idx, line in enumerate(lines):
            if line.startswith(robot + ","):
                lines[idx] = f"{robot},{lon},{lat}\n"
                robot_found = True
                break

        if not robot_found:
            lines.append(f"{robot},{lon},{lat}\n")

        with open("db.txt", 'w') as db:
            db.writelines(lines)

    # Final variance heatmap
    plt.imshow(var_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Kriging Variance")
    plt.title("Kriging Variance Map")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.show()
