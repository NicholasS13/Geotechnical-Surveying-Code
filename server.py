#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, render_template_string,send_from_directory
from flask_cors import CORS
from kriging_traverse import *
import multiprocessing
import os
import numpy as np
import logging
import threading
import re

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture all levels

# Formatter (same for all files)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Custom filter to allow only specific level
class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level
    def filter(self, record):
        return record.levelno == self.level

# Dictionary of level → filename
log_files = {
    logging.DEBUG:   "files/logs/debug.log",
    logging.INFO:    "files/logs/info.log",
    logging.WARNING: "files/logs/warning.log",
    logging.ERROR:   "files/logs/error.log",
    logging.CRITICAL:"files/logs/critical.log",
}

# Create a handler per log level
for level, filepath in log_files.items():
    handler = logging.FileHandler(filepath, mode="a")
    handler.setLevel(level)
    handler.addFilter(LevelFilter(level))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

'''
logger.debug("This message should go to the debug.log file")
logger.info("This message should go to the info.log file")
logger.warning("This message should go to the warning.log file")
logger.error("This message should go to the error.log file")
'''

app = Flask(__name__)
CORS(app)
cell_size = 0.0001

Zhat = None
robots: list = list()

device_count = 0
DATA_FILE_NAME = "files/sensor.txt"

grid_x_file = "files/grid_X.npy"
grid_y_file = "files/grid_Y.npy"
# Global status dictionary (could also be stored in a file or DB)
kriging_status = {"running": False}
grid_X, grid_Y = None, None


def get_different_last_values(file_path):
    last_values = set()
    different_entries = []
    with open(file_path, "r") as file:
        for line in file:
            # Strip whitespace and split the line by commas
            entries = line.strip().split(",")
            if entries:  # Check if the line is not empty
                last_value = entries[-1]  # Get the last value
                if last_value not in last_values:
                    last_values.add(last_value)
                    different_entries.append(line.strip())  # Store the entire line
    return different_entries


run_kriging_counter = 0


def run_kriging_with_status():
    """Wrapper to run kriging traverse and update status flag."""
    global kriging_status
    global DATA_FILE_NAME
    global robots
    logger.debug("in run_kriging_with_status")
    save_grid()
    if len(get_different_last_values(DATA_FILE_NAME)) >= 3:
        try:
            kriging_status["running"] = True
            logger.debug("Starting kriging traversal process...")
            grid_X = np.load(grid_x_file)
            grid_Y = np.load(grid_y_file)

            logger.debug(f"Grid X {str(grid_X)}")
            logger.debug(f"Grid Y {str(grid_Y)}")
            vmc_map = np.full_like(grid_X, np.nan)
            all_positions = []
            all_vmc = []
            latest_robot_data = {}
            visited_cells = set()

            with open(DATA_FILE_NAME, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
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
                    except Exception as e:
                        logger.critical("Line 129:"+str(e))

            # --- Step 7: Kriging and scoring ---
            Zhat, Zvar = perform_kriging(all_positions, all_vmc, grid_X, grid_Y)
            center_map = compute_closeness_to_center_map(grid_X, grid_Y)
            shared_score = compute_shared_score_map(
                Zhat, Zvar, center_map, w_exp=1.0, w_var=1.0, w_center=0.1
            )

            # --- Step 8: Load previous goals ---
            goals_file = "files/goals_file.txt"
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
                holistic_map = robot.compute_holistic_score_map(
                    shared_score, grid_X, grid_Y
                )
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
        except Exception as e:
            logger.debug(f"Kriging traverse failed: {e}")
        finally:
            kriging_status["running"] = False
            logger.debug("Kriging traversal completed.")


def save_grid():
    global cell_size
    global DATA_FILE_NAME
    try:
        with open(DATA_FILE_NAME, "r") as f:
            first_line = f.readline().strip().split(",")
            lon = float(first_line[3])
            lat = float(first_line[4])
            initial_pos = (lon, lat)

        # --- Step 2: Create grid around initial position ---
        grid_X1, grid_Y1 = create_grid(initial_pos, grid_size=7, cell_size_lat=0.0001)

        # --- Step 3: Save to disk ---
        np.save(grid_x_file, grid_X1)
        np.save(grid_y_file, grid_Y1)

        return True
    except Exception as e:
        logger.debug(
            f"Exception in /save_grid: {e}"
        )  # Logs error to your Flask console
        return False

#to remove annoying 404 msgs
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static/icons'),
        'favicon.ico',
        mimetype='image/png'
    )

@app.route("/sendSensorValues", methods=["POST"])
def send_sensor_values():
    global device_count
    data = request.json
    value = data.get("sensorValue")
    device_id = data.get("deviceID")

    logger.debug(f"Received value: {value} from device: {device_id}")

    if device_id == -1:
        device_id = device_count
        device_count += 1
        response = {"msg": f"Value received: {value}", "deviceID": device_id}
    else:
        response = {"msg": f"Value received: {value} from {device_id}"}

    lon = value.get("Longitude", "NA")
    lat = value.get("Latitude", "NA")
    sensor_data_line = f"{value['VWC']},{value['TEMP']},{value['EC']},{lon},{lat},{value['Timestamp']},{device_id}\n"

    with open(DATA_FILE_NAME, "a") as f:
        f.write(sensor_data_line)

    # Only start kriging if not already running
    if not kriging_status["running"]:
        # Launch kriging in a separate process
        t = threading.Thread(target=run_kriging_with_status)
        t.start()
    else:
        logger.debug("Kriging process already running. Skipping restart.")

    return jsonify(response)


@app.route("/krigingStatus", methods=["GET"])
def get_kriging_status():
    """Return whether the kriging process is currently running and how many devices have data."""
    try:
        entries = get_different_last_values(DATA_FILE_NAME)
        device_count = len(entries)
        return jsonify(
            {"running": kriging_status["running"], "deviceCount": device_count}
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "running": kriging_status["running"],
                    "deviceCount": 0,
                    "error": str(e),
                }
            ),
            500,
        )


@app.route("/getGoal/<int:device_id>", methods=["GET"])
def get_goal(device_id):
    """Fetch lon/lat from db.txt for the specified robot/device."""
    global robots
    logger.debug(f"Robots: {robots}")
    goal_data = None
    try:
        """
        with open("db.txt", "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if fields[0] == str(device_id):
                    lon = float(fields[2])
                    lat = float(fields[1])
                    goal_data = {"lon": lon, "lat": lat}
                    break
        """
        temp = robots[device_id].next_pos
        goal_data = {"lon": temp[0], "lat": temp[1]}
        if goal_data:
            return jsonify(goal_data)
        else:
            return jsonify({"error": "No goal found for device"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/latest-holistic-plot/<int:robot_id>")
def latest_holistic_plot(robot_id):
    folder = "figures/Holistic Score Map"
    directory = os.listdir(folder)
    if len(directory) == 0:
        return "No figures yet", 404

    # Match filenames like: Robot <id> <counter>.png
    pattern = re.compile(rf"{robot_id} (\d+)\.png")
    latest_file = None
    max_counter = -1

    for filename in directory:
        match = pattern.match(filename)
        if match:
            counter = int(match.group(1))
            if counter > max_counter:
                max_counter = counter
                latest_file = filename

    if latest_file:
        return send_from_directory(folder, latest_file)

    return f"No Holistic Score Map Plots for Robot {robot_id}", 404

@app.route("/latest-path-plot/<int:robot_id>")
def latest_path_plot(robot_id):
    folder = "figures/Path To Goal"
    directory = os.listdir(folder)
    if len(directory) == 0:
        return "No figures yet", 404

    # Match filenames like: Robot <id> <counter>.png
    pattern = re.compile(rf"{robot_id} (\d+)\.png")
    latest_file = None
    max_counter = -1

    for filename in directory:
        match = pattern.match(filename)
        if match:
            counter = int(match.group(1))
            if counter > max_counter:
                max_counter = counter
                latest_file = filename

    if latest_file:
        return send_from_directory(folder, latest_file)

    return f"No Path Plots for Robot {robot_id}", 404

@app.route("/statusSnapshot/<int:device_id>", methods=["GET"])
def get_status_snapshot(device_id):
    global robots
    global DATA_FILE_NAME

    try:
        # --- Kriging & device count ---
        entries = get_different_last_values(DATA_FILE_NAME)
        device_count = len(entries)
        kriging_running = kriging_status["running"]

        # --- Goal for this device ---
        try:
            temp = robots[device_id].next_pos
            goal_data = {"lon": temp[0], "lat": temp[1]}
        except Exception as e:
            goal_data = {"error": f"Goal not found: {str(e)}"}

        # --- File lookup helpers ---
        def get_latest_plot(folder, pattern):
            try:
                fs_path = os.path.join("static", folder)
                directory = os.listdir(fs_path)
                latest_file, max_counter = None, -1

                for filename in directory:
                    match = pattern.match(filename)
                    if match:
                        counter = int(match.group(1))
                        if counter > max_counter:
                            max_counter = counter
                            latest_file = filename

                if latest_file:
                    return f"/static/{folder}/{latest_file}"  # public URL
                return None
            except Exception as e:
                print(f"Plot lookup error in {folder}: {e}")
                return None

        # --- Find the latest files ---
        holistic_url = get_latest_plot(
            "figures/Holistic Score Map",
            re.compile(rf"{device_id} (\d+)\.png")
        )
        path_url = get_latest_plot(
            "figures/Path To Goal",
            re.compile(rf"{device_id} (\d+)\.png")
        )
        zhat_url = get_latest_plot(
            "figures/Zhat",
            re.compile(r"(\d+)\.png")
        )
        zvar_url = get_latest_plot(
            "figures/Zvar",
            re.compile(r"(\d+)\.png")
        )

        # --- Build response ---
        return jsonify({
            "krigingRunning": kriging_running,
            "deviceCount": device_count,
            "goal": goal_data,
            "plots": {
                "holistic": holistic_url,
                "path": path_url,
                "zhat": zhat_url,
                "zvar": zvar_url
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/clear", methods=["GET"])
def clear_confirmation():
    """Renders a page asking for user confirmation."""
    return render_template_string(
        """
        <h2>Are you sure you want to delete all entries?</h2>
        <form action="{{ url_for('clear_confirmed') }}" method="POST">
            <button type="submit">Yes, delete everything</button>
        </form>
        <a href="/">Cancel</a>
    """
    )


@app.route("/clear/confirm", methods=["POST"])
def clear_confirmed():
    global device_count
    global logger
    
    try:
        open(DATA_FILE_NAME, "w").close()
        open("files/goals_file.txt", "w").close()
        #open("files/app.log", "w").close()
        open("files/grid_X.npy", "w").close()
        open("files/grid_Y.npy", "w").close()
        open("files/zhat_zvar.txt", "w").close()

        for filename in os.listdir("files/logs"):
            file_path = os.path.join("files/logs", filename)
            try:
                open(file_path,'w').close()

                print(f"Cleared: {filename}")
            except OSError as e:
                print(f"Error clearing {filename}: {e}")
                logger.critical(f"Error clearing {filename}: {e}")

        for folder_path in os.listdir("static/figures"):
            for filename in os.listdir("static/figures/"+folder_path):
                file_path = os.path.join("static/figures/"+folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                except OSError as e:
                    print(f"Error deleting {filename}: {e}")
        device_count = 0
        return jsonify({"msg": "System ready for New Test"})
    except Exception as e:
        return jsonify({"msg": f"Error clearing files: {e}"}), 500


@app.route("/webble", methods=["GET"])
def webble_api():
    return render_template("webble.html")

@app.route("/webblemock", methods=["GET"])
def webbleMock_api():
    return render_template("webbleMock.html")

def run_ngrok():
    command = "ngrok http --url=awaited-definite-cockatoo.ngrok-free.app 8080"
    os.system(command)


def run_server():
    app.run(port=8080)


if __name__ == "__main__":
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    ngrok_process = multiprocessing.Process(target=run_ngrok)
    ngrok_process.start()
