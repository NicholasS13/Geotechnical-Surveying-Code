#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_cors import CORS
from kriging_traverse import *
import multiprocessing
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger.debug('This message should go to the log file')
logger.info('So should this')
logger.warning('And this, too')
logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
app = Flask(__name__)
CORS(app)
cell_size=0.0001

Zhat = None
robots:list = list()

device_count = 0
DATA_FILE_NAME = "sensor.txt"

sensor_file = "sensor.txt"
grid_x_file = "grid_X.npy"
grid_y_file = "grid_Y.npy"
# Global status dictionary (could also be stored in a file or DB)
kriging_status = {
    "running": False
}
grid_X, grid_Y = None,None;


def get_different_last_values(file_path):
    last_values = set()
    different_entries = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line by commas
            entries = line.strip().split(',')
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
    logger.debug("Starting kriging traversal process...")
    kriging_status["running"] = True
    
    save_grid()
    if len(get_different_last_values(DATA_FILE_NAME))>=3:
        try:
            grid_X = np.load(grid_x_file)
            grid_Y = np.load(grid_y_file)
            
            logger.debug("Grid X ", grid_X)
            logger.debug("Grid Y ", grid_Y)
            vmc_map = np.full_like(grid_X, np.nan)
            all_positions = []
            all_vmc = []
            latest_robot_data = {}
            visited_cells = set()
            
            with open(DATA_FILE_NAME, "r") as f:
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
        except Exception as e:
            logger.debug(f"Kriging traverse failed: {e}")
        finally:
            kriging_status["running"] = False
            logger.debug("Kriging traversal completed.")

def save_grid():
    global cell_size
    try:
        with open(sensor_file, 'r') as f:
            first_line = f.readline().strip().split(',')
            lon = float(first_line[3])
            lat = float(first_line[4])
            initial_pos = (lon, lat)

        # --- Step 2: Create grid around initial position ---
        grid_X1, grid_Y1 = create_grid(initial_pos, grid_size=11, cell_size_lat=0.00012)

        # --- Step 3: Save to disk ---
        np.save(grid_x_file, grid_X1)
        np.save(grid_y_file, grid_Y1)
            
        return jsonify({"grid_X": grid_X, "grid_Y": grid_Y})
    except Exception as e:
        logger.debug(f"Exception in /save_grid: {e}")  # Logs error to your Flask console
        return jsonify({'error': str(e)}), 500
    
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

    lon = value.get('Longitude', 'NA')
    lat = value.get('Latitude', 'NA')
    sensor_data_line = f"{value['VWC']},{value['TEMP']},{value['EC']},{lon},{lat},{value['Timestamp']},{device_id}\n"

    with open(DATA_FILE_NAME, "a") as f:
        f.write(sensor_data_line)

    # Only start kriging if not already running
    if not kriging_status["running"]:
        # Launch kriging in a separate process
        p = multiprocessing.Process(target=run_kriging_with_status)
        p.start()
    else:
        logger.debug("Kriging process already running. Skipping restart.")

    return jsonify(response)

@app.route("/krigingStatus", methods=["GET"])
def get_kriging_status():
    """Return whether the kriging process is currently running and how many devices have data."""
    try:
        entries = get_different_last_values(DATA_FILE_NAME)
        device_count = len(entries)
        return jsonify({
            "running": kriging_status["running"],
            "deviceCount": device_count
        })
    except Exception as e:
        return jsonify({
            "running": kriging_status["running"],
            "deviceCount": 0,
            "error": str(e)
        }), 500


@app.route("/getGoal/<int:device_id>", methods=["GET"])
def get_goal(device_id):
    """Fetch lon/lat from db.txt for the specified robot/device."""
    global robots
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


@app.route('/clear', methods=['GET'])
def clear_confirmation():
    """Renders a page asking for user confirmation."""
    return render_template_string('''
        <h2>Are you sure you want to delete all entries?</h2>
        <form action="{{ url_for('clear_confirmed') }}" method="POST">
            <button type="submit">Yes, delete everything</button>
        </form>
        <a href="/">Cancel</a>
    ''')

@app.route('/clear/confirm', methods=['POST'])
def clear_confirmed():
    try:
        open(DATA_FILE_NAME, "w").close()
        open("db.txt", "w").close()
        open("points.txt",'w').close()
        return jsonify({"msg": "Both files cleared"})
    except Exception as e:
        return jsonify({"msg": f"Error clearing files: {e}"}), 500

@app.route('/get_zhat')
def get_zhat():
    global Zhat
    return jsonify({'zHat': Zhat})
@app.route("/get_visited_cells", methods=["GET"])
def get_visited():
    points = set()
    with open('points.txt', 'r') as f:
        points = {ast.literal_eval(line.strip()) for line in f}
    return jsonify({"Points": list(points)})

@app.route("/webble", methods=["GET"])
def webble_api():
    return render_template("webble.html")


def run_ngrok():
    command = 'ngrok http --url=awaited-definite-cockatoo.ngrok-free.app 8080'
    os.system(command)


def run_server():
    app.run(port=8080)

if __name__ == "__main__":
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    ngrok_process = multiprocessing.Process(target=run_ngrok)
    ngrok_process.start()
