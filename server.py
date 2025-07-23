#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_cors import CORS
from new_kriging_traverse import visualize_initial_state, create_grid_around_robot1, run_multi_robot_exploration_with_visualization
import matplotlib.pyplot as plt
import multiprocessing
import threading
import os
import numpy as np

app = Flask(__name__)
CORS(app)
cell_size=0.0001

Zhat = None


device_count = 0
DATA_FILE_NAME = "sensorData.txt"

# Global status dictionary (could also be stored in a file or DB)
kriging_status = {
    "running": False
}


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
run_init_once = False
grid_X, grid_Y = None,None;

run_kriging_counter = 0
def run_kriging_with_status():
    """Wrapper to run kriging traverse and update status flag."""
    global kriging_status
    global run_init_once
    global Zhat
    #global grid_X 
    #global grid_Y
    global run_kriging_counter
    run_kriging_counter = run_kriging_counter+1
    print("Starting kriging traversal process...")
    kriging_status["running"] = True
    if (len(get_different_last_values("sensorData.txt"))>=3):
        try:
            # Define the grid size and cell size
            # Example run
            grid_size = 11
            robot_vmc_values = []
            robot_positions = []
            robot_positions_1 = []
            robot_vmc_values_1 = []
            lowest_entries = {}
            with open('sensorData.txt', 'r') as file:
                for line in file:
                    values = line.strip().split(',')
                    # Get the unique value at index -1
                    unique_value = values[-1]
                    # Convert VMC to float for comparison
                    vmc = float(values[0])
                    lowest_entries[unique_value] = (vmc, values)
                    
                    #adds all robot vmc and coordinate values
                    robot_positions_1.append((values[4], float(values[3])))
                    robot_vmc_values_1.append(vmc)
            # Now process the latest unique entries (each individual device)
            for _, values in lowest_entries.values():# _ is the
                vmc = float(values[0])
                lon = float(values[3])
                lat = float(values[4])
                robot_positions.append((lat, lon))
                robot_vmc_values.append(vmc)
            #if not run_init_once:
            grid_X = np.load('grid_X.npy')
            grid_Y = np.load('grid_Y.npy')
                #grid_X, grid_Y = create_grid_around_robot1(robot_positions[0], grid_size, cell_size)
            #    run_init_once = True
            _ ,Zhat = run_multi_robot_exploration_with_visualization(grid_X, grid_Y, robot_positions.copy(), robot_vmc_values.copy(), cell_size, robot_positions_1,robot_vmc_values_1, num_iterations=0)
            visualize_initial_state(grid_X, grid_Y, robot_positions, Zhat, cell_size)   
        except Exception as e:
            print(f"Kriging traverse failed: {e}")
        finally:
            kriging_status["running"] = False
            print("Kriging traversal completed.")

@app.route('/save_grid', methods=['POST'])
def save_grid():
    global cell_size
    try:
        data = request.get_json()
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'Missing lat or lon in JSON'}), 400
        
        # Parse floats safely
        lat = float(data['lat'])
        lon = float(data['lon'])
        
        # Pass coordinates in (lat, lon) order if that's what create_grid_around_robot1 expects
        #grid_X, grid_Y = create_grid_around_robot1((lat, lon), 11, cell_size)
        
        grid_X, grid_Y = create_grid_around_robot1((lat, lon), 11, cell_size)

        # Save as npy files or serialize as JSON lists (not recommended for very large arrays)
        np.save('grid_X.npy', grid_X)
        np.save('grid_Y.npy', grid_Y)
            
        return jsonify({"grid_X": grid_X, "grid_Y": grid_Y})
    except Exception as e:
        print(f"Exception in /save_grid: {e}")  # Logs error to your Flask console
        return jsonify({'error': str(e)}), 500
    
@app.route("/sendSensorValues", methods=["POST"])
def send_sensor_values():
    global device_count
    data = request.json
    value = data.get("sensorValue")
    device_id = data.get("deviceID")

    print(f"Received value: {value} from device: {device_id}")

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
        print("Kriging process already running. Skipping restart.")

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
    goal_data = None
    try:
        with open("db.txt", "r") as f:
            for line in f:
                fields = line.strip().split(",")
                if fields[0] == str(device_id):
                    lon = float(fields[2])
                    lat = float(fields[1])
                    goal_data = {"lon": lon, "lat": lat}
                    break
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
