#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_cors import CORS
from new_kriging_traverse import visualize_initial_state, create_grid_around_robot1, run_multi_robot_exploration_with_visualization
import matplotlib.pyplot as plt
import multiprocessing
import threading
import os
import collections

app = Flask(__name__)
CORS(app)
cell_size=0.00012

device_count = 0
DATA_FILE_NAME = "sensorData.txt"

# Global status dictionary (could also be stored in a file or DB)
kriging_status = {
    "running": False
}

run_kriging_traverse_call_count=0;

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
run_init_once = False
grid_X, grid_Y = None, None
run_kriging_counter = 0
iteration_counter = 0
def run_kriging_with_status():
    """Wrapper to run kriging traverse and update status flag when all robots have submitted for current iteration."""
    global kriging_status
    global run_kriging_traverse_call_count
    global run_init_once
    global grid_X, grid_Y
    global run_kriging_counter
    global iteration_counter

    print("Starting kriging traversal process...")
    kriging_status["running"] = True
    try:
        # Read and parse sensor data
        with open('sensorData.txt', 'r') as file:
            lines = [line.strip().split(',') for line in file if line.strip()]

        # Group lines by device ID (last column)
        device_entries = collections.defaultdict(list)
        for values in lines:
            device_id = values[-1]
            device_entries[device_id].append(values)

        # Check that we have at least 3 devices and each has enough entries for this iteration
        if len(device_entries) >= 3 and all(len(entries) > iteration_counter for entries in device_entries.values()):
            robot_positions = []
            robot_vmc_values = []

            for device_id in sorted(device_entries.keys()):
                values = device_entries[device_id][iteration_counter]
                vmc = float(values[0])
                lon = float(values[3])
                lat = float(values[4])
                robot_positions.append((lon, lat))
                robot_vmc_values.append(vmc)

            # Initialize grid if this is the first run
            if not run_init_once:
                grid_size = 11
                grid_X, grid_Y = create_grid_around_robot1(robot_positions[0], grid_size, cell_size)
                run_init_once = True

            # Run kriging
            run_kriging_counter += 1
            _, Zhat = run_multi_robot_exploration_with_visualization(
                grid_X, grid_Y,
                robot_positions.copy(),
                robot_vmc_values.copy(),
                num_iterations=run_kriging_counter
            )

            visualize_initial_state(grid_X, grid_Y, robot_positions, Zhat, cell_size)
            print(f"Iteration {iteration_counter + 1} kriging completed.")
            iteration_counter += 1  # Advance to next round
        else:
            print("Not all robots have submitted measurements for this iteration yet.")

    except Exception as e:
        print(f"Kriging traverse failed: {e}")
    finally:
        kriging_status["running"] = False
        print("Kriging traversal completed.")


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
                    lon = float(fields[1])
                    lat = float(fields[2])
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
        return jsonify({"msg": "Both files cleared"})
    except Exception as e:
        return jsonify({"msg": f"Error clearing files: {e}"}), 500



@app.route("/webble", methods=["GET"])
def webble_api():
    return render_template("webble.html")

#for BLE Comms mock
@app.route("/webble2", methods=["GET"])
def webble2_api():
    return render_template("webblewithBLEMock.html")

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
