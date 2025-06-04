# !/usr/bin/env python3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pykrig_implementation import kriging_traverse_maze_decisive_only_when_choose_goal
import os
from datetime import datetime
import multiprocessing
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)

device_count = 0

DATA_FILE_NAME = "sensorData.txt"
# Kriging Maze Stuff
def load_csv_data():
    """_summary_

    Args:
        filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    parsed_data = []
    with open(DATA_FILE_NAME, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 6: 
                continue
            vwc = float(parts[0])
            temp = float(parts[1])
            ec = float(parts[2])
            lon, lat = map(float, parts[3:5])
            timestamp = parts[5]
            robot_id = parts[6] if len(parts) > 6 else "0"
            parsed_data.append(
                {
                    "VWC": vwc,
                    "TEMP": temp,
                    "EC": ec,
                    "Longitude": lon,
                    "Latitude": lat,
                    "Timestamp": timestamp,
                    "RobotID": robot_id,
                }
            )
    return pd.DataFrame(parsed_data)


def map_coords_to_grid(df, grid_size):
    """_summary_

    Args:
        df (_type_): _description_
        grid_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
    min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
    lat_scale = (df["Latitude"] - min_lat) / (max_lat - min_lat + 1e-8)
    lon_scale = (df["Longitude"] - min_lon) / (max_lon - min_lon + 1e-8)
    df["GridRow"] = (lat_scale * (grid_size - 1)).astype(int)
    df["GridCol"] = (lon_scale * (grid_size - 1)).astype(int)
    return df


# Create the true_map using VWC as the value
def create_true_map(df, grid_size):
    """_summary_

    Returns:
        _type_: _description_
    """
    true_map = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    for _, row in df.iterrows():
        r, c = row["GridRow"], row["GridCol"]
        true_map[r, c] += row["VWC"]
        counts[r, c] += 1
    true_map[counts > 0] /= counts[counts > 0]  # Average overlapping values
    return true_map


def run_traverse_maze():
    df = load_csv_data()
    grid_size = 50
    df = map_coords_to_grid(df, grid_size)
    true_map = create_true_map(df, grid_size)

    # Parameters for the simulation
    num_robots = df["RobotID"].nunique()
    last_err = kriging_traverse_maze_decisive_only_when_choose_goal(
        true_map=true_map,
        weight_expected_value=1.0,
        weight_uncertainty=1.0,
        weight_center_preference=1.0,
        weight_closeness=1.0,
        weight_prefer_existing_goal=1.0,
        num_iter=20,
        num_robots=num_robots,
        which_map=1,
        val_penalize_visited_locations=1.0,
        multirobot_starting_strategy=2,
        env_params=None,
        dir_target="./output_results",
        step_cost_multiplier=1.0
    )
    print("Final Error:", last_err)

# Server stuff
@app.route("/")
def welcome():
    """_summary_

    Returns:
        _type_: _description_
    """
    run_traverse_maze()
    return "Welcome to CORS server 😁"


# how app receives data
@app.route("/sendSensorValues", methods=["POST"])
def send_sensor_values():
    """_summary_

    Returns:
        _type_: _description_
    """
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

    # Save the sensor value to a file
    sensor_data = f"Device ID: {device_id}, Sensor Value: {value}, Timestamp: {datetime.now().isoformat()}\n"
    with open("sensorData.txt", "a") as f:
        f.write(sensor_data)
        print("Sensor data saved to file")
    """
    # Call the MATLAB script to process and plot the data
    command = ['matlab', '-batch', 'plot_sensor_data()']
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"exec error: {e}")
    """
    return jsonify(response)


# clears stored data
@app.route("/clear", methods=["DELETE"])
def clear_sensor_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    try:
        open(DATA_FILE_NAME, "w").close()  # Clear the file
        print("Sensor data file cleared")
        return jsonify({"msg": "Sensor data file cleared"})
    except Exception as e:
        print(f"Error clearing file: {e}")
        return jsonify({"msg": "Error clearing file"}), 500


# BLE user interface, may move it to root
@app.route("/webble", methods=["GET"])
def webble_api():
    """_summary_

    Returns:
        _type_: _description_
    """
    return render_template("webble.html")


# runs flask server
def run_server():
    """_summary_
    """
    app.run(port=8080)


if __name__ == "__main__":

    # creating 2 multiprocesses to run the server (Server code) and the port forwarding process (connects computer to web so IP/MAC address is not needed)
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()


    from pyngrok import ngrok
    '''
    # Replace 'your-static-domain.ngrok-free.app' with your actual domain
    domain = "awaited-definite-cockatoo.ngrok-free.app"
    port = 8080  # Replace with the port your app is running on

    # Start a tunnel with the static domain
    tunnel = ngrok.connect(addr=port, domain=domain)

    # Print the public URL
    print(f"Public URL: {tunnel.public_url}")

    # Keep the tunnel running until you stop the script
    try:
        input("Press Enter to stop the application...\n")
    except KeyboardInterrupt:
        print(" Shutting down server.")
    finally:
        ngrok.kill()
        quit();
    ''' 
