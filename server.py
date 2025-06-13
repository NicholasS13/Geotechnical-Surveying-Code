
#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from KrigingTraverse import kriging_traverse_maze, grid_to_longlat
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import threading
import os

app = Flask(__name__)
CORS(app)

device_count = 0
DATA_FILE_NAME = "sensorData.txt"

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

    # Append sensor data line in CSV style to sensorData.txt
    # Example line: "VWC,TEMP,EC,Lon,Lat,Timestamp,RobotID"
    sensor_data_line = f"{value['VWC']},{value['TEMP']},{value['EC']},{value['Longitude']},{value['Latitude']},{value['Timestamp']},{device_id}\n"
    with open(DATA_FILE_NAME, "a") as f:
        f.write(sensor_data_line)
        print("Sensor data saved to file")

    return jsonify(response)


@app.route("/clear", methods=["DELETE"])
def clear_sensor_data():
    try:
        open(DATA_FILE_NAME, "w").close()
        print("Sensor data file cleared")
        return jsonify({"msg": "Sensor data file cleared"})
    except Exception as e:
        print(f"Error clearing file: {e}")
        return jsonify({"msg": "Error clearing file"}), 500
def runTraverseFunc():
    var_map, next_goals, bounds = kriging_traverse_maze("sensorData.txt")
    lon_min, lon_max, lat_min, lat_max = bounds

    print("\nNext Robot Goals:")
    for robot, (i, j) in next_goals.items():
        lon, lat = grid_to_longlat(i, j, (50, 50), lon_min, lon_max, lat_min, lat_max)
        print(f"Robot {robot} → Cell ({i}, {j}) → GPS ({lon:.6f}, {lat:.6f})")

    plt.imshow(var_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Kriging Variance")
    plt.title("Kriging Variance Map")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.savefig("results/Figure_1_python.png")
    #plt.show()
    


@app.route("/runKriging", methods=["GET"])
def runK():
    thread = threading.Thread(target=runTraverseFunc)
    thread.start();
    return "RUNNING RUN_TRAVERSE"

@app.route("/webble", methods=["GET"])
def webble_api():
    return render_template("webble.html")

#To be replaced with ngrok-flask but connects flask/computer to specified web domain
def run_ngrok():
    #CREATE NGROK ACCOUNT AND REPLACE STATIC URL WITH THE ONE U GET BECAUSE THIS URL IS ASSIGNED TO ME 
    command = 'ngrok http --url=awaited-definite-cockatoo.ngrok-free.app 8080'
    os.system(command)

def run_server():
    app.run(port=8080)


if __name__ == "__main__":
    server_process = multiprocessing.Process(target=run_server)
    #ngrok_process = multiprocessing.Process(target=run_ngrok) # Port Forwarding to static domain
    server_process.start()
