#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from KrigingTraverse import run_kriging_traverse
import matplotlib.pyplot as plt
import multiprocessing
import threading
import os

app = Flask(__name__)
CORS(app)

device_count = 0
DATA_FILE_NAME = "sensorData.txt"

# Global status dictionary (could also be stored in a file or DB)
kriging_status = {
    "running": False
}


def run_kriging_with_status():
    """Wrapper to run kriging traverse and update status flag."""
    global kriging_status
    print("Starting kriging traversal process...")
    kriging_status["running"] = True

    try:
        run_kriging_traverse()
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

    sensor_data_line = f"{value['VWC']},{value['TEMP']},{value['EC']},{value['Longitude']},{value['Latitude']},{value['Timestamp']},{device_id}\n"
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
    """Return whether the kriging process is currently running."""
    return jsonify({"running": kriging_status["running"]})

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


@app.route("/clear", methods=["DELETE"])
def clear_sensor_data():
    try:
        open(DATA_FILE_NAME, "w").close()
        print("Sensor data file cleared")
        return jsonify({"msg": "Sensor data file cleared"})
    except Exception as e:
        print(f"Error clearing file: {e}")
        return jsonify({"msg": "Error clearing file"}), 500


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
    # ngrok_process = multiprocessing.Process(target=run_ngrok)
    # ngrok_process.start()
