# !/usr/bin/env python3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import datetime
import multiprocessing


app = Flask(__name__)
CORS(app)

device_count = 0


@app.route('/')
def welcome():
    return 'Welcome to CORS server 😁'


#how app receives data
@app.route('/sendSensorValues', methods=['POST'])
def send_sensor_values():
    global device_count
    data = request.json
    value = data.get('sensorValue')
    device_id = data.get('deviceID')
    
    print(f"Received value: {value} from device: {device_id}")

    if device_id == -1:
        device_id = device_count
        device_count += 1
        response = {"msg": f"Value received: {value}", "deviceID": device_id}
    else:
        response = {"msg": f"Value received: {value} from {device_id}"}

    # Save the sensor value to a file
    sensor_data = f"Device ID: {device_id}, Sensor Value: {value}, Timestamp: {datetime.datetime.now().isoformat()}\n"
    with open('sensorData.txt', 'a') as f:
        f.write(sensor_data)
        print('Sensor data saved to file')
    '''
    # Call the MATLAB script to process and plot the data
    command = ['matlab', '-batch', 'plot_sensor_data()']
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"exec error: {e}")
    '''
    return jsonify(response)

#clears stored data
@app.route('/clear', methods=['DELETE'])
def clear_sensor_data():
    try:
        open('sensorData.txt', 'w').close()  # Clear the file
        print('Sensor data file cleared')
        return jsonify({"msg": "Sensor data file cleared"})
    except Exception as e:
        print(f'Error clearing file: {e}')
        return jsonify({"msg": "Error clearing file"}), 500

#BLE user interface, may move it to root 
@app.route('/webble', methods=['GET'])
def webble_api():
    return render_template('webble.html')

#runs flask server
def run_server():
    app.run(port=8080)

#To be replaced with ngrok-flask but connects flask/computer to specified web domain
def run_ngrok():
    #CREATE NGROK ACCOUNT AND REPLACE STATIC URL WITH THE ONE U GET BECAUSE THIS URL IS ASSIGNED TO ME 
    command = 'ngrok http --url=awaited-definite-cockatoo.ngrok-free.app 8080'
    os.system(command)
if __name__ == '__main__':
    
    #creating 2 multiprocesses to run the server (Server code) and the port forwarding process (connects computer to web so IP/MAC address is not needed)
    server_process= multiprocessing.Process(target=run_server)
    tunnel_process = multiprocessing.Process(target=run_ngrok)
    
    server_process.start()
    tunnel_process.start()

    server_process.join()
    tunnel_process.join()