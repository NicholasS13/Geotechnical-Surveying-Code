let deviceHeading = 0;
let goalBearing = null;
let deviceID = -1;
let currentRobotId = deviceID;
let devicesWithReadings = new Set();
let lastSensorTime = 0;
const SENSOR_UPDATE_INTERVAL_MS = 2000;
const goalReachThreshold = 5;

const measurementStatusEl = document.getElementById("measurementStatus");
const readingStatusBox = document.getElementById("readingStatusBox");
const mainDeviceIdEl = document.getElementById("mainDeviceId");
const coordinateD = document.getElementById("coordinateDIV");

const SERVICE_UUID = '91bad492-b950-4226-aa2b-4ede9fa42f59';
const CHAR_UUID_SENSOR_DATA = '98e025d3-23e5-4b62-9916-cc6c330c84ac';
const CHAR_UUID_COMMAND = 'f78ebbff-c8b7-4107-93de-889a6a06d408';

let commandCharacteristic;

document.getElementById('enableCompassBtn').addEventListener('click', () => {
  setupDeviceOrientation();
  document.getElementById('enableCompassBtn').style.display = 'none';
});

document.getElementById('connect').addEventListener('click', async () => {
  try {
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [SERVICE_UUID] }]
    });

    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);

    const sensorDataCharacteristic = await service.getCharacteristic(CHAR_UUID_SENSOR_DATA);
    commandCharacteristic = await service.getCharacteristic(CHAR_UUID_COMMAND);

    await sensorDataCharacteristic.startNotifications();
    sensorDataCharacteristic.addEventListener('characteristicvaluechanged', handleSensorData);

    console.log('Connected to ESP32 S3 Feather');
    document.getElementById("connect").style.display = 'none';
  } catch (error) {
    console.error('Error connecting to device:', error);
  }
});

document.getElementById('viewDataHistory').addEventListener('click', () => {
  const sensorDataHistory = document.getElementById('sensorDataHistory');
  sensorDataHistory.style.display =
    sensorDataHistory.style.display === 'none' ? 'block' : 'none';
});

document.getElementById("cannotgo").addEventListener("click", () => {
  const newDirection = confirm("Is it shorter to go right?") ? 90 : 270;
  updateArrow(newDirection);
});

document.getElementById('submitCommandBtn').addEventListener('click', () => {
  sendCommand(document.getElementById("commandInput").value);
});

document.getElementById('getReadingBtn').addEventListener('click', async () => {
  try {
    if (!commandCharacteristic) {
      console.error("BLE command characteristic not available. Connect first.");
      return;
    }
    const command = "get reading";
    const commandData = new TextEncoder().encode(command);
    await commandCharacteristic.writeValue(commandData);
    console.log("Sent command to BLE device:", command);
    alert("Command sent to ESP32")
  } catch (err) {
    console.error("Failed to send get reading command:", err);
    alert("command not sent")
  }
});


function setupDeviceOrientation() {
  if (
    typeof DeviceOrientationEvent !== "undefined" &&
    typeof DeviceOrientationEvent.requestPermission === "function"
  ) {
    DeviceOrientationEvent.requestPermission()
      .then((response) => {
        if (response === "granted") {
          window.addEventListener("deviceorientation", handleOrientation, true);
        }
      })
      .catch(console.error);
  } else {
    window.addEventListener("deviceorientation", handleOrientation, true);
  }
}

function handleOrientation(event) {
  if (event.absolute || event.webkitCompassHeading !== undefined) {
    deviceHeading = event.webkitCompassHeading || 0;
  } else if (event.alpha !== null) {
    deviceHeading = 360 - event.alpha;
  }

  if (goalBearing !== null) {
    updateArrow(goalBearing);
  } else {
    updateArrow(deviceHeading);
  }
}

function updateArrow(goalBearingInput) {
  goalBearing = goalBearingInput;
  const relativeAngle = (goalBearing - deviceHeading + 360) % 360;
  document.getElementById("arrow").style.transform = `rotate(${relativeAngle}deg)`;
}

function sendCommand(command) {
  if (commandCharacteristic) {
    const commandData = new TextEncoder().encode(command);
    commandCharacteristic.writeValue(commandData).catch(console.error);
  }
}

function getCachedPosition() {
  const posEl = document.getElementById("currentPosition");
  const lat = parseFloat(posEl.dataset.lat);
  const lon = parseFloat(posEl.dataset.lon);
  if (isNaN(lat) || isNaN(lon)) return null;
  return { lat, lon };
}

function handleSensorData(event) {
  const now = Date.now();
  if (now - lastSensorTime < SENSOR_UPDATE_INTERVAL_MS) return;
  lastSensorTime = now;

  const pos = getCachedPosition();
  if (!pos) {
    coordinateD.innerHTML = "Location not ready.";
    return;
  }

  const { lat, lon } = pos;
  coordinateD.innerHTML = `${lat},${lon}`;

  const value = event.target.value;
  const data = new TextDecoder().decode(value);
  const values = data.split(",");
  const timestamp = new Date().toLocaleTimeString();

  document.getElementById("value1").innerText = values[0];
  document.getElementById("value2").innerText = values[1];
  document.getElementById("value3").innerText = (values[2] / 1000).toFixed(3);

  document.getElementById("sensorDataHistory").innerHTML =
    `${timestamp} - Sensor Data: ${data}<br>` +
    document.getElementById("sensorDataHistory").innerHTML;

  document.getElementById("currentPosition").textContent = `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`;
  document.getElementById("currentPosition").dataset.lat = lat;
  document.getElementById("currentPosition").dataset.lon = lon;

  fetch("/sendSensorValues", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sensorValue: {
        VWC: values[0],
        TEMP: values[1],
        EC: values[2],
        Longitude: lon,
        Latitude: lat,
        Timestamp: timestamp,
      },
      deviceID: deviceID,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.deviceID !== undefined && data.deviceID !== -1) {
        if (deviceID === -1) {
          deviceID = data.deviceID;
          mainDeviceIdEl.textContent = deviceID;
          devicesWithReadings.add(deviceID);
        } else {
          deviceID = data.deviceID;
        }
        devicesWithReadings.add(deviceID);
        updateMeasurementStatus();
      }

      fetch("/krigingStatus")
        .then((r) => r.json())
        .then((kriging) => {
          if (!kriging.running) {
            document.getElementById("loading").style.display = "none";
            document.querySelector(".arrow-container").style.display = "flex";
            updateAllGoalsAndCompass(lat, lon);
            updateMeasurementStatus();
          }
        });
    })
    .catch(console.error);
}

navigator.geolocation.watchPosition(
  (position) => {
    const lat = position.coords.latitude;
    const lon = position.coords.longitude;
    const posEl = document.getElementById("currentPosition");
    posEl.textContent = `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`;
    posEl.dataset.lat = lat;
    posEl.dataset.lon = lon;

    if (deviceID !== -1) {
      updateAllGoalsAndCompass(lat, lon);
    }
  },
  (err) => {
    console.error("Geolocation error:", err);
  },
  { enableHighAccuracy: true, timeout: 3000, maximumAge: 1000 }
);

async function updateAllGoalsAndCompass(currentLat, currentLon) {
  const pos = getCachedPosition();
  if (!pos || deviceID === -1) return;

  const { lat, lon } = pos;

  const goal = await fetchGoalForDevice(deviceID);
  if (!goal) return;

  document.getElementById("goalPosition").textContent =
    `Lat: ${goal.lat.toFixed(6)}, Lon: ${goal.lon.toFixed(6)}`;

  const bearing = calculateBearing(lat, lon, goal.lat, goal.lon);
  const distance = haversineDistance(lat, lon, goal.lat, goal.lon);
  updateArrow(bearing);

  document.getElementById("distanceBox").textContent = `Distance to goal: ${distance.toFixed(1)} meters`;

  updateReadingStatus(distance);
}

function updateReadingStatus(distance) {
  if (distance <= goalReachThreshold) {
    readingStatusBox.textContent = "✅ You have reached your goal. Please take a reading.";
    readingStatusBox.style.backgroundColor = "#d4edda";
    readingStatusBox.style.color = "#155724";
    readingStatusBox.style.borderColor = "#c3e6cb";
  } else {
    readingStatusBox.textContent = "🚶 Need to travel closer to your goal. Do not take reading";
    readingStatusBox.style.backgroundColor = "#f0f8ff";
    readingStatusBox.style.color = "#004085";
    readingStatusBox.style.borderColor = "#b8daff";
  }
}

function calculateBearing(lat1, lon1, lat2, lon2) {
  const toRad = (x) => (x * Math.PI) / 180;
  const dLon = toRad(lon2 - lon1);
  lat1 = toRad(lat1);
  lat2 = toRad(lat2);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) -
            Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toRad = (x) => x * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
            Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

async function updateMeasurementStatus() {
  try {
    const res = await fetch("/krigingStatus");
    const data = await res.json();
    const count = data.deviceCount || 0;
    if (count < 3) {
      measurementStatusEl.textContent = `Only ${count} device(s) reporting. Please take more measurements (need at least 3).`;
    } else {
      measurementStatusEl.textContent = "3 or more devices reporting. You may proceed with measurements.";
    }
  } catch (e) {
    measurementStatusEl.textContent = "Could not load device count.";
    console.error("Measurement status error:", e);
  }
}

async function fetchGoalForDevice(id) {
  try {
    const response = await fetch(`/getGoal/${id}`);
    if (!response.ok) throw new Error("No goal found");
    return await response.json();
  } catch (e) {
    console.warn(`Goal fetch failed for device ${id}:`, e);
    return null;
  }
}

function generateMockSensorData() {
  const vmc = document.getElementById('VMCInput').value || "1";
  const ec = (Math.random() * 2000).toFixed(2);
  const temp = (20 + Math.random() * 10).toFixed(2);
  const mockData = `${vmc},${temp},${ec}`;
  const encoder = new TextEncoder();
  const fakeEvent = {
    target: {
      value: encoder.encode(mockData)
    }
  };
  handleSensorData(fakeEvent);
}

document.querySelector('.arrow-container').addEventListener('click', async () => {
  const arrowContainer = document.querySelector('.arrow-container');
  arrowContainer.style.display = 'none';
  document.getElementById('map').style.display = 'block';

  const pos = getCachedPosition();
  if (!pos) return;
  const { lat: currentLat, lon: currentLon } = pos;

  const goal = await fetchGoalForDevice(deviceID);
  if (!goal) return;

  const map = L.map('map').setView([currentLat, currentLon], 20);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 22,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri'
  }).addTo(map);

  L.marker([currentLat, currentLon]).addTo(map).bindPopup("<b>Your Location</b>").openPopup();
  L.marker([goal.lat, goal.lon]).addTo(map).bindPopup("<b>Goal Location</b>");

  const bounds = L.latLngBounds([[currentLat, currentLon], [goal.lat, goal.lon]]).pad(0.3);
  map.fitBounds(bounds);
});

function sendPosition() {
  const pos = getCachedPosition();
  if (!pos) {
    alert('Coordinates not ready');
    return;
  }
  fetch('/save_grid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pos)
  }).then(response => response.json())
    .then(data => alert('Grid saved: X=' + data.grid_X + ', Y=' + data.grid_Y));
}


function updateRobotPlots() {
  const input = document.getElementById("robotIdInput").value;
  currentRobotId = input !== "" ? input : deviceID;

  document.getElementById("holisticRobotId").textContent = currentRobotId;
  document.getElementById("pathRobotId").textContent = currentRobotId;

  updateImages();
}

function updateImages() {
  const timestamp = new Date().getTime();
  document.getElementById("holisticPlot").src = `/latest-holistic-plot/${currentRobotId}?cb=${timestamp}`;
  document.getElementById("pathPlot").src = `/latest-path-plot/${currentRobotId}?cb=${timestamp}`;
}

async function fetchAndApplyDeviceSnapshot() {
  try {
    const res = await fetch('/device-snapshot');
    if (!res.ok) throw new Error('Failed to fetch snapshot');
    const snapshot = await res.json();

    // Example snapshot structure now WITHOUT deviceID:
    // { lat: 40.12345, lon: -74.12345, vmc: 5.4, temp: 21.1, ec: 1250 }

    // Update current position display and dataset
    document.getElementById("currentPosition").textContent = `Lat: ${snapshot.lat.toFixed(6)}, Lon: ${snapshot.lon.toFixed(6)}`;
    document.getElementById("currentPosition").dataset.lat = snapshot.lat;
    document.getElementById("currentPosition").dataset.lon = snapshot.lon;

    // Update sensor values (VMC still mocked or from snapshot)
    document.getElementById("value1").innerText = snapshot.vmc;
    document.getElementById("value2").innerText = snapshot.temp;
    document.getElementById("value3").innerText = (snapshot.ec / 1000).toFixed(3);

    // Update sensor history with timestamp
    const timestamp = new Date().toLocaleTimeString();
    document.getElementById("sensorDataHistory").innerHTML =
      `${timestamp} - Sensor Snapshot: VWC=${snapshot.vmc}, TEMP=${snapshot.temp}, EC=${snapshot.ec}<br>` +
      document.getElementById("sensorDataHistory").innerHTML;

    // Update UI based on position & status
    updateMeasurementStatus();
    updateAllGoalsAndCompass(snapshot.lat, snapshot.lon);

  } catch (e) {
    console.warn('Failed to fetch device snapshot:', e);
  }
}
setInterval(fetchAndApplyDeviceSnapshot, 10000);//10s