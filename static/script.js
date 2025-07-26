let deviceHeading = 0;
let goalBearing = null;
let deviceID = -1;
let devicesWithReadings = new Set();

const measurementStatusEl = document.getElementById("measurementStatus");
const readingStatusBox = document.getElementById("readingStatusBox");
const mainDeviceIdEl = document.getElementById("mainDeviceId");
const coordinateD = document.getElementById("coordinateDIV");

const SERVICE_UUID = '91bad492-b950-4226-aa2b-4ede9fa42f59';
const CHAR_UUID_SENSOR_DATA = '98e025d3-23e5-4b62-9916-cc6c330c84ac';
const CHAR_UUID_COMMAND = 'f78ebbff-c8b7-4107-93de-889a6a06d408';

let commandCharacteristic;
let pollingInterval = null;
let lastDistanceToGoal = null;
const goalReachThreshold = 5; // meters

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

document.getElementById('getReadingBtn').addEventListener('click', generateMockSensorData);

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

function handleSensorData(event) {
  const value = event.target.value;
  const data = new TextDecoder().decode(value);
  const now = new Date();
  const dateTimeString = now.toLocaleTimeString();

  getGeoData((geoError, position) => {
    if (geoError) {
      console.error("Geolocation error:", geoError);
      coordinateD.innerHTML = `ERROR: ${geoError.message}`;
      return;
    }

    const lat = position.coords.latitude;
    const lon = position.coords.longitude;
    coordinateD.innerHTML = `${lat},${lon}`;
    const values = data.split(",");

    document.getElementById("value1").innerText = values[0];
    document.getElementById("value2").innerText = values[1];
    document.getElementById("value3").innerText = (values[2] / 1000).toFixed(3);

    document.getElementById("sensorDataHistory").innerHTML =
      `${dateTimeString} - Sensor Data: ${data}<br>` +
      document.getElementById("sensorDataHistory").innerHTML;

    document.getElementById("currentPosition").textContent = `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`;

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
          Timestamp: dateTimeString,
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
            updateAllGoalsAndCompass(lat, lon);
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
              clearInterval(pollingInterval);
              document.getElementById("loading").style.display = "none";
              document.querySelector(".arrow-container").style.display = "flex";
              updateAllGoalsAndCompass(lat, lon);
              updateMeasurementStatus();
            }
          });
      })
      .catch(console.error);
  });
}
//Temp overrides for debugging

function getCustomPosition() {
  const lat = parseFloat(document.getElementById("latitudeInput").value);
  const lon = parseFloat(document.getElementById("longitudeInput").value);

  return {
    coords: {
      latitude: lat,
      longitude: lon,
      accuracy: 5,   // fixed value; adjust if you want
      altitude: null,
      altitudeAccuracy: null,
      heading: null,
      speed: null
    },
    timestamp: Date.now()
  };
}

// Override getCurrentPosition
navigator.geolocation.getCurrentPosition = function(success, error) {
  const position = getCustomPosition();

  if (!isNaN(position.coords.latitude) && !isNaN(position.coords.longitude)) {
    success(position);
  } else if (typeof error === "function") {
    error({ code: 1, message: "No position set in input fields." });
  }
};

// Override watchPosition
navigator.geolocation.watchPosition = function(success, error) {
  function dispatchPosition() {
    const position = getCustomPosition();
    if (!isNaN(position.coords.latitude) && !isNaN(position.coords.longitude)) {
      success(position);
    } else if (typeof error === "function") {
      error({ code: 1, message: "No position set in input fields." });
    }
  }

  // Hook up onchange listeners to live-update as fields change
  document.getElementById("latitudeInput").addEventListener("input", dispatchPosition);
  document.getElementById("longitudeInput").addEventListener("input", dispatchPosition);

  // Optionally, dispatch once on call
  dispatchPosition();

  // Return a mock watch ID
  return 1;
};


//end of temp overrides
function getGeoData(callback) {
  if (!navigator.geolocation) {
    callback(new Error("Geolocation not supported"), null);
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (position) => callback(null, position),
    (error) => callback(error, null),
    { enableHighAccuracy: true, timeout: 5000, maximumAge: 1000 }
  );
}

function startLocationTracking() {
  if (!navigator.geolocation) return;

  navigator.geolocation.watchPosition(
    (position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      document.getElementById("currentPosition").innerText = `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`;
      if (deviceID !== -1) {
        updateAllGoalsAndCompass(lat, lon);
      }
    },
    (err) => {
      console.error("Tracking error:", err);
    },
    { enableHighAccuracy: true, timeout: 5000, maximumAge: 1000 }
  );
}

async function updateAllGoalsAndCompass(currentLat, currentLon) {
  if (deviceID === -1) return;

  const goal = await fetchGoalForDevice(deviceID);
  if (!goal) return;

  const goalLat = goal.lat;
  const goalLon = goal.lon;

  document.getElementById("goalPosition").textContent =
    `Lat: ${goalLat.toFixed(6)}, Lon: ${goalLon.toFixed(6)}`;

  const bearing = calculateBearing(currentLat, currentLon, goalLat, goalLon);
  const distance = haversineDistance(currentLat, currentLon, goalLat, goalLon);
  updateArrow(bearing);

  document.getElementById("distanceBox").textContent = `Distance to goal: ${distance.toFixed(1)} meters`;

  lastDistanceToGoal = distance;
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

// When the compass arrow is clicked, show a satellite map
document.querySelector('.arrow-container').addEventListener('click', async () => {
  const arrowContainer = document.querySelector('.arrow-container');
  arrowContainer.style.display = 'none'; // Hide the compass
  document.getElementById('map').style.display = 'block'; // Show the map div

  // Get current position again to ensure up to date
  getGeoData(async (geoError, position) => {
    if (geoError) {
      console.error("Geolocation error:", geoError);
      return;
    }
    const currentLat = position.coords.latitude;
    const currentLon = position.coords.longitude;

    const goal = await fetchGoalForDevice(deviceID);
    if (!goal) {
      console.error("Goal not available");
      return;
    }
    const goalLat = goal.lat;
    const goalLon = goal.lon;

    // Initialize Leaflet map
    const map = L.map('map').setView([currentLat, currentLon], 20); // zoom 20: very close range

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 22,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // Add satellite layer (Esri satellite)
    const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri'
    }).addTo(map);

    // Markers
    const currentMarker = L.marker([currentLat, currentLon]).addTo(map);
    currentMarker.bindPopup("<b>Your Location</b>").openPopup();

    const goalMarker = L.marker([goalLat, goalLon]).addTo(map);
    goalMarker.bindPopup("<b>Goal Location</b>");

    // Fit bounds tightly around both points
    const bounds = L.latLngBounds([
      [currentLat, currentLon],
      [goalLat, goalLon]
    ]).pad(0.3);
    map.fitBounds(bounds);
  });
});
function sendPosition() {
    const text = document.getElementById('currentPosition').textContent;
    const regex = /Lat:\s*(-?\d+\.?\d*), Lon:\s*(-?\d+\.?\d*)/;
    const match = text.match(regex);
    if (match) {
        const lat = match[1];
        const lon = match[2];
        fetch('/save_grid', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({lat, lon})
        }).then(response => response.json())
          .then(data => alert('Grid saved: X=' + data.grid_X + ', Y=' + data.grid_Y));
    } else {
        alert('Could not find coordinates.');
    }
}
// Start updates
startLocationTracking();
setInterval(updateMeasurementStatus, 5000);
