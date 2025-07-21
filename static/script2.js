let bleDevice;
let bleServer;
let sensorDataCharacteristic;
let commandCharacteristic;

const SERVICE_UUID = '91bad492-b950-4226-aa2b-4ede9fa42f59';
const CHAR_UUID_SENSOR_DATA = '98e025d3-23e5-4b62-9916-cc6c330c84ac';
const CHAR_UUID_COMMAND = 'f78ebbff-c8b7-4107-93de-889a6a06d408';

document.getElementById('connect').addEventListener('click', async () => {
  try {
    bleDevice = await navigator.bluetooth.requestDevice({
      filters: [{ name: "ESP32S3 Feather" }],
      optionalServices: [SERVICE_UUID]
    });

    bleServer = await bleDevice.gatt.connect();
    const service = await bleServer.getPrimaryService(SERVICE_UUID);

    sensorDataCharacteristic = await service.getCharacteristic(CHAR_UUID_SENSOR_DATA);
    commandCharacteristic = await service.getCharacteristic(CHAR_UUID_COMMAND);

    await sensorDataCharacteristic.startNotifications();
    sensorDataCharacteristic.addEventListener('characteristicvaluechanged', handleSensorData);

    console.log('Connected to ESP32 and started notifications');
  } catch (error) {
    console.error('BLE connection error:', error);
  }
});

document.getElementById('submitCommandBtn').addEventListener('click', () => {
  const command = document.getElementById('commandInput').value;
  sendCommand(command);
});

document.getElementById('getReadingBtn').addEventListener('click', generateMockSensorData);

function sendCommand(command) {
  if (commandCharacteristic && command) {
    const commandData = new TextEncoder().encode(command.trim());
    commandCharacteristic.writeValue(commandData).catch(console.error);
  } else {
    console.warn("Command not sent: BLE not connected or input empty");
  }
}

function handleSensorData(event) {
  const value = new TextDecoder().decode(event.target.value);
  const [vmc, temp, ec] = value.split(',');

  if (vmc && temp && ec) {
    document.getElementById('value1').textContent = parseFloat(vmc).toFixed(1);
    document.getElementById('value2').textContent = parseFloat(temp).toFixed(1);
    document.getElementById('value3').textContent = (parseFloat(ec) / 1000).toFixed(3);

    const entry = `<div>VMC: ${vmc}, Temp: ${temp} °C, EC: ${ec} µS/cm</div>`;
    document.getElementById('sensorDataHistory').innerHTML += entry + "<hr/>";
  } else {
    console.warn('Received malformed BLE data:', value);
  }
}

async function generateMockSensorData() {
  const vmc = document.getElementById('VMCInput').value.trim() || "1";

  if (commandCharacteristic) {
    const encoder = new TextEncoder();
    await commandCharacteristic.writeValue(encoder.encode(vmc));
    console.log("VMC sent to ESP32:", vmc);
  } else {
    // fallback mock (no BLE connection)
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
    console.log("Displayed fallback mock data:", mockData);
  }
}
