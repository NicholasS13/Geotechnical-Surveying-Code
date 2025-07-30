#include <SDI12.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define DATA_PIN 10
#define SENSOR_ADDR '3'
#define BAUD_RATE 115200


// UUIDs must match frontend
#define SERVICE_UUID           "91bad492-b950-4226-aa2b-4ede9fa42f59"
#define CHARACTERISTIC_UUID_SENSOR_DATA "98e025d3-23e5-4b62-9916-cc6c330c84ac"
#define CHARACTERISTIC_UUID_COMMAND     "f78ebbff-c8b7-4107-93de-889a6a06d408"


SDI12 sdi12(DATA_PIN);

BLECharacteristic *sensorDataCharacteristic;
BLECharacteristic *commandCharacteristic;

float convertVWC(float raw) {
  return (3.879e-4 * raw) - 0.6956;
}
float convertEC(float rawEC, float temperature) {
  return (rawEC / (1 + 0.019 * (temperature - 25))) / 1000;
}
void getReading() {
  sdi12.clearBuffer();
  // Send M! to request measurement
  sdi12.sendCommand(String(SENSOR_ADDR) + "M!");
  delay(100);
  String mResponse = sdi12.readStringUntil('\n');
  mResponse.trim();
  Serial.println("M! response: " + mResponse);
  if (mResponse.length() < 5) {
    Serial.println("Invalid M! response.");
    delay(10000);
    return;
  }
  int waitTime = mResponse.substring(1, 4).toInt(); // 'ttt'
  int numResults = mResponse.substring(4, 5).toInt(); // 'n'
  Serial.print("Wait time: "); Serial.print(waitTime / 10.0); Serial.println(" sec");
  Serial.print("Expected results: "); Serial.println(numResults);
  delay(waitTime * 100);  // Convert tenths of seconds to milliseconds
  // Now send D0! to get actual data
  sdi12.clearBuffer();
  sdi12.sendCommand(String(SENSOR_ADDR) + "D0!");
  delay(300);
  if (sdi12.available()) {
    String response = sdi12.readStringUntil('\n');
    response.trim();
    Serial.println("Raw: " + response);
    // Make sure response has valid + signs
    if (!response.startsWith(String(SENSOR_ADDR))) {
      Serial.println("Invalid start — does not begin with sensor address.");
    } else if (response.indexOf('+') == -1) {
      Serial.println("No valid '+' found — cannot parse.");
    } else {
      String data = response.substring(1);  // Remove address prefix (e.g., '3')
      int plus1 = data.indexOf('+');
      int plus2 = data.indexOf('+', plus1 + 1);
      int plus3 = data.indexOf('+', plus2 + 1);
      if (plus1 == -1 || plus2 == -1 || plus3 == -1) {
        Serial.println("Failed to parse values.");
      } else {
        float rawVWC = data.substring(plus1 + 1, plus2).toFloat();
        float temp = data.substring(plus2 + 1, plus3).toFloat();
        float rawEC = data.substring(plus3 + 1).toFloat();
        float vwc = convertVWC(rawVWC);
        float ec = convertEC(rawEC, temp);
        Serial.print("VWC: "); Serial.print(vwc, 4); Serial.print(" m³/m³ | ");
        Serial.print("Temp: "); Serial.print(temp, 2); Serial.print(" °C | ");
        Serial.print("EC: "); Serial.print(ec, 4); Serial.println(" dS/m");
        
        char payload[50];
        snprintf(payload, sizeof(payload), "%.4f,%.4f,%.4f", vwc, temp, ec);
        Serial.println("Sending: " + String(payload));
        sensorDataCharacteristic->setValue(payload);
        sensorDataCharacteristic->notify();
      }
    }
  } else {
    Serial.println("No response from D0! command.");
  }
  sdi12.clearBuffer();
}

// Callback for command input (VMC)
class CommandCallback : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    String value = characteristic->getValue();
    if (!value.isEmpty()) {
      getReading();
    }
  }
};

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial);
  sdi12.begin();
  delay(2000);
  Serial.println("Starting Teros 12 SDI-12 sensor read...");
  BLEDevice::init("ESP32S3 Feather");
  BLEServer *server = BLEDevice::createServer();
  BLEService *service = server->createService(SERVICE_UUID);

  // Sensor Data Characteristic (Notify)
  sensorDataCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_UUID_SENSOR_DATA,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  sensorDataCharacteristic->addDescriptor(new BLE2902());

  // Command Characteristic (Write from frontend)
  commandCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_UUID_COMMAND,
    BLECharacteristic::PROPERTY_WRITE
  );
  commandCharacteristic->setCallbacks(new CommandCallback());

  service->start();
  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->start();
}
void loop(){}
