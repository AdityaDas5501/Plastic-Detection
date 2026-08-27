import serial
import requests
import time

# Connect to the ESP32 over USB
esp32 = serial.Serial('COM8', 115200, timeout=1)    #CHANGE THIS COM PORT
time.sleep(2) 

print("Bridge running! Listening to ESP32...")

while True:
    if esp32.in_waiting > 0:
        message = esp32.readline().decode('utf-8').strip()
        
        if message == "SCAN":
            print("Object detected! Asking FastAPI server...")
            
            try:
                # 1. Ask your local FastAPI server to run the model
                response = requests.get("http://127.0.0.1:8000/scan")
                data = response.json()
                
                if "angle" in data:
                    angle = data["angle"]
                    print(f"FastAPI says {angle} degrees. Sending to ESP32...")
                    
                    # 2. Send the angle back to the ESP32 over USB
                    esp32.write(f"{angle}\n".encode('utf-8'))
                else:
                    print("Error: No angle returned from FastAPI.")
                    
            except Exception as e:
                print(f"Failed to connect to FastAPI: {e}")