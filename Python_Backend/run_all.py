import subprocess
import sys
import time
import webbrowser

print("Starting Plastic Sorter System...")
print("---------------------------------")

try:
    # 1. Start the Logs server
    print("Starting Logs Server on port 8001...")
    logs_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "logs_server:app", "--host", "0.0.0.0", "--port", "8001"]
    )
    
    # Wait a moment for it to initialize
    time.sleep(1)

    # 2. Start the main FastAPI server
    print("Starting Main App Server on port 8000...")
    fastapi_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    )
    
    # Wait for the server to fully boot up
    time.sleep(2) 
    
    # 3. Open the browser automatically
    print("Opening dashboard in browser...")
    webbrowser.open("http://127.0.0.1:8000/")

    # 4. Start the USB Bridge
    print("Starting USB Bridge...")
    bridge_process = subprocess.Popen(
        [sys.executable, "usb_bridge.py"]
    )

    print("\nAll systems go! Press Ctrl+C to shut everything down.\n")

    # Keep the script running
    logs_process.wait()
    fastapi_process.wait()
    bridge_process.wait()

except KeyboardInterrupt:
    print("\nShutting down servers...")
    logs_process.terminate()
    fastapi_process.terminate()
    bridge_process.terminate()
    print("Shutdown complete.")