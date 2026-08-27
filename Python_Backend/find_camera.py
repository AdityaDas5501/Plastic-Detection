import cv2

print("Scanning USB ports for connected cameras...")
for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"-> Camera found at index: {index}")
        cap.release()
    else:
        print(f"No camera at index: {index}")