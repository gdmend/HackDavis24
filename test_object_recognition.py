import cv2
import serial
import numpy as np

# Load the pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# List of aquatic related classes in a general model
aquatic_classes = ['boat', 'bottle']  # Adjust this list based on your actual model's capability

# Setup serial connection
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=None)

def detect_and_display(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            class_name = classes[idx]
            if class_name in aquatic_classes:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{class_name}: {confidence:.2f}%"
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detected Objects', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

def receive_and_show_image():
    read_buffer = bytearray()
    collecting = False

    print("Connected to Arduino. Press 'q' to quit.")

    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                read_buffer.extend(data)
                if b'\xff\xd8' in read_buffer and b'\xff\xd9' in read_buffer:
                    start = read_buffer.find(b'\xff\xd8')
                    end = read_buffer.find(b'\xff\xd9', start) + 2
                    jpg_data = read_buffer[start:end]
                    read_buffer = read_buffer[end:]  # Clear buffer after processing

                    if jpg_data:
                        img = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            detect_and_display(img)
                        else:
                            print("Failed to decode image")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        ser.close()

receive_and_show_image()
