from ultralytics import YOLO
import cv2

# Load a YOLOV10n model
model = YOLO('yolov10n.pt')

# Perform inference with camera
cam = cv2.VideoCapture(0)

# Only show inference when it pass a certain confidence threshold
while True:
  ret, frame = cam.read()
  if not ret:
      break

  results = model(frame)

  # Filter results by confidence threshold
  conf_threshold = 0.75
  filtered_results = []
  for result in results:
      boxes = result.boxes
      filtered_boxes = [box for box in boxes if box.conf[0] >= conf_threshold]
      result.boxes = filtered_boxes
      filtered_results.append(result)

  # Display the results
  for result in filtered_results:
      annotated_frame = result.plot()
      cv2.imshow('YOLOv10 Inference', annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break