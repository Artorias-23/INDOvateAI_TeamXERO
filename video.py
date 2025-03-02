import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Define line coordinates (adjust according to your video)
line_start = (140,200)
line_end = (575,240)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Frame size:", frame_width, "x", frame_height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Reduce output FPS to match processing rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps//3, (frame_width, frame_height))
    
    frame_counter = 0
    last_status = False
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Always draw the line FIRST on every frame
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            
        #draw_line(frame, line_start, line_end)
        current_boxes = []
        vehicle_detected = last_status

        # Process only every 3rd frame
        if frame_counter % 2 == 0:
            results = model(frame)
            vehicle_detected = False
            current_boxes = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    current_boxes.append((x1, y1, x2, y2))
                    
                    # Check line intersection
                    rect = (x1, y1, x2 - x1, y2 - y1)
                    intersects, _, _ = cv2.clipLine(rect, line_start, line_end)
                    if intersects:
                        vehicle_detected = True
                        
            # Update tracking variables
            last_status = vehicle_detected
            last_boxes = current_boxes
            
            if vehicle_detected:
                print("Vehicle on pedestrian crossing...")

        # Draw boxes from last detection
        for box in last_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw status
        status_text = "Vehicle on crossing!" if last_status else "No vehicles"
        color = (0, 0, 255) if last_status else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame)
        cv2.imshow('Processing', frame)
        
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def draw_line(image, line_start, line_end, color=(0, 0, 255), thickness=2):
    cv2.line(image, line_start, line_end, color, thickness)

# Usage
process_video('video_cropped.mp4', 'output.mp4')