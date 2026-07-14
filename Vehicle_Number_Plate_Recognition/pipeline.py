import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

class VNPRPipeline:
    def __init__(self, vehicle_model="yolov8n.pt", plate_model="plate_model.pt"):
        # Load YOLO models
        self.vehicle_model = YOLO(vehicle_model)
        self.plate_model = YOLO(plate_model)
        
        # Initialize EasyOCR (Set gpu=True if you have an NVIDIA graphics card to make it 10x faster)
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        # COCO Classes: Car(2), Motorcycle(3), Bus(5), Truck(7)
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Memory cache to hold proportional coordinates and text for steady UI tracking
        self.track_cache = {} 

    def process_frame(self, frame, frame_count):
        # Stage 1: Vehicle Detection with Persistent Tracking
        # imgsz=480 speeds up processing by resizing the frame during inference
        results = self.vehicle_model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            imgsz=480, 
            verbose=False
        )[0]
        
        if results.boxes is None:
            return frame

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_classes:
                continue
                
            # Get current vehicle bounding box dimensions
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vw, vh = x2 - x1, y2 - y1 # Vehicle width and height
            
            # Extract Vehicle ID assigned by the tracker
            track_id = int(box.id[0]) if box.id is not None else None
            label = f"ID: {track_id}" if track_id is not None else "Vehicle"
            
            # Draw the main vehicle bounding box (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
            # Crop the vehicle for plate detection
            vehicle_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if vehicle_crop.size == 0:
                continue
                
            # Stage 2: Plate Detection
            # imgsz=320 speeds up processing for the cropped vehicle image
            plate_results = self.plate_model(vehicle_crop, imgsz=320, verbose=False)[0]
            
            plate_drawn = False

            if len(plate_results.boxes) > 0:
                # A license plate was successfully found in this exact frame
                p_box = plate_results.boxes[0]
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                
                # Proportional Math: Calculate plate position as a percentage of vehicle size
                rel_x, rel_y = px1 / vw, py1 / vh
                rel_w, rel_h = (px2 - px1) / vw, (py2 - py1) / vh
                
                # Convert back to global frame coordinates
                gx1, gy1 = x1 + px1, y1 + py1
                gx2, gy2 = x1 + px2, y1 + py2
                
                # Stage 3: OCR Logic
                # Run OCR only every 5th frame to maintain video processing speed
                if frame_count % 5 == 0:
                    plate_crop = frame[gy1:gy2, gx1:gx2]
                    text = "?"
                    if plate_crop.size > 0:
                        # Convert to grayscale for better OCR accuracy
                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                        ocr_results = self.reader.readtext(gray_plate, detail=0)
                        
                        if ocr_results:
                            # Clean up the text: remove all spaces and force uppercase
                            text = "".join(ocr_results).replace(" ", "").upper()
                    
                    # Update Memory Cache (Ignore garbage readings shorter than 3 characters)
                    if track_id is not None and len(text) > 2: 
                        self.track_cache[track_id] = {
                            'prop': (rel_x, rel_y, rel_w, rel_h),
                            'text': text
                        }
                
                plate_drawn = True

            # If plate is NOT found in this frame, but we remember seeing it earlier on this ID
            elif track_id is not None and track_id in self.track_cache:
                cached_data = self.track_cache[track_id]
                rel_x, rel_y, rel_w, rel_h = cached_data['prop']
                
                # Reconstruct global coordinates based on the CURRENT size of the vehicle
                # This ensures the box scales smoothly as the car drives closer or further away
                gx1 = x1 + int(rel_x * vw)
                gy1 = y1 + int(rel_y * vh)
                gx2 = gx1 + int(rel_w * vw)
                gy2 = gy1 + int(rel_h * vh)
                
                plate_drawn = True
            
            # --- STAGE 4: DRAWING THE UI PERFECTLY ---
            if plate_drawn and track_id in self.track_cache:
                final_text = self.track_cache.get(track_id, {}).get('text', "?")
                
                if final_text != "?":
                    # Draw Plate Bounding Box (Yellow)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                    
                    # Draw a solid black background for the text to ensure perfect readability
                    (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (gx1, gy1 - text_h - 10), (gx1 + text_w, gy1), (0, 0, 0), cv2.FILLED)
                    
                    # Overlay the final plate text (Yellow)
                    cv2.putText(frame, final_text, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Plate has never been found for this vehicle yet
                cv2.putText(frame, "Scanning...", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame