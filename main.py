# Codes by Vision

import cv2
import numpy as np
import threading
import pygame
import time
import os

class FireDetector:
    def __init__(self):
        pygame.mixer.init()
        
        self.alarm_sound_path = os.path.join("sounds", "alarm.mp3")
        if not os.path.exists(self.alarm_sound_path):
            self.alarm_sound_path = os.path.join("sounds", "alert.mp3")
        
        if not os.path.exists(self.alarm_sound_path):
            print(f"Warning: {self.alarm_sound_path} not found!")
            
        try:
            self.alarm_sound = pygame.mixer.Sound(self.alarm_sound_path)
        except pygame.error:
            print("Error loading alarm sound. Make sure pygame can play MP3 files.")
            self.alarm_sound = None
            
        self.alarm_playing = False
        self.alarm_thread = None
        self.stop_alarm_flag = threading.Event()
        
        self.fire_threshold = 500
        self.fire_intensity_threshold = 100
        
    def play_alarm(self):
        if not self.alarm_sound:
            return
            
        self.stop_alarm_flag.clear()
        while not self.stop_alarm_flag.is_set():
            self.alarm_sound.play()
            while pygame.mixer.get_busy() and not self.stop_alarm_flag.is_set():
                time.sleep(0.1)
                
    def start_alarm(self):
        if not self.alarm_playing and self.alarm_sound:
            self.alarm_playing = True
            self.stop_alarm_flag.clear()
            self.alarm_thread = threading.Thread(target=self.play_alarm)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
            
    def stop_alarm(self):
        if self.alarm_playing:
            self.alarm_playing = False
            self.stop_alarm_flag.set()
            if self.alarm_thread:
                self.alarm_thread.join(timeout=1)
                
    def detect_fire(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        mask_fire = cv2.bitwise_or(mask_red1, mask_red2)
        mask_fire = cv2.bitwise_or(mask_fire, mask_orange)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        total_fire_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                
                roi = frame[y:y+h, x:x+w]
                avg_intensity = np.mean(roi)
                
                if area > self.fire_threshold and avg_intensity > self.fire_intensity_threshold:
                    fire_regions.append({
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'intensity': avg_intensity
                    })
                    total_fire_area += area
                    
        return fire_regions, mask_fire, total_fire_area
    
    def process_frame(self, frame):
        fire_regions, mask_fire, total_fire_area = self.detect_fire(frame)
        
        fire_detected = len(fire_regions) > 0 and total_fire_area > self.fire_threshold
        
        for region in fire_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire: {int(region['area'])} px", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if fire_detected:
            self.start_alarm()
            cv2.putText(frame, "FIRE DETECTED!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.stop_alarm()
            cv2.putText(frame, "NO FIRE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.putText(frame, f"Fire Area: {int(total_fire_area)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame, fire_detected

def main():
    detector = FireDetector()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Fire Detection System Started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
            
        processed_frame, fire_detected = detector.process_frame(frame)
        
        cv2.imshow('Fire Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    detector.stop_alarm()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Fire Detection System Stopped")

if __name__ == "__main__":
    main()
