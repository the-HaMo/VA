#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
from datetime import datetime

from umucv.util import ROI, putText, Slider
from umucv.stream import autoStream

from bot.botVideo import send_video_telegram, remove_video


from ultralytics import YOLO
model = YOLO("yolo11n.pt") 

# detector de caras pre-entrenado de OpenCV (Haar Cascade)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Detección de movimiento con MOG2
bgsub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

conf_slider = Slider("Conf YOLO", "detector", 0.5, 0, 1, 0.01)
region = ROI("detector")
writer = None   
frames_without_motion = 0
motion_frames_to_stop = 20  
is_recording = False
current_video_file = None

recording_frames = 0  
sec_video_seconds = 5
recording_size = None  
motion_run = 0  
motion_start_frames_required = 3  

person_detected_in_video = False 

def finalize_and_send(video_path, recorded_frames, has_person, fps=30.0):
    min_frames = int(sec_video_seconds * fps)
    
    if video_path is None or recorded_frames < min_frames:
        remove_video(video_path)
        return

    if (not os.path.exists(video_path)) or os.path.getsize(video_path) <= 0:
        remove_video(video_path)
        return

    if has_person:
        print(f"¡Persona confirmada! Enviando {video_path} a Telegram...")
        send_video_telegram(video_path)
    else:
        remove_video(video_path)

for key, frame in autoStream():
    
    if region.roi:
        x1, x2 = sorted([region.roi[0], region.roi[2]])
        y1, y2 = sorted([region.roi[1], region.roi[3]])
        
        if key == ord('x'):
            region.roi = []
            recording_size = None
            if writer is not None:
                writer.release()
                writer = None
                is_recording = False
                finalize_and_send(current_video_file, recording_frames, person_detected_in_video)
        
        # Extraemos el trozo marcado
        trozo = frame[y1:y2+1, x1:x2+1]
        
        # MOG2 analiza si hay movimiento
        fgmask = bgsub.apply(trozo)
        motion = cv.countNonZero(fgmask) 
        
        person_seen_now = False

        if motion > 500:
            motion_run += 1
            frames_without_motion = 0
            
            # Iniciamos grabación si el movimiento es constante
            if not is_recording and motion_run >= motion_start_frames_required:
                h, w = trozo.shape[:2]
                recording_size = (w, h)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for codec, ext in [('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]:
                    candidate = f"video_{timestamp}{ext}"
                    fourcc = cv.VideoWriter_fourcc(*codec)
                    test_writer = cv.VideoWriter(candidate, fourcc, 30.0, recording_size)
                    if test_writer is not None and test_writer.isOpened():
                        writer = test_writer
                        current_video_file = candidate
                        break
                    if test_writer is not None:
                        test_writer.release()

                if writer is not None:
                    is_recording = True
                    recording_frames = 0
                    person_detected_in_video = False 
                else:
                    is_recording = False
                    recording_size = None

            # yolo + anonimizacion si estamos grabando
            if is_recording:
                rgb_trozo = cv.cvtColor(trozo, cv.COLOR_BGR2RGB)
                # la clase 0 (personas)
                [result] = model(rgb_trozo, classes=[0], verbose=False)
                
                for b in result.boxes:
                    conf = b.conf.cpu().numpy()[0]
                    
                    if conf > conf_slider.value: # Umbral de confianza de YOLO
                        person_seen_now = True
                        person_detected_in_video = True 
                        
                        [[bx1, by1, bx2, by2]] = np.array(b.xyxy.cpu()).astype(int)
                        
                        #anonimazcion de la cara
                        area_persona = trozo[by1:by2, bx1:bx2]
                        if area_persona.size > 0:
                            gray_persona = cv.cvtColor(area_persona, cv.COLOR_BGR2GRAY)
                            caras = face_cascade.detectMultiScale(gray_persona, scaleFactor=1.1, minNeighbors=4)
                            for (fx, fy, fw, fh) in caras:
                                cara_trozo = area_persona[fy:fy+fh, fx:fx+fw]
                                area_persona[fy:fy+fh, fx:fx+fw] = cv.blur(cara_trozo, (40, 40))
                            
                        # Dibujamos el contorno verde de la persona entera
                        cv.rectangle(trozo, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        putText(trozo, f"Persona {conf:.2f}", (bx1, max(0, by1 - 5)), scale=0.7)

        else:
            motion_run = 0
            frames_without_motion += 1
            if frames_without_motion > motion_frames_to_stop and is_recording:
                writer.release()
                writer = None
                is_recording = False
                recording_size = None
                finalize_and_send(current_video_file, recording_frames, person_detected_in_video)
        
        
        if is_recording and writer is not None:
            recording_frames += 1
            if recording_size is not None and (trozo.shape[1], trozo.shape[0]) != recording_size:
                trozo_to_write = cv.resize(trozo, recording_size, interpolation=cv.INTER_LINEAR)
            else:
                trozo_to_write = trozo
            writer.write(trozo_to_write)

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'MOG2 Mov: {motion}', orig=(x1,y1-8))
        
        if is_recording:
            putText(frame, f'REC', orig=(x1,y1+20), color=(0,0,255))

    cv.imshow('detector',frame)

    if key == 27:
        break

if writer is not None:
    writer.release()
    finalize_and_send(current_video_file, recording_frames, person_detected_in_video)

cv.destroyAllWindows()