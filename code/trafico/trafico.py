#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

bgsub = cv.createBackgroundSubtractorMOG2(500, 25, False)

# Contadores
contador_izq_der = 0
contador_der_izq = 0

# para saber la direccion
tracks_activos = []

# actualizar del fondo
update_bg = True

roi_y1 = 210
roi_y2 = 320

for key, frame in autoStream():
    linea_x = frame.shape[1] // 2 
    
    # capturar cuando el puente esté vacio 
    if key == ord('c'):
        update_bg = not update_bg
    
    lr = -1 if update_bg else 0
    fgmask = bgsub.apply(frame, learningRate=lr)
    
    # ROI en la carretera 
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.rectangle(mask_roi, (0, roi_y1), (frame.shape[1], roi_y2), 255, -1)
    fgmask = cv.bitwise_and(fgmask, fgmask, mask=mask_roi)
    
    # enmarcar los coches detectados
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    fgmask = cv.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    

    centros_actuales = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        
        # Filtro: Área mínima, altura mínima y anchura máxima de un coche   
        if cv.contourArea(c) > 80 and h > 12 and w < 200: 
            cx, cy = int(x + w/2), int(y + h/2)
            centros_actuales.append((cx, cy))
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    nuevos_tracks = []
    
    # control de dirección de los coches detectados
    for cx_act, cy_act in centros_actuales:
        mejor_dist = 50  
        mejor_idx = -1
        
        for i, track in enumerate(tracks_activos):
            cx_ant, cy_ant = track['centro']
            dist = np.hypot(cx_act - cx_ant, cy_act - cy_ant)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_idx = i
                
        if mejor_idx != -1:
            track_asignado = tracks_activos.pop(mejor_idx)
            cx_ant, cy_ant = track_asignado['centro']
            contado = track_asignado['contado']
            
            # Comprobamos el cruce de la línea central
            if not contado:
                if cx_ant < linea_x and cx_act >= linea_x:
                    contador_izq_der += 1
                    contado = True 
                elif cx_ant > linea_x and cx_act <= linea_x:
                    contador_der_izq += 1
                    contado = True 
                    
            nuevos_tracks.append({'centro': (cx_act, cy_act), 'contado': contado})
        else:
            nuevos_tracks.append({'centro': (cx_act, cy_act), 'contado': False})
            
    tracks_activos = nuevos_tracks

    # ROI
    cv.line(frame, (0, roi_y1), (frame.shape[1], roi_y1), (255, 0, 0), 1)
    cv.line(frame, (0, roi_y2), (frame.shape[1], roi_y2), (255, 0, 0), 1)
    cv.line(frame, (linea_x, roi_y1), (linea_x, roi_y2), (0, 255, 255), 2)
    
    # Contadores
    putText(frame, f"Derecha: {contador_izq_der}", orig=(20, 30), color=(255, 255, 255), div=3, scale=1, thickness=1)
    putText(frame, f"Izquierda: {contador_der_izq}", orig=(20, 50), color=(255, 255, 255), div=3, scale=1, thickness=1)

    # Detectar coches quietos
    if update_bg:
        putText(frame, "Fondo: ACTUALIZANDO (Pulsa 'c' con puente vacio)", orig=(20, frame.shape[0] - 20), color=(0, 255, 255), div=3, scale=0.8, thickness=1)
    else:
        putText(frame, "Fondo: CONGELADO (Detectando coches quietos)", orig=(20, frame.shape[0] - 20), color=(0, 255, 0), div=3, scale=0.8, thickness=1)

    cv.imshow('Trafico - Contador', frame)
    cv.imshow('Mascara de movimiento', fgmask)

    if key == 27:
        break

cv.destroyAllWindows()