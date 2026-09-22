#!/usr/bin/env python3
"""
Ejercicio LK — Dirección de movimiento de cámara y velocidad angular.

Extiende lk_track.py con:
  a) Dirección dominante de movimiento: UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD
  b) Velocidad angular de rotación en grados/segundo

La transformación afín estimada por RANSAC sobre las correspondencias LK
descompone el flujo en:
  - traslación (tx, ty)  → movimiento lateral/vertical de cámara
  - escala (s)           → movimiento hacia delante/atrás (s>1 → FORWARD)
  - ángulo (θ)           → rotación en el plano (roll)

La velocidad angular total combina roll, yaw y pitch aproximados.
"""

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
from collections import deque
import time

# ── parámetros ────────────────────────────────────────────────────────────
track_len       = 20
detect_interval = 5

corners_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10, blockSize=7)
lk_params      = dict(winSize=(15,15), maxLevel=2,
                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

TRANS_THRESH = 1.5   # píxeles mínimos de traslación para no considerar STATIC
ZOOM_THRESH  = 0.003 # cambio mínimo de escala para considerar FORWARD/BACKWARD

# ── estado ────────────────────────────────────────────────────────────────
tracks    = []
prev_time = None
direction = 'STATIC'
omega     = 0.0      # deg/s


def estimate_motion(pts0, pts1, W, H, dt):
    """Devuelve (direction, omega_deg_s) a partir de correspondencias."""
    M, _ = cv.estimateAffinePartial2D(pts0, pts1,
                                      method=cv.RANSAC,
                                      ransacReprojThreshold=2)
    if M is None:
        return 'STATIC', 0.0

    tx, ty = M[0, 2], M[1, 2]
    scale   = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    angle_r = np.arctan2(M[1, 0], M[0, 0])

    # velocidad angular: roll directo del ángulo afín;
    # yaw y pitch aproximados usando focal length estimada (f ≈ max(W,H))
    f           = float(max(W, H))
    omega_roll  = np.degrees(angle_r)          / dt
    omega_yaw   = np.degrees(np.arctan2(tx, f)) / dt
    omega_pitch = np.degrees(np.arctan2(ty, f)) / dt
    omega_total = np.sqrt(omega_roll**2 + omega_yaw**2 + omega_pitch**2)

    # dirección dominante: convertimos escala a "píxeles equivalentes"
    # para poder comparar con tx, ty en las mismas unidades
    zoom_px = abs(scale - 1.0) * W
    candidates = sorted([
        (zoom_px, 'FORWARD'  if scale > 1 else 'BACKWARD'),
        (abs(tx), 'LEFT'     if tx > 0   else 'RIGHT'),
        (abs(ty), 'DOWN'     if ty > 0   else 'UP'),
    ], key=lambda x: x[0], reverse=True)

    mag, label = candidates[0]
    # umbral adaptado: escala o traslación deben superar el mínimo
    if (label in ('FORWARD', 'BACKWARD') and abs(scale - 1.0) > ZOOM_THRESH) or \
       (label not in ('FORWARD', 'BACKWARD') and mag > TRANS_THRESH):
        return label, omega_total
    return 'STATIC', omega_total


# ── bucle principal ────────────────────────────────────────────────────────
for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    H_fr, W_fr = frame.shape[:2]
    now = time.time()

    if tracks and prev_time is not None:
        dt = now - prev_time

        p0 = np.float32([t[-1] for t in tracks])
        p1,  _, _ = cv.calcOpticalFlowPyrLK(prevgray, gray,     p0, None, **lk_params)
        p0r, _, _ = cv.calcOpticalFlowPyrLK(gray,     prevgray, p1, None, **lk_params)
        good = abs(p0 - p0r).reshape(-1, 2).max(axis=1) < 1

        new_tracks = []
        for t, pt, ok in zip(tracks, p1.reshape(-1, 2), good):
            if not ok: continue
            t.append(pt)
            new_tracks.append(t)
        tracks = new_tracks

        # pares de puntos consecutivos para estimar la transformación
        pts0 = np.float32([t[-2] for t in tracks if len(t) >= 2])
        pts1 = np.float32([t[-1] for t in tracks if len(t) >= 2])

        if len(pts0) >= 8 and dt > 0:
            direction, omega = estimate_motion(pts0, pts1, W_fr, H_fr, dt)

        # dibujar trayectorias
        cv.polylines(frame, [np.int32(t) for t in tracks], False, (0, 0, 255))
        for t in tracks:
            cv.circle(frame, np.int32(t[-1]), 2, (0, 0, 255), -1)

        # flecha de flujo medio amplificada, centrada en la imagen
        if len(pts0) >= 4:
            mean_flow = (pts1 - pts0).mean(axis=0)
            cx, cy = W_fr // 2, H_fr // 2
            AMPLIFY = 20
            dx, dy = (mean_flow * AMPLIFY).astype(int)
            if dx != 0 or dy != 0:
                cv.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                               (0, 255, 0), 3, tipLength=0.3)

    if n % detect_interval == 0:
        mask = np.full_like(gray, 255)
        for x, y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [pt] in np.float32(corners):
                tracks.append(deque([pt], maxlen=track_len))

    putText(frame, f'{len(tracks)} pts | {direction} | {omega:.1f} deg/s')
    cv.imshow('lk-motion', frame)
    prevgray  = gray
    prev_time = now
