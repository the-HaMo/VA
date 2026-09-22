#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from collections import deque

import cv2   as cv
import numpy as np

from umucv.stream import autoStream
from umucv.util   import putText

# argumentos 
parser = argparse.ArgumentParser()
parser.add_argument('--ref',   help='archivo de correspondencias imagen↔real (px py rx ry)')
parser.add_argument('--units', default='cm', help='unidades reales (mm, cm, m...)')
parser.add_argument('--pick',  action='store_true',
                    help='modo recogida: clic para registrar puntos de referencia y guardar .txt')
args, _ = parser.parse_known_args()

# MODO --pick 
if args.pick:
    picked = []

    def pick_cb(event, x, y, *_):
        if event == cv.EVENT_LBUTTONDOWN:
            picked.append((x, y))
            print(f"  punto {len(picked):2d}:  px={x:4d}  py={y:4d}")
        elif event == cv.EVENT_RBUTTONDOWN and picked:
            picked.pop()

    cv.namedWindow('pick')
    cv.setMouseCallback('pick', pick_cb)
    print("Haz click en los puntos de referencia en orden. Click der: borrar último. ESC: terminar.\n")

    for key, frame in autoStream():
        vis = frame.copy()
        for i, (px, py) in enumerate(picked):
            cv.circle(vis, (px, py), 5, (0, 0, 255), -1)
            putText(vis, str(i + 1), orig=(px + 6, py - 4), color=(0, 0, 255),
                    div=3, scale=0.8, thickness=1)
        putText(vis, f"{len(picked)} puntos | Click der: borrar | ESC: terminar",
                orig=(8, 18), color=(255, 255, 255), div=3, scale=0.8, thickness=1)
        cv.imshow('pick', vis)
        if key == 27:
            break

    if len(picked) < 4:
        print("Se necesitan al menos 4 puntos.")
        sys.exit(1)

    print(f"\n{len(picked)} puntos. Introduce las coordenadas reales (rx ry) de cada uno:")
    real_coords = []
    for i, (px, py) in enumerate(picked):
        while True:
            try:
                rx, ry = map(float, input(f"  punto {i+1} (px={px}, py={py}) → rx ry: ").split())
                real_coords.append((rx, ry))
                break
            except ValueError:
                print("    Escribe dos números, ej: 0.0 85.6")

    dev = next((a for a in sys.argv if '--dev=' in a), None)
    stem = Path(dev.split('=')[1]).stem if dev else 'imagen'
    out = Path(f"ref-{stem}.txt")
    with open(out, 'w') as f:
        f.write(f"# Referencia para {stem}\n# px  py   rx  ry\n")
        for (px, py), (rx, ry) in zip(picked, real_coords):
            f.write(f"{px:6d}  {py:6d}    {rx}  {ry}\n")
    print(f"\nGuardado: {out}")
    sys.exit(0)

# cargar referencias
if not args.ref:
    print("Especifica --ref archivo.txt  (o usa --pick para crear el archivo).")
    sys.exit(1)

data = np.loadtxt(args.ref, comments='#')
if data.ndim == 1:
    data = data.reshape(1, -1)

img_pts  = data[:, 0:2].astype(np.float32)   # píxeles en la imagen
real_pts = data[:, 2:4].astype(np.float32)   # coordenadas reales
units    = args.units

# homografía imagen → plano real 
H, mask = cv.findHomography(img_pts, real_pts, cv.RANSAC, 2.0)
inliers  = mask.ravel().astype(bool) if mask is not None else np.ones(len(img_pts), bool)
print(f"Homografía:  {inliers.sum()}/{len(img_pts)} inliers")

# imagen rectificada
SCALE  = 10.0   # 10 px por unidad real 
H_disp = None
W_r = H_r = 0
min_r  = None

clicks    = deque(maxlen=2)   
clicks_r  = deque(maxlen=2) 
show_rect = False
snap_n    = 0

def on_mouse(event, x, y, *_):
    if event == cv.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        clicks.clear()

def on_mouse_rect(event, x, y, *_):
    if event == cv.EVENT_LBUTTONDOWN:
        clicks_r.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        clicks_r.clear()

cv.namedWindow('rectificacion')
cv.setMouseCallback('rectificacion', on_mouse)

for key, frame in autoStream():
    # calcular H_disp en el primer frame
    if H_disp is None:
        h_img, w_img = frame.shape[:2]
        corners = np.float32([[0,0],[w_img,0],[w_img,h_img],[0,h_img]]).reshape(-1,1,2)
        cr = cv.perspectiveTransform(corners, H).reshape(-1,2)
        min_r = cr.min(axis=0)
        max_r = cr.max(axis=0)
        W_r   = min(int((max_r[0]-min_r[0])*SCALE)+1, 2000)
        H_r   = min(int((max_r[1]-min_r[1])*SCALE)+1, 2000)
        off   = np.array([[1,0,-min_r[0]],[0,1,-min_r[1]],[0,0,1]], dtype=np.float64)
        H_disp = np.diag([SCALE, SCALE, 1.0]) @ off @ H

    vis = frame.copy()

    # contorno del objeto de referencia
    pts_poly = img_pts.astype(np.int32).reshape((-1,1,2))
    cv.polylines(vis, [pts_poly], isClosed=True, color=(0, 255, 255), thickness=2)

    # puntos de referencia
    for ok, (px, py), (rx, ry) in zip(inliers, img_pts.astype(int), real_pts):
        col = (0, 200, 0) if ok else (0, 0, 200)
        cv.circle(vis, (px, py), 4, col, -1)
        putText(vis, f"({rx:.0f},{ry:.0f})", orig=(px + 5, py - 5),
                color=col, div=3, scale=0.65, thickness=1)

    # puntos de medición en imagen original
    for p in clicks:
        cv.circle(vis, p, 6, (0, 0, 255), -1)

    if len(clicks) == 2:
        r1 = cv.perspectiveTransform(np.array([[clicks[0]]], dtype=np.float32), H)[0][0]
        r2 = cv.perspectiveTransform(np.array([[clicks[1]]], dtype=np.float32), H)[0][0]
        dist = float(np.linalg.norm(r1 - r2))
        mid  = tuple(((np.array(clicks[0], float) + np.array(clicks[1], float)) / 2).astype(int))
        cv.line(vis, clicks[0], clicks[1], (0, 0, 255), 2, cv.LINE_AA)
        putText(vis, f"{dist:.1f} {units}", orig=(mid[0] + 8, mid[1] - 8),
                color=(255, 255, 255), div=2, scale=1.2, thickness=1)

    putText(vis, "Click: medir | Der: borrar | r: rectif | s: guardar | ESC: salir",
            orig=(8, 18), color=(255, 255, 255), div=3, scale=0.8, thickness=1)
    cv.imshow('rectificacion', vis)

    # imagen rectificada
    if show_rect:
        rectif = cv.warpPerspective(frame, H_disp, (W_r, H_r))

        # puntos de referencia en rectificada
        for ok, (rx, ry) in zip(inliers, real_pts):
            px_r = int((rx - min_r[0]) * SCALE)
            py_r = int((ry - min_r[1]) * SCALE)
            cv.circle(rectif, (px_r, py_r), 4, (0, 200, 0) if ok else (0, 0, 200), -1)

        # medición en imagen rectificada
        for p in clicks_r:
            cv.circle(rectif, p, 6, (0, 0, 255), -1)

        if len(clicks_r) == 2:
            dist_r = float(np.linalg.norm(np.array(clicks_r[0]) - np.array(clicks_r[1]))) / SCALE
            mid_r  = tuple(((np.array(clicks_r[0], float) + np.array(clicks_r[1], float)) / 2).astype(int))
            cv.line(rectif, clicks_r[0], clicks_r[1], (0, 0, 255), 2, cv.LINE_AA)
            putText(rectif, f"{dist_r:.1f} {units}", orig=(mid_r[0] + 8, mid_r[1] - 8),
                    color=(255, 255, 255), div=2, scale=1.2, thickness=1)

        cv.imshow('rectificada', rectif)
        cv.setMouseCallback('rectificada', on_mouse_rect)
    else:
        cv.destroyWindow('rectificada')

    if key == ord('r'):
        show_rect = not show_rect
        if not show_rect:
            clicks_r.clear()
    elif key == ord('s'):
        snap_n += 1
        fname = f"snap_{snap_n:02d}.png"
        cv.imwrite(fname, vis)
        print(f"Guardado: {fname}")
    elif key in (27, ord('q')):
        break

cv.destroyAllWindows()
