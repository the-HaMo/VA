#! /usr/bin/env python

import math
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np

from umucv.stream import autoStream
from umucv.util import Slider, putText

points = deque(maxlen=2)

def on_mouse(event, x, y, _flags, _param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        points.clear()

cv.namedWindow("medidor")
cv.setMouseCallback("medidor", on_mouse)

slider_fov = Slider("fov", "medidor", 56, 10, 90, 1)
slider_Z   = Slider("Z",   "medidor", 3.0, 1.0, 30.0, 0.1)
slider_A   = Slider("A",   "medidor", 0.8, 0.0, 2.0,  0.1)
slider_X   = Slider("X",   "medidor", 0.0, -1.0, 1.0, 0.01)

# pinhole
def proj_points(X, Y, Z, f, cx, cy):
    u = f * (X / Z) + cx
    v = f * (Y / Z) + cy
    return np.column_stack((u, v))

# Dibuja la rejilla 
def draw_curve(frame, pts2d, color, thickness=1):
    if len(pts2d) < 2:
        return
    ok = np.isfinite(pts2d).all(axis=1)
    if ok.sum() < 2:
        return
    pts = np.round(pts2d[ok]).astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(frame, [pts], False, color, thickness, cv.LINE_AA)

# color de la rejilla
def style_for_grid(value, axis_color, axis_thickness, tol=1e-6):
    if abs(value) < tol:
        return axis_color, axis_thickness
    if abs(value - round(value)) < tol:
        return (230, 230, 230), 1
    return (150, 150, 150), 1

# cuantas cuadrículas dibujar en el plano frontal
def build_front_grid_ranges(w, cx, cy, Z, f, A, grid_step=0.2):
    visible_x_half = max(abs((0 - cx) * Z / f), abs((w - cx) * Z / f))
    x_span = max(2.0, visible_x_half + 1.0)

    y_visible_top = A + (cy * Z / f)
    y_min = 0.0
    y_max = max(3.2, y_visible_top + 0.2)

    xs = np.arange(-x_span, x_span + 1e-9, grid_step)
    ys = np.arange(y_min, y_max + 1e-9, grid_step)
    return xs, ys, x_span, y_max

# dibuja la rejilla frontal
def draw_front_grid(frame, xs, ys, Xoff, A, Z, f, cx, cy):
    for y in ys:
        X = xs + Xoff
        Y = np.full_like(xs, A - y)
        Zs = np.full_like(xs, Z)
        pts = proj_points(X, Y, Zs, f, cx, cy)
        color, thick = style_for_grid(y, (255, 255, 255), 3)
        draw_curve(frame, pts, color, thick)

    for x in xs:
        X = np.full_like(ys, x + Xoff)
        Y = A - ys
        Zs = np.full_like(ys, Z)
        pts = proj_points(X, Y, Zs, f, cx, cy)
        color, thick = style_for_grid(x, (60, 220, 60), 2)
        draw_curve(frame, pts, color, thick)

# dibuja la rejilla del suelo
def draw_ground_grid(frame, x_span, Xoff, A, Z, f, cx, cy, h):
    den = max(h - cy, 1.0)
    z_visible = max(0.25, (f * A) / den)

    z_near = min(max(0.25, 0.8 * z_visible), max(0.25, Z - 0.1))
    z_far = Z
    if z_near >= z_far:
        z_near = max(0.25, z_far - 0.5)

    z_vals = np.arange(z_near, z_far + 1e-9, 0.5)
    xg = np.arange(-x_span, x_span + 1e-9, 0.5)

    for x in xg:
        Zline = np.linspace(z_near, z_far, 180)
        X = np.full_like(Zline, x + Xoff)
        Y = np.full_like(Zline, A)
        pts = proj_points(X, Y, Zline, f, cx, cy)
        draw_curve(frame, pts, (190, 190, 190), 1)

    for z in z_vals:
        if abs(z - Z) < 0.06:
            continue
        X = xg + Xoff
        Y = np.full_like(xg, A)
        Zline = np.full_like(xg, z)
        pts = proj_points(X, Y, Zline, f, cx, cy)
        if abs(z - round(z)) < 1e-6:
            draw_curve(frame, pts, (220, 220, 220), 1)
        else:
            draw_curve(frame, pts, (140, 140, 140), 1)

# enumera las alturas en la rejilla frontal
def draw_height_labels(frame, y_max, Xoff, A, Z, f, cx, cy, w, h):
    for ym in range(0, int(max(3, math.floor(y_max))) + 1):
        pp = proj_points(np.array([Xoff]), np.array([A - ym]), np.array([Z]), f, cx, cy)[0]
        if 0 <= pp[0] < w and 0 <= pp[1] < h:
            putText(frame, str(ym), orig=(int(pp[0] + 4), int(pp[1] + 4)), color=(235, 235, 235), div=3, scale=0.9, thickness=1)

# texto informativo 
def draw_hud(frame, hfov, f, w, h, Z, A):
    putText(frame, f"FOV={hfov:.1f} deg, f={f:.0f}px ({w}x{h})", orig=(5, 16), color=(255, 255, 255), div=3, scale=0.9, thickness=1)
    putText(frame, f"Z={Z:.1f} m", orig=(5, 31), color=(255, 255, 255), div=3, scale=0.9, thickness=1)
    putText(frame, f"alt={A:.1f} m", orig=(5, 46), color=(255, 255, 255), div=3, scale=0.9, thickness=1)

# distancia entre dos puntos en pixeles y en centímetros
def draw_measurement(frame, points, Z, f):
    if len(points) == 2:
        p1 = np.array(points[0], dtype=float)
        p2 = np.array(points[1], dtype=float)
        pm = np.round((p1 + p2) / 2).astype(int)

        d_pix = float(np.linalg.norm(p2 - p1))
        d_m = d_pix * Z / max(f, 1e-9)
        d_cm = 100.0 * d_m

        p1i = tuple(np.round(p1).astype(int))
        p2i = tuple(np.round(p2).astype(int))
        cv.circle(frame, p1i, 4, (0, 0, 255), -1, cv.LINE_AA)
        cv.circle(frame, p2i, 4, (0, 0, 255), -1, cv.LINE_AA)
        cv.line(frame, p1i, p2i, (0, 0, 255), 2, cv.LINE_AA)

        putText(frame, f"{d_pix:.1f} pix", orig=(int(pm[0] + 8), int(pm[1] - 16)), color=(255, 255, 255), div=3, scale=0.9, thickness=1)
        putText(frame, f"{d_cm:.1f} cm", orig=(int(pm[0] + 8), int(pm[1] - 2)), color=(0, 255, 255), div=3, scale=0.9, thickness=1)
    elif len(points) == 1:
        p = tuple(points[0])
        cv.circle(frame, p, 4, (0, 0, 255), -1, cv.LINE_AA)

# lee la matriz de calibración
def read_calibration():
    calib_file = Path(__file__).resolve().with_name("calib.txt")
    if not calib_file.exists():
        return None

    data = np.loadtxt(str(calib_file))
    K = data[:9].reshape(3, 3)
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, cx, cy

calib = read_calibration()

if calib is None:
    print("No se encuentra calib.txt")
    exit(1)
else:
    fx_calib, cx_calib, cy_calib = calib
    print("Parámetros de calibración:")
    print(f"fx={fx_calib:.1f}, cx={cx_calib:.1f}, cy={cy_calib:.1f}")

show_ground_grid = True

for key, frame in autoStream():
    h, w = frame.shape[:2]

    cx = cx_calib
    cy = cy_calib
    hfov = slider_fov.value
    Z = slider_Z.value
    A = slider_A.value
    Xoff = slider_X.value

    f = w / (2 * math.tan(math.radians(hfov / 2)))

    cv.line(frame, (0, int(round(cy))), (w, int(round(cy))), (128, 128, 128), 1, cv.LINE_AA)

    xs, ys, x_span, y_max = build_front_grid_ranges(w, cx, cy, Z, f, A)
    draw_front_grid(frame, xs, ys, Xoff, A, Z, f, cx, cy)

    if show_ground_grid:
        draw_ground_grid(frame, x_span, Xoff, A, Z, f, cx, cy, h)

    draw_height_labels(frame, y_max, Xoff, A, Z, f, cx, cy, w, h)
    draw_hud(frame, hfov, f, w, h, Z, A)
    draw_measurement(frame, points, Z, f)

    cv.imshow("medidor", frame)

    if key == 27:
        break

cv.destroyAllWindows()