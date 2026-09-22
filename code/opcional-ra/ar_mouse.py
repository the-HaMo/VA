#!/usr/bin/env python3

import time
import cv2 as cv
import numpy as np
import numpy.linalg as la
from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose, Kfov, sepcam
from umucv.util     import lineType, putText
from umucv.contours import extractContours, redu


# ── marcador (6 vértices, mundo z=0)
marker = np.array([
    [0.0, 0.0, 0],
    [0.0, 1.0, 0],
    [0.5, 1.0, 0],
    [0.5, 0.5, 0],
    [1.0, 0.5, 0],
    [1.0, 0.0, 0],
])

def polygons(cs, n, prec=2):
    for eps in [prec, prec*2, prec*4, prec*8, prec*16]:
        rs = [redu(c, eps) for c in cs]
        found = [r for r in rs if len(r) == n]
        if found:
            return found
    return []

def rots(c):
    return [np.roll(c, k, 0) for k in range(len(c))]

def bestPose(K, view, model):
    poses = [Pose(K, v.astype(float), model) for v in rots(view)]
    return sorted(poses, key=lambda p: p.rms)[0]


# ── esfera (malla triangulada con patrón damero negro/blanco)
def sphere_mesh(r=0.12, n_lat=12, n_lon=18):
    verts = []
    for i in range(n_lat + 1):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            verts.append([r * np.sin(th) * np.cos(ph),
                          r * np.sin(th) * np.sin(ph),
                          r * np.cos(th)])
    verts = np.array(verts)

    tris, colors = [], []
    for i in range(n_lat):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            v0 = i * n_lon + j
            v1 = i * n_lon + j2
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + j2
            c = np.array([30, 30, 30], np.float32) if (i + j) % 2 == 0 \
                else np.array([245, 245, 245], np.float32)
            tris.append([v0, v2, v1]); colors.append(c)
            tris.append([v1, v2, v3]); colors.append(c)
    return verts, np.array(tris, int), np.array(colors, np.float32)


SPHERE_R = 0.12
sphere_verts, sphere_tris, sphere_tri_colors = sphere_mesh(r=SPHERE_R)

LIGHT_DIR = np.array([1.0, -1.0, 2.0])
LIGHT_DIR /= la.norm(LIGHT_DIR)
AMBIENT = 0.35

SPEED = 0.8

sphere_pos    = np.array([0.5, 0.5, 0.0])
sphere_target = np.array([0.5, 0.5, 0.0])
sphere_R      = np.eye(3)
click_xy      = None


def mouse_cb(event, x, y, flags, param):
    global click_xy
    if event == cv.EVENT_LBUTTONDOWN:
        click_xy = (x, y)


def draw_sphere(frame, M):
    _, _, C = sepcam(M)
    C = C.flatten()

    rot_v   = sphere_verts @ sphere_R.T
    world_v = rot_v + sphere_pos + np.array([0, 0, SPHERE_R])
    n_world = rot_v / (la.norm(rot_v, axis=1, keepdims=True) + 1e-9)
    img_v   = htrans(M, world_v)

    visible = []
    for k, tri in enumerate(sphere_tris):
        v0, v1, v2 = tri
        n = n_world[v0] + n_world[v1] + n_world[v2]
        n /= la.norm(n) + 1e-9
        centroid = (world_v[v0] + world_v[v1] + world_v[v2]) / 3
        view = C - centroid
        view /= la.norm(view) + 1e-9
        if n @ view <= 0:
            continue
        shade = AMBIENT + (1 - AMBIENT) * max(0.0, n @ LIGHT_DIR)
        color = (sphere_tri_colors[k] * shade).clip(0, 255).astype(int)
        depth = la.norm(centroid - C)
        visible.append((depth, tri, color))

    visible.sort(key=lambda t: -t[0])

    for _, tri, color in visible:
        pts = img_v[tri].astype(int)
        cv.fillPoly(frame, [pts], tuple(int(c) for c in color), cv.LINE_AA)


H_fr, W_fr = next(autoStream())[1].shape[:2]
K = Kfov((W_fr, H_fr), 60)

cv.namedWindow('ar-mouse')
cv.setMouseCallback('ar-mouse', mouse_cb)

prev_t = time.time()

for key, frame in autoStream():
    now = time.time()
    dt  = max(now - prev_t, 1e-3)
    prev_t = now

    g    = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cs   = extractContours(g, minarea=2, reduprec=3)
    good = polygons(cs, 6, 3)

    M, p_rms = None, 0.0
    best_rms  = 1e6
    for c in good:
        p = bestPose(K, c, marker)
        if p.rms < best_rms:
            best_rms = p.rms
        if p.rms < 8:
            M, p_rms = p.M, p.rms
            break

    if M is not None:
        H = M[:, [0, 1, 3]]

        if click_xy is not None:
            uv  = np.array([click_xy[0], click_xy[1], 1.0])
            xyw = la.inv(H) @ uv
            xyw /= xyw[2]
            sphere_target = np.array([xyw[0], xyw[1], 0.0])
            click_xy = None

        delta = sphere_target - sphere_pos
        dist  = la.norm(delta)
        if dist > 1e-4:
            step  = min(SPEED * dt, dist)
            move  = delta / dist * step
            sphere_pos = sphere_pos + move
            axis = np.cross(np.array([0.0, 0.0, 1.0]), move)
            an   = la.norm(axis)
            if an > 1e-9:
                axis  = axis / an
                dR    = cv.Rodrigues(axis * (step / SPHERE_R))[0]
                sphere_R = dR @ sphere_R

        cv.drawContours(frame, [htrans(M, marker).astype(int)],
                        -1, (0, 180, 255), 2, lineType)
        tgt_px = htrans(M, sphere_target.reshape(1, 3))[0].astype(int)
        cv.drawMarker(frame, tuple(tgt_px), (255, 0, 255), cv.MARKER_CROSS, 14, 2)
        draw_sphere(frame, M)

        putText(frame, f'pos=({sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}) | '
                       f'dist={dist:.2f} | rms={p_rms:.2f}')
    else:
        msg = f'marcador no detectado  rms={best_rms:.1f}' if good else 'marcador no detectado'
        putText(frame, msg)

    cv.imshow('ar-mouse', frame)
    if key in (27, ord('q')):
        break

cv.destroyAllWindows()
