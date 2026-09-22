#!/usr/bin/env python3

import os
import cv2   as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util   import putText

HERE = os.path.dirname(os.path.abspath(__file__))

mclovin   = cv.imread(os.path.join(HERE, 'mclovin.jpg'))
H_m, W_m  = mclovin.shape[:2]
m_corners = np.float32([[0,0],[W_m,0],[W_m,H_m],[0,H_m]])

# ISO ID-1 a 10 px/mm → 856×540
W_C, H_C  = 856, 540
card_canon = np.float32([[0,0],[W_C,0],[W_C,H_C],[0,H_C]])

# foto del carnet
photo_card = np.float32([[643, 278], [832, 282], [832, 517], [647, 517]])

MW, MH    = 171, 108
mini_dst  = np.float32([[0,0],[MW,0],[MW,MH],[0,MH]])

def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax, uc, axis=-1)

def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[..., [-1]]

def line_h(pts):
    vx,vy,x0,y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.cross(homog([x0, y0]), [vx, vy, 0])   # punto × dirección

def cross_h(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-10: return None
    return inhomog(p)

def score(frame, pts, rot):
    rp = np.roll(pts, rot, axis=0)
    H, _ = cv.findHomography(rp, mini_dst)
    if H is None: return 0
    mini = cv.warpPerspective(frame, H, (MW, MH))
    roi  = mini[:int(MH*0.3), int(MW*0.5):]
    hsv  = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    r1   = cv.inRange(hsv, (0,  30, 60), (15, 255,255))
    r2   = cv.inRange(hsv, (155,30, 60), (180,255,255))
    return int(cv.bitwise_or(r1,r2).sum()//255)

def detect_card(frame):
    """Devuelve las coordenadas [TL,TR,BR,BL] del carnet o None."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    hsv     = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    colored = cv.inRange(hsv, (0,40,40), (180,255,255))
    mask    = cv.bitwise_and(mask, cv.bitwise_not(colored))
    mask    = cv.morphologyEx(mask, cv.MORPH_CLOSE,
                  cv.getStructuringElement(cv.MORPH_RECT,(7,7)))

    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in sorted(cnts, key=cv.contourArea, reverse=True)[:5]:
        if cv.contourArea(c) < 4000: break
        (cx,cy),(rw,rh),ang = cv.minAreaRect(c)
        if min(rw,rh)<5: continue
        if not 1.2 < max(rw,rh)/min(rw,rh) < 2.2: continue

        box = cv.boxPoints(((cx,cy),(rw,rh),ang))
        e0,e1 = box[1]-box[0], box[2]-box[1]
        l0,l1 = np.linalg.norm(e0), np.linalg.norm(e1)
        if l0>=l1: u,v,hu,hv = e0/l0,e1/l1,l0/2,l1/2
        else:      u,v,hu,hv = e1/l1,e0/l0,l1/2,l0/2

        pts = c.reshape(-1,2).astype(np.float32)
        cc  = pts - np.array([cx,cy],np.float32)
        pu,pv = cc@u, cc@v
        S,C = 0.85,0.55
        on_pu=(pu> S*hu)&(np.abs(pv)<C*hv)
        on_nu=(pu<-S*hu)&(np.abs(pv)<C*hv)
        on_pv=(pv> S*hv)&(np.abs(pu)<C*hu)
        on_nv=(pv<-S*hv)&(np.abs(pu)<C*hu)
        if min(on_pu.sum(),on_nu.sum(),on_pv.sum(),on_nv.sum())<5: continue

        Lpu= line_h(pts[on_pu]); Lnu= line_h(pts[on_nu])
        Lpv= line_h(pts[on_pv]); Lnv= line_h(pts[on_nv])
        c_nn= cross_h(Lnu,Lnv); c_pn= cross_h(Lpu,Lnv)
        c_pp= cross_h(Lpu,Lpv); c_np= cross_h(Lnu,Lpv)
        if any(x is None for x in (c_nn,c_pn,c_pp,c_np)): continue

        corners = np.float32([c_nn,c_pn,c_pp,c_np])
        e1 = corners[1]-corners[0]; e2 = corners[2]-corners[1]
        if e1[0]*e2[1]-e1[1]*e2[0] < 0:
            corners = corners[::-1].copy()
        best = int(np.argmax([score(frame,corners,r) for r in range(4)]))
        return np.roll(corners, best, axis=0)   # [TL,TR,BR,BL]
    return None


def best_cyclic(corners, prev):
    """Rotación que minimiza la distancia a prev."""
    best_rot, best_d = 0, float('inf')
    for r in range(4):
        d = float(np.sum((np.roll(corners, r, axis=0) - prev)**2))
        if d < best_d:
            best_d, best_rot = d, r
    return np.roll(corners, best_rot, axis=0)

prev_card_px = None

cv.namedWindow('ejer-CARD')
for key, frame in autoStream():
    H_fr, W_fr = frame.shape[:2]
    out = frame.copy()

    corners  = detect_card(frame)
    photo_px = None
    card_px  = None

    if corners is not None:
        if prev_card_px is not None:
            corners = best_cyclic(corners, prev_card_px)
        prev_card_px = card_px = corners.copy()
        H, _ = cv.findHomography(card_canon, card_px)
        if H is not None and abs(np.linalg.det(H)) > 1e-8:
            photo_px = inhomog(homog(photo_card) @ H.T).astype(np.float32)
    else:
        prev_card_px = None

    if photo_px is not None:
        M = cv.getPerspectiveTransform(m_corners, photo_px)
        cv.warpPerspective(mclovin, M, (W_fr, H_fr), out, 0, cv.BORDER_TRANSPARENT)

    if card_px is not None:
        cv.polylines(out, [card_px.astype(np.int32).reshape(-1,1,2)], True, (0,220,0), 2)
    if photo_px is not None:
        cv.polylines(out, [photo_px.astype(np.int32).reshape(-1,1,2)], True, (0,220,220), 2)
    putText(out, 'ESC/q: salir', orig=(8,18), color=(255,255,255), div=3, scale=0.8, thickness=1)

    cv.imshow('ejer-CARD', out)
    if key in (27, ord('q')): break

cv.destroyAllWindows()
