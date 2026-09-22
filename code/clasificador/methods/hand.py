import cv2 as cv
import numpy as np
import mediapipe as mp


class Method:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1,
        )

    def _extract_points(self, img_bgr):
        h, w = img_bgr.shape[:2]
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None

        hand = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark], dtype=np.float32)
        return pts

    def _draw_points(self, img, pts, color=(0, 255, 255)):
        out = img.copy()
        for x, y in pts.astype(int):
            cv.circle(out, (x, y), 2, color, -1)
        return out

    def _show_best(self, frame, qpts, best_name, best_score, best_model):
        h, w = frame.shape[:2]
        model_img = cv.resize(best_model['img'], (w, h), interpolation=cv.INTER_LINEAR)

        qvis = self._draw_points(frame, qpts, color=(0, 255, 255))

        mpts = best_model['pts'].copy()
        mh, mw = best_model['img'].shape[:2]
        mpts[:, 0] *= w / mw
        mpts[:, 1] *= h / mh
        mvis = self._draw_points(model_img, mpts, color=(0, 255, 255))

        cv.putText(qvis, 'Actual', (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(mvis, f"Modelo: {best_name}", (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

        vis = np.hstack([qvis, mvis])
        cv.putText(vis, f"score={best_score:.3f}", (10, 50), cv.FONT_HERSHEY_PLAIN, 1.2, (200, 255, 200), 1, cv.LINE_AA)
        cv.imshow('Hand procrustes', vis)

    def _show_live(self, frame, text='Sin mano detectada'):
        live = frame.copy()
        cv.putText(live, text, (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1, cv.LINE_AA)
        vis = np.hstack([live, live.copy()])
        cv.imshow('Hand procrustes', vis)

    def _normalize_shape(self, pts):
        c = np.mean(pts, axis=0, keepdims=True)
        x = pts - c
        n = np.linalg.norm(x)
        if n < 1e-8:
            return None
        return x / n

    def _procrustes_distance(self, p, q):
        p0 = self._normalize_shape(p)
        q0 = self._normalize_shape(q)
        if p0 is None or q0 is None:
            return np.inf

        # Buscamos la mejor rotacion 2D via SVD (Kabsch).
        h = p0.T @ q0
        u, _, vt = np.linalg.svd(h)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = u @ vt

        p_aligned = p0 @ r
        return float(np.linalg.norm(p_aligned - q0))

    def prepare(self, img):
        pts = self._extract_points(img)
        if pts is None:
            return None
        return {
            'pts': pts,
            'img': img.copy(),
        }

    def compare(self, frame, trained_models):
        q = self._extract_points(frame)
        if q is None:
            self._show_live(frame, 'Sin mano detectada')
            return []

        results = []
        best = None
        for model in trained_models:
            m = model['data']
            if m is None:
                continue

            pts_model = m['pts'] if isinstance(m, dict) else m
            d = self._procrustes_distance(q, pts_model)
            # Convertimos distancia a score (mas alto = mejor) para interfaz comun.
            score = 1.0 / (1.0 + d)
            results.append({'name': model['name'], 'score': score})

            if best is None or score > best['score']:
                best = {
                    'name': model['name'],
                    'score': score,
                    'data': m,
                }

        if best is not None and isinstance(best['data'], dict):
            self._show_best(frame, q, best['name'], best['score'], best['data'])
        else:
            self._show_live(frame, 'Sin modelos validos')

        return sorted(results, key=lambda x: x['score'], reverse=True)
