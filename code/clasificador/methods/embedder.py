import os

import cv2 as cv
import numpy as np
from umucv.util import check_and_download

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite'


class Method:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'embedder.tflite')
        check_and_download(model_path, MODEL_URL)

        options = vision.ImageEmbedderOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            l2_normalize=True,
            quantize=False,
        )
        self.embedder = vision.ImageEmbedder.create_from_options(options)

    def _embed(self, img_bgr):
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        emb = self.embedder.embed(mp_image).embeddings
        if not emb:
            return None
        vec = emb[0].embedding
        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float32)

    def _show_best(self, frame, best_name, best_score, best_model_img):
        h, w = frame.shape[:2]
        left = frame.copy()
        right = cv.resize(best_model_img, (w, h), interpolation=cv.INTER_LINEAR)

        cv.putText(left, 'Actual', (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(right, f"Modelo: {best_name}", (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

        vis = np.hstack([left, right])
        cv.putText(vis, f"score={best_score:.3f}", (10, 50), cv.FONT_HERSHEY_PLAIN, 1.2, (200, 255, 200), 1, cv.LINE_AA)
        cv.imshow('MP embedder', vis)

    def _show_live(self, frame, text='Sin modelos validos'):
        left = frame.copy()
        right = left.copy()
        cv.putText(left, 'Actual', (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(right, text, (10, 25), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1, cv.LINE_AA)
        cv.imshow('MP embedder', np.hstack([left, right]))

    def prepare(self, img):
        emb = self._embed(img)
        if emb is None:
            return None
        return {
            'emb': emb,
            'img': img.copy(),
        }

    def compare(self, frame, trained_models):
        q = self._embed(frame)
        if q is None:
            self._show_live(frame, 'Sin frame valido')
            return []

        results = []
        best = None
        for model in trained_models:
            m = model['data']
            if m is None:
                continue

            emb = m['emb'] if isinstance(m, dict) else m
            # Con l2_normalize=True, el producto escalar equivale al coseno.
            score = float(np.dot(q, emb))
            results.append({'name': model['name'], 'score': score})

            if best is None or score > best['score']:
                best = {
                    'name': model['name'],
                    'score': score,
                    'data': m,
                }

        if best is not None and isinstance(best['data'], dict):
            self._show_best(frame, best['name'], best['score'], best['data']['img'])
        else:
            self._show_live(frame, 'Sin modelos validos')

        return sorted(results, key=lambda x: x['score'], reverse=True)
