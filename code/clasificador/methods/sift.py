import cv2 as cv
import time
import numpy as np
from umucv.util import putText

class Method:
    def __init__(self):
        self.sift = cv.SIFT_create(nfeatures=500)
        self.matcher = cv.BFMatcher()

    def prepare(self, img):
        kp, des = self.sift.detectAndCompute(img, mask=None)
        if des is None or len(des) == 0:
            return None
        return {
            'kp': kp,
            'des': des,
            'img': img.copy(),
        }

    def compare(self, frame, trained_models):
        t0 = time.time()
        kp_frame, des_frame = self.sift.detectAndCompute(frame, mask=None)
        t1 = time.time()
        
        if des_frame is None or len(des_frame) == 0:
            vis = frame.copy()
            cv.imshow('SIFT matches', vis)
            return []

        results = []
        best_pack = None

        for model in trained_models:
            if model['data'] is None:
                continue

            des_model = model['data']['des']
            kp_model = model['data']['kp']
            if des_model is None or len(des_model) == 0:
                continue

            t2 = time.time()
            matches = self.matcher.knnMatch(des_frame, des_model, k=2)
            t3 = time.time()

            good_matches = []
            for m in matches:
                if len(m) >= 2:
                    best, second = m
                    if best.distance < 0.75 * second.distance:
                        good_matches.append(best)
            
            # El porcentaje se basa en cuántos puntos del modelo (referencia) hemos emparejado
            num_kp_model = len(kp_model)
            percentage = (len(good_matches) / num_kp_model * 100) if num_kp_model > 0 else 0
            
            results.append({
                'name': model['name'],
                'score': len(good_matches),
                'percent': percentage  # Añadimos el dato al resultado
            })

            if best_pack is None or len(good_matches) > best_pack['n']:
                best_pack = {
                    'good': good_matches,
                    'kp_model': kp_model,
                    'img_model': model['data']['img'],
                    'match_ms': 1000 * (t3 - t2),
                    'n': len(good_matches),
                    'percent': percentage
                }

        # Ordenamos por número de aciertos
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        # visualizacion
        if best_pack is not None:
            vis = cv.drawMatches(
                frame,
                kp_frame,
                best_pack['img_model'],
                best_pack['kp_model'],
                best_pack['good'],
                outImg=None,
                matchColor=(128, 255, 128),
                singlePointColor=(128, 128, 128),
                flags=0,
            )
            
            putText(vis, f"Frame: {len(kp_frame)} pts | {1000*(t1-t0):.0f} ms", orig=(5, 16))
            info_match = f"Matches: {best_pack['n']} ({best_pack['percent']:.1f}%) | {best_pack['match_ms']:.0f} ms"
            putText(vis, info_match, orig=(5, 36), color=(200, 255, 200))
        else:
            vis = frame.copy()
            flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            cv.drawKeypoints(frame, kp_frame, vis, color=(100, 150, 255), flags=flag)
            putText(vis, f"{len(kp_frame)} pts  {1000*(t1-t0):.0f} ms", orig=(5, 16))

        cv.imshow('SIFT matches', vis)

        return results