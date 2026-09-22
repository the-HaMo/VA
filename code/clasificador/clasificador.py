#!/usr/bin/env python

import cv2 as cv
import argparse
import os
import glob
import importlib
from datetime import datetime
from umucv.stream import autoStream, sourceArgs
from umucv.util import putText, Help


def list_methods(methods_dir):
    files = glob.glob(os.path.join(methods_dir, '*.py'))
    names = []
    for f in files:
        b = os.path.basename(f)
        if b.startswith('_'):
            continue
        names.append(os.path.splitext(b)[0])
    return sorted(names)


def load_method(method_name):
    module = importlib.import_module(f'methods.{method_name}')
    if not hasattr(module, 'Method'):
        raise ImportError(f"El módulo methods.{method_name} no define la clase Method")
    return module.Method()


def draw_hud(frame, method_name, nmodels, pending_saves, results):
    putText(frame, f"Metodo: {method_name}", orig=(20, 25), color=(255, 255, 255), div=3, scale=1, thickness=1)
    putText(frame, f"Modelos: {nmodels}", orig=(20, 50), color=(255, 255, 255), div=3, scale=1, thickness=1)
    putText(frame, f"Pendientes guardar: {pending_saves}", orig=(20, 75), color=(255, 255, 255), div=3, scale=1, thickness=1)

    if not results:
        putText(frame, "Top: sin coincidencias", orig=(20, 115), color=(0, 255, 255), div=3, scale=1, thickness=1)
        return

    best = results[0]
    putText(frame, f"Top: {best['name']} ({best['score']:.2f})", orig=(20, 115), color=(0, 255, 0), div=3, scale=1, thickness=1)

    for i, r in enumerate(results[:5]):
        putText(
            frame,
            f"{r['name']}: {r['score']:.2f}",
            orig=(20, 145 + i * 22),
            color=(255, 255, 255),
            div=3,
            scale=0.9,
            thickness=1,
        )

def main():
    methods_dir = os.path.join(os.path.dirname(__file__), 'methods')
    available_methods = list_methods(methods_dir)

    help = Help(
        """
        CLASIFICADOR

        h: mostrar/ocultar ayuda
        c: capturar modelo del frame actual
        s: guardar capturas en carpeta de modelos
        ESC: salir
        """
    )

    parser = argparse.ArgumentParser()
    sourceArgs(parser)
    parser.add_argument('--models', type=str, required=True, help='Carpeta de modelos')
    parser.add_argument('--method', type=str, required=True,
                        help=f"Nombre del método ({', '.join(available_methods)})")
    args = parser.parse_args()

    # 1. Carga dinámica del método
    try:
        method = load_method(args.method)
    except Exception as e:
        print(f"Error cargando método '{args.method}': {e}")
        print(f"Métodos disponibles: {available_methods}")
        return

    # 2. Precomputar modelos
    print(f"Precomputando modelos desde {args.models}...")
    model_files = glob.glob(os.path.join(args.models, '*.*'))
    trained_models = []

    for f in model_files:
        img = cv.imread(f)
        if img is None: continue
        name = os.path.basename(f).split('.')[0]
        # Cada método extrae lo que necesita (SIFT descriptors, embeddings, etc.)
        data = method.prepare(img)
        if data is not None:
            trained_models.append({'name': name, 'data': data})

    print(f"Modelos cargados: {[m['name'] for m in trained_models]}")

    pending_saves = []

    # 3. Bucle principal
    for key, frame in autoStream():
        help.show_if(key, ord('h'))

        if key == ord('c'):
            data = method.prepare(frame)
            if data is not None:
                name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trained_models.append({'name': name, 'data': data})
                pending_saves.append((name, frame.copy()))
                print(f"imagen como modelo añadido: {name}")
            else:
                print('No se pudo extraer un modelo válido del frame actual.')

        if key == ord('s') and pending_saves:
            os.makedirs(args.models, exist_ok=True)
            for name, img in pending_saves:
                path = os.path.join(args.models, f"{name}.png")
                cv.imwrite(path, img)
                print(f"guardado la  imagen: {path}")
            pending_saves = []

        # Procesamos el frame actual con el método seleccionado
        results = method.compare(frame, trained_models)
    
        draw_hud(frame, args.method, len(trained_models), len(pending_saves), results)

        cv.imshow('Clasificador', frame)
        if key == 27: break

if __name__ == '__main__':
    main()