#!/usr/bin/env python

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Carga las variables desde el archivo token.env
# Asegúrate de tener un archivo token.env con:
# TOKEN=tu_token_aqui
# USER_ID=tu_chat_id_aqui
env_path = Path(__file__).resolve().with_name('token.env')
load_dotenv(env_path)

TELEGRAM_TOKEN = os.environ.get('TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('USER_ID')

def notify_telegram(message):
    """Envía un mensaje de texto a Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: No se han encontrado las credenciales de Telegram en token.env")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

def send_video_telegram(video_path):
    """Envía el archivo de video a Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: No se han encontrado las credenciales de Telegram en token.env")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID}, files=files, timeout=30)
        print(f"Video enviado: {video_path}")
    except Exception as e:
        print(f"Error enviando video a Telegram: {e}")


def remove_video(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass