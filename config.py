import os
from dotenv import load_dotenv

load_dotenv()

# .env dosyasında değişkenleri çeker
CAMERA_ID = os.getenv("CAMERA_ID")
MODEL_PATH = os.getenv("MODEL_PATH")
LOG_FİLE = os.getenv("LOG_FİLE")
FOTO_FOLDER = os.getenv("FOTO_FOLDER")
SENDER_MAIL = os.getenv("SENDER_MAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_MAIL = os.getenv("RECIPIENT_MAIL")
# Gerekli değişkenleri belirler
ERROR_THRESHOLD = 40
ERROR_RATE = 0.7
MAX_LOG_WAIT_MINUTE = 5