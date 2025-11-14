import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://116.39.208.72:26443")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# 필수값 체크
if ADMIN_EMAIL is None:
    raise ValueError("❌ ADMIN_EMAIL is missing in .env")

if ADMIN_PASSWORD is None:
    raise ValueError("❌ ADMIN_PASSWORD is missing in .env")
