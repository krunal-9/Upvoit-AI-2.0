import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    JWT_SECRET = os.getenv("JWT_KEY")
    JWT_ALGORITHM = "HS256"
    VERSION = os.getenv("VERSION", "1.0.0")

    CORS_ORIGINS = ["*"]

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

    raw_origins = os.getenv("CORS_ORIGINS", "")

    CORS_ORIGINS = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    mongo_chat_collection = os.getenv("MONGO_CHAT_COLLECTION")
    mongo_report_collection = os.getenv("MONGO_REPORT_COLLECTION")
    mongo_dashboard_collection = os.getenv("MONGO_DASHBOARD_COLLECTION")

    checkpointing_db = os.getenv("CHECKPOINTING_DB", "checkpointing_db")
    checkpoint_collection = os.getenv("CHECKPOINT_COLLECTION", "checkpoints")
    checkpoint_writes_collection = os.getenv("CHECKPOINT_WRITES_COLLECTION", "checkpoint_writes")

appConfig = Config()