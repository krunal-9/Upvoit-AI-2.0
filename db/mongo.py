from pymongo import MongoClient
from core.config import appConfig

mongo_client = MongoClient(appConfig.MONGO_URI)

db = mongo_client[appConfig.mongo_db]
chat_logs = db[appConfig.mongo_chat_collection]
report_logs = db[appConfig.mongo_report_collection]
dashboard_logs = db[appConfig.mongo_dashboard_collection]

checkpoint_db = mongo_client[appConfig.checkpointing_db]
checkpoints = checkpoint_db[appConfig.checkpoint_collection]
checkpoint_writes = checkpoint_db[appConfig.checkpoint_writes_collection]


