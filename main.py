from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import VERSION

from core.config import appConfig
from routers import chat, charts,report,chat_logs, threads

from core.logging_config import configure_logging
from core.swagger import custom_openapi
from middlewares.logging import LoggingMiddleware


configure_logging()


app = FastAPI(
    title="LangChain SQL Chatbot API",
    description="AI-powered conversational SQL, charts, and reports API",
    version=appConfig.VERSION,
    docs_url="/swagger",     # Swagger UI
    redoc_url="/redoc",      # ReDoc
    openapi_url="/openapi.json"
)

app.openapi = lambda: custom_openapi(app)
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=appConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(charts.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(chat_logs.router, prefix="/api")
# app.include_router(threads.router, prefix="/api")

@app.get("/")
def home():
    return {"message": "LangChain SQL Chatbot API is running"}
