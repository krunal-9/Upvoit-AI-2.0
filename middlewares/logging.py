import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from core.logging_config import log_agent_step, log_error

logger = logging.getLogger("http")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Make request_id available everywhere
        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)

            duration_ms = round((time.time() - start_time) * 1000, 2)
            user = getattr(request.state, "user", {}) or {}

            log_agent_step(
                agent_name="HTTP",
                message="Request completed",
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "companyId": user.get("companyId"),
                    "userId": user.get("userId"),
                },
                level="INFO",
            )

            # Expose request_id to client
            response.headers["X-Request-Id"] = request_id
            return response

        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            user = getattr(request.state, "user", {}) or {}

            log_error(
                agent_name="HTTP",
                error=exc,
                context={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "companyId": user.get("companyId"),
                    "userId": user.get("userId"),
                },
            )

            raise
