from fastapi import Request, HTTPException

async def auth_middleware(request: Request, call_next):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(401, "Unauthorized")

    # Decode token → user_info
    request.state.user = {
        "companyId": "1",
        "userId": "123"
    }

    return await call_next(request)
