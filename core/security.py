import jwt
from fastapi import Request, HTTPException
from core.config import appConfig

def token_required(request: Request) -> dict:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid token")

    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(token, appConfig.JWT_SECRET, algorithms=[appConfig.JWT_ALGORITHM])
        request.state.user = payload
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
    except Exception as e:
        raise HTTPException(401, f"Token decoding error: {str(e)}")
