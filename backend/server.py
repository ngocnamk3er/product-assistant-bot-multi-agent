# server.py
import os
import datetime
from datetime import timezone
from functools import lru_cache

import uvicorn
import asyncpg
import jwt
import bcrypt
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.middleware import Middleware
from dotenv import load_dotenv
from database.database import Database  # Giả sử bạn đã tạo file database.py chứa lớp Database

# Import AuthMiddleware từ file mới
from middleware.auth import AuthMiddleware

# Tải các biến môi trường từ file .env
load_dotenv()

# --- Cấu hình ---
@lru_cache()
def get_settings():
    return {
        "database_url": os.getenv("DATABASE_URL"),
        "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
        "jwt_algorithm": os.getenv("JWT_ALGORITHM"),
        "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")),
    }

SETTINGS = get_settings()

db = Database(SETTINGS["database_url"])

# --- Các hàm tiện ích về JWT và Mật khẩu ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.now(timezone.utc) + datetime.timedelta(minutes=SETTINGS["access_token_expire_minutes"])
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SETTINGS["jwt_secret_key"], algorithm=SETTINGS["jwt_algorithm"])
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# <<<<<<< LỚP AuthMiddleware ĐÃ ĐƯỢC XÓA KHỎI ĐÂY >>>>>>>

# --- Lớp Server chính ---
class Server:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        
        routes = [
            Route("/login", self._handle_login, methods=["POST"]),
            Route("/logout", self._handle_logout, methods=["POST"]),
            Route("/users/me", self._handle_get_me, methods=["GET"]),
        ]

        # <<<<<<< CẬP NHẬT CÁCH KHỞI TẠO MIDDLEWARE >>>>>>>
        # Truyền db và SETTINGS vào AuthMiddleware khi khởi tạo
        middleware = [
            Middleware(AuthMiddleware, db=db, settings=SETTINGS)
        ]

        self.app = Starlette(
            debug=True,
            routes=routes,
            middleware=middleware,
            on_startup=[db.connect],
            on_shutdown=[db.disconnect],
        )

    async def _handle_login(self, request: Request):
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        if not username or not password:
            return JSONResponse({"error": "Username and password are required"}, status_code=400)

        user = await db.fetch_one("SELECT * FROM users WHERE username = $1", username)
        if not user or not verify_password(password, user['password_hash']):
            return JSONResponse({"error": "Incorrect username or password"}, status_code=401)
        
        access_token = create_access_token(data={"sub": str(user['id'])})
        return JSONResponse({"access_token": access_token, "token_type": "bearer"})

    async def _handle_logout(self, request: Request):
        return JSONResponse({"message": "Logout successful. Please clear your token."})

    async def _handle_get_me(self, request: Request):
        current_user = request.state.user
        if not current_user:
            return JSONResponse(status_code=401, content={"detail": "Authentication credentials were not provided."})
        
        return JSONResponse(current_user)

    def start(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

if __name__ == "__main__":
    server = Server()
    server.start()