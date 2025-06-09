# server.py
import os
import datetime
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

# --- Lớp xử lý Database ---
class Database:
    def __init__(self, db_url):
        self._db_url = db_url
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=self._db_url, min_size=1, max_size=10)
            print("Database connection pool created.")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("Database connection pool closed.")

    async def fetch_one(self, query, *params):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *params)

db = Database(SETTINGS["database_url"])

# --- Các hàm tiện ích về JWT và Mật khẩu ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=SETTINGS["access_token_expire_minutes"])
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
        
        access_token = create_access_token(data={"sub": user['id']})
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