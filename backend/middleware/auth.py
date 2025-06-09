# middleware/auth.py
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class AuthMiddleware(BaseHTTPMiddleware):
    # Chúng ta thêm db và settings vào hàm khởi tạo
    # để truyền các dependency cần thiết vào.
    def __init__(self, app, db, settings):
        super().__init__(app)
        self.db = db
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        public_paths = ["/login", "/docs", "/openapi.json"]
        if request.url.path in public_paths:
            return await call_next(request)

        request.state.user = None

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Not authenticated: Missing or invalid token"},
            )

        token = auth_header.split(" ")[1]
        
        print(f"🔍 Decoding token: {token}")  # Log token for debugging
        
        print(f"🔍 Using settings: {self.settings}")  # Log settings for debugging
        
        print("🔍 Decoding JWT with secret key:", self.settings["jwt_secret_key"])  # Log secret key for debugging
        
        try:
            # Sử dụng self.settings thay vì SETTINGS toàn cục
            payload = jwt.decode(
                jwt = token, 
                key = self.settings["jwt_secret_key"], 
                algorithms=[self.settings["jwt_algorithm"]]
            )
            user_id: int = int(payload.get("sub"))
            print(f"🔍 Extracted user_id: {user_id}")  # Log user_id for debugging
            if user_id is None:
                raise jwt.InvalidTokenError
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"error": "Token has expired"})
        except jwt.InvalidTokenError as e:  # Bắt đối tượng lỗi vào biến 'e'
            # In ra thông báo lỗi chi tiết trong terminal của server
            print(f"🔴 JWT DECODE ERROR: {e}") 
            
            # Trả về thông báo lỗi chi tiết cho client (Postman) để dễ debug
            return JSONResponse(
                status_code=401, 
                content={"error": "Invalid token", "detail": str(e)}
            )

        # Sử dụng self.db thay vì db toàn cục
        user_record = await self.db.fetch_one(
            "SELECT id, username, email, full_name, address, phone_number FROM users WHERE id = $1", 
            user_id
        )
        if not user_record:
            return JSONResponse(status_code=401, content={"error": "User not found"})
        
        request.state.user = dict(user_record)
        
        response = await call_next(request)
        return response