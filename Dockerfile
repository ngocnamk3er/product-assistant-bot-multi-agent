# Dockerfile
FROM python:3.12.7-slim AS base

# Cài đặt curl và các gói cần thiết khác nếu có
# apt-get update chỉ cần chạy một lần trước các lệnh apt-get install
# --no-install-recommends để tránh cài các gói không cần thiết
# rm -rf /var/lib/apt/lists/* để dọn dẹp cache, giảm kích thước image
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt uv (đã có trong Dockerfile của bạn, giữ nguyên)
RUN pip install uv

COPY pyproject.toml uv.lock ./
# Hoặc nếu bạn có file requirements.txt:
# COPY requirements.txt ./

# Sử dụng uv để cài đặt dependencies (đã có, giữ nguyên)
RUN uv pip install . --system
# Hoặc:
# RUN uv pip install -r requirements.txt --system
# Hoặc:
# RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m"]