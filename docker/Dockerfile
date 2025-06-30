# NVIDIA CUDA가 포함된 파이썬 환경의 베이스 이미지 선택
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 소스 코드 전체 복사
COPY . .

# 6. 컨테이너 실행 시 기본 명령어 설정 
# ENTRYPOINT ["python", "src/train.py"]