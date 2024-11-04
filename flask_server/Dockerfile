# 1. Python 3.9 이상의 이미지 사용
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 패키지 설치 (예: gcc 설치는 일부 라이브러리에 필요할 수 있습니다)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 필요한 Python 패키지 복사 및 설치
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 5. Flask 애플리케이션 코드 복사
COPY . /app

# 6. Python path 설정 (optional, 경로 문제 방지용)
ENV PYTHONPATH=/app

# 7. Flask 애플리케이션 실행
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5001
CMD ["flask", "run", "--port=5001"]
