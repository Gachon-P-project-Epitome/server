# 1. Python 3.9 이상의 이미지 사용
# FROM python:3.10-slim
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# 2. 작업 디렉토리 설정
WORKDIR /app

# # 3. 필요한 패키지 설치 (예: gcc 설치는 일부 라이브러리에 필요할 수 있습니다)
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     gcc \
#     libsndfile1 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*


# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

# NVIDIA GPG 키 추가 및 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y curl && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        gcc \
        libsndfile1 \
        ffmpeg \
        python3.10 python3.10-distutils python3.10-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean




# 3. get-pip.py를 사용하여 pip 설치
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# 4. Python 3.10을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 4. 필요한 Python 패키지 복사 및 설치
COPY requirements.txt /app/requirements.txt

# 5. requirements.txt 파일 수정
RUN sed -i 's/tensorboard==2.18.0/tensorboard==2.15.0/' /app/requirements.txt  # tensorboard 버전 변경
RUN sed -i 's/tensorflow==2.18.0/tensorflow==2.15.0/' /app/requirements.txt
RUN sed -i 's/ml-dtypes==0.4.1/ml-dtypes==0.2.0/' /app/requirements.txt  # ml-dtypes 버전 변경
RUN sed -i 's/wrapt==1.17.0/wrapt<1.15/' /app/requirements.txt  # wrapt 버전 변경
RUN sed -i 's/protobuf==5.29.1/protobuf>=3.19.6,<4.24/' /app/requirements.txt  # protobuf 버전 변경
RUN sed -i 's/keras==3.7.0/keras==2.15.0/' /app/requirements.txt  # keras 버전 변경
RUN sed -i 's/numpy==2.0.2/numpy==1.23.5/' /app/requirements.txt  # numpy 버전 변경



RUN pip install --no-cache-dir -r requirements.txt

# 5. Flask 애플리케이션 코드 복사
COPY . /app

# 6. Python path 설정 (optional, 경로 문제 방지용)
ENV PYTHONPATH=/app

# 7. Flask 애플리케이션 실행
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5001
CMD ["flask", "run", "--port=5001"]
