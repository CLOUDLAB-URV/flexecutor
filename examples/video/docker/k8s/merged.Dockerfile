FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make wget cmake unzip \
    libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxext6 libxfixes3 libdrm2 \
    libwayland-client0 \
    ffmpeg \
    tk python3-tk \
    zip redis-server curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Update pip and install Python packages
RUN pip install --upgrade --no-cache-dir pip wheel setuptools six \
    && pip install --no-cache-dir \
    boto3 \
    redis \
    httplib2 \
    requests \
    numpy \
    scipy \
    pandas \
    pika \
    kafka-python \
    cloudpickle \
    ps-mem \
    tblib \
    psutil \
    moviepy \
    Pillow \
    opencv-python \
    kubernetes \
    flask \
    gevent \
    PyYAML \
    urllib3 \
    torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    tensorflow-cpu \
    imageai \
    wrapt

COPY ../../tiny-yolov3.pt ./


ENV PYTHONUNBUFFERED TRUE

# Copy Lithops proxy and lib to the container image.
ENV APP_HOME /lithops
WORKDIR $APP_HOME

COPY lithops_k8s.zip .
RUN unzip lithops_k8s.zip && rm lithops_k8s.zip