FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    cmake \
    unzip \
    libegl1 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libdrm2 \
    python3-tk \
    tk \
    libglib2.0-0

ARG FUNCTION_DIR="/function"

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

# Update pip
RUN pip install --upgrade --ignore-installed pip wheel six setuptools \
    && pip install --upgrade --no-cache-dir --ignore-installed \
        awslambdaric \
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
        urllib3 \
        torch \
        torchvision -f https://download.pytorch.org/whl/torch_stable.html \
        tensorflow-cpu \
        imageai \
        opencv-python \
        awslambdaric

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

COPY tiny-yolov3.pt ./

# Add Lithops
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
