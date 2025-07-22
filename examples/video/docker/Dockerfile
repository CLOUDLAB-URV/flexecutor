FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    cmake \
    unzip \
    tar \
    xz-utils \
    libegl1 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libdrm2 \
    libwayland-client0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

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
        "moviepy<2" \
        Pillow \
        opencv-python \
        # TODO: avoid to install flexecutor dependencies \
        # -----
        deap \
        boto3 \
        cloudpickle \
        scipy \
        numpy \
        pandas
        # -----


# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Add Lithops
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
