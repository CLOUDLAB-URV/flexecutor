FROM python:3.10-slim

# Dependencies for graph analysis
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    g++ \
    cmake \
    unzip \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

ARG FUNCTION_DIR="/function"

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

#Update pip
RUN pip install --upgrade --ignore-installed pip wheel setuptools six \
    && pip install --no-cache-dir --ignore-installed \
        awslambdaric \
        lithops \
        boto3 \
        cloudpickle \
        networkx \
        python-louvain \
        numpy \
        matplotlib \
        psutil \
        tblib \
        deap \
        lightgbm \
        scipy

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Add Lithops  
COPY lithops_lambda.zip ${FUNCTION_DIR}
RUN unzip lithops_lambda.zip \
    && rm lithops_lambda.zip \
    && mkdir -p handler \
    && touch handler/__init__.py \
    && mv entry_point.py handler/

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.entry_point.lambda_handler" ]
