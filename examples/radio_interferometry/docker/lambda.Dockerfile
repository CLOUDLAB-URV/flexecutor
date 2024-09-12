# Python 3.10
FROM python:3.10-slim-buster

# Python 3.8, needed for COMPss
# FROM python:3.8-slim-buster

# Install initial dependencies
RUN apt-get update && apt-get install -y \
    zip redis-server curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG FUNCTION_DIR="/function"

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

# Upgrade pip and setuptools, and install Python packages
RUN pip install --upgrade setuptools six pip \
    && pip install --no-cache-dir --ignore-installed \
    flask \
    pika \
    boto3 \
    awslambdaric \
    redis \
    gevent \
    requests \
    PyYAML \
    kubernetes \
    # numpy version < 2.0.0
    numpy==1.22.4 \
    cloudpickle \
    ps-mem \
    tblib \
    matplotlib \
    psutil \
    scipy \
    wrapt \
    python-casacore

# Install additional system dependencies
RUN apt-get update \
    && apt-get install -y \
    g++ \
    libboost-all-dev \
    libhdf5-serial-dev \
    libfftw3-dev \
    libcfitsio-dev \
    libarmadillo-dev \
    liblog4cplus-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    wcslib-dev \
    casacore-dev \
    casacore-tools \
    libpython3-dev \
    libgsl-dev \
    liblua5.3-dev \
    libpng-dev \
    libgtkmm-3.0-dev \
    ninja-build \
    pkg-config \
    swig \
    wget \
    unzip \
    flex \
    bison \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y make libssl-dev && rm -rf /var/lib/apt/lists/*


RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz \
    && tar -xzvf cmake-3.21.1.tar.gz \
    && cd cmake-3.21.1 \
    && ./bootstrap && make -j8 && make install \
    && cd / && rm -rf cmake-3.21.1.tar.gz cmake-3.21.1

ENV OPENBLAS_NUM_THREADS=1 


# Install AOFlagger from source
RUN git clone https://gitlab.com/aroffringa/aoflagger.git /AOFlagger \
    && cd /AOFlagger \
    && mkdir build && cd build \
    && cmake .. \
    && make -j8 \
    && make install \
    && cd / && rm -rf /AOFlagger

# Install EveryBeam
RUN git clone https://git.astron.nl/RD/EveryBeam.git -b v0.5.8 /EveryBeam \
    && cd /EveryBeam \
    && mkdir build && cd build \
    && cmake .. \
    && make -j8 \
    && make install \
    && cd / && rm -rf /EveryBeam

# Install WSClean
RUN git clone --recursive -j8  https://gitlab.com/aroffringa/wsclean.git -b v3.4 /wsclean \
    && cd /wsclean \
    && mkdir build && cd build \
    && cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr \
    -DUSE_OPENMP=ON -DBUILD_GUI=OFF .. \
    && make -j8 \
    && make install \
    && cd / && rm -rf /wsclean

# Install DP3
RUN git clone https://github.com/lofar-astron/DP3.git -b v6.0 /DP3 \
    && cd /DP3 \
    && mkdir build && cd build \
    && cmake .. \
    && make -j4 \
    && make install \
    && cd / && rm -rf /DP3


#Install dysco

RUN git clone https://github.com/aroffringa/dysco.git /dysco \
    && cd /dysco \
    && mkdir build && cd build \
    && cmake .. \
    && make -j8 \
    && make install \
    && cd / && rm -rf /dysco


ENV PATH=$PATH:/usr/local/lib

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