FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc gcc-c++ make wget cmake unzip

RUN yum install -y libEGL libGL libX11 libXext libXfixes libdrm

# install epel-release
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
RUN rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm

RUN yum install -y libwayland-client

RUN wget http://mirror.centos.org/centos/7/os/x86_64/Packages/libva-1.8.3-1.el7.x86_64.rpm && rpm -i libva-1.8.3-1.el7.x86_64.rpm

RUN yum install -y ffmpeg

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
        moviepy \
        Pillow \
        opencv-python

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
