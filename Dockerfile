FROM cnstark/pytorch:1.10.2-py3.9.12-cuda11.3.1-devel-ubuntu20.04

RUN apt update && apt upgrade -y &&\
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    tzdata git libsparsehash-dev unzip wget vim tmux ffmpeg libsm6 libxext6

WORKDIR /home/pcseg
COPY ./ .

# package installation
ENV PATH "/usr/.local/bin:${PATH}"
RUN pip3 install -r requirements.txt
RUN cd /home/pcseg/package &&\
    mkdir torchsparse_dir/ &&\
    unzip -o sparsehash.zip -d sparsehash &&\
    unzip -o torchsparse.zip &&\
    unzip -o range_lib.zip
RUN cd /home/pcseg/package/sparsehash/sparsehash-master &&\
    ./configure --prefix=/home/pcseg/package/torchsparse_dir/spash &&\
    make &&\
    make install
RUN pip3 install -e package/range_lib &&\
    pip3 install -e package/torchsparse


# filesystem and permissions setup
ARG UNAME=pcseg
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME &&\
    useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME &&\
    chown -R $UNAME:$UNAME /home/pcseg
USER $UNAME