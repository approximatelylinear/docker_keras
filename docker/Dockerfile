FROM debian:7.4
MAINTAINER MJ Berends <mj@dose.com>

# fix timezone
ENV TZ $TZ
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# grab what we need to install Miniconda
RUN apt-get update --fix-missing \
    && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    ssmtp \
    mailutils \
    mercurial \
    subversion \
    apt-utils

# install Miniconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-3.19.0-Linux-x86_64.sh && \
    /bin/bash /Miniconda2-3.19.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda2-3.19.0-Linux-x86_64.sh

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install global needs
RUN apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libyaml-dev \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install dependencies from conda
RUN conda install -y \
    numpy==1.11.0 \
    scipy==0.17.0 \
    scikit-learn \
    pandas==0.18.0 \
    sqlalchemy==1.0 \
    mysql-python

# install dependencies from pypi
COPY configuration/pypirc /.pypirc
COPY configuration/requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt
RUN rm /opt/requirements.txt
