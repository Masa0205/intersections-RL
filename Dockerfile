# ベース：Python入りの軽量Ubuntuイメージ
FROM ubuntu:22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/opt/sumo

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        cmake \
        libproj-dev \
        libfox-1.6-dev \
        libgdal-dev \
        libxerces-c-dev \
        libxml2-dev \
        libgl2ps-dev \
        libtool \
        libboost-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        git \
        wget \
        unzip \
        ca-certificates \
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# SUMOのビルドとインストール
WORKDIR /opt
RUN git clone --recursive https://github.com/eclipse/sumo.git && \
    cd sumo && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/sumo && \
    make -j$(nproc) && \
    make install

# 環境変数にSUMOツールを追加
ENV PATH="$SUMO_HOME/bin:$PATH"
ENV PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# 作業ディレクトリ
WORKDIR /app

# 必要なファイルをコピー
COPY . /app

# Pythonパッケージのインストール（requirements.txtが必要）
RUN pip3 install --no-cache-dir -r requirements.txt
