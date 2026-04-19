# Use Python base image (you can specify the version you need)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Copy everything in your current folder into the container's /app folder
COPY . /app

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    wget \
    python3.11 \
    python3.11-venv \
    python3-pip \
    apt-utils \
    ca-certificates \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gnupg2 ca-certificates apt-utils software-properties-common && \
    apt-key adv --fetch-keys https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB && \
    echo "deb [arch=amd64] https://repo.arrayfire.com/debian all main" | tee /etc/apt/sources.list.d/arrayfire.list && \
    apt-get update && \
    apt-get install -y arrayfire-cpu3-dev arrayfire-cpu3-openblas
# Set Python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip 

# 3) Create and pre-install pip packages in venv
RUN python3.11 -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip

# install from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"


# Download and install ArrayFire (CUDA) from .sh script
WORKDIR /tmp
RUN mkdir -p /opt/arrayfire && \
    wget https://arrayfire.gateway.scarf.sh/linux/3.9.0/ArrayFire.sh && \
    bash ArrayFire.sh --skip-license --prefix=/opt/arrayfire && \
    rm ArrayFire.sh

# Set environment variables so Rust can find ArrayFire
ENV AF_PATH=/opt/arrayfire
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib
ENV PATH=$PATH:$AF_PATH/bin

# Optional: build Rust code with maturin early
# RUN maturin develop

# Set the default command to run your model (adjust if it's different)
#CMD ["python3", "/app/Rust_Code/example.py"]
