# LLM Inference example on Azure using CycleCloud + Slurm

This repo shows how to setup LLM inference example workload on Azure's CycleCloud + Slurm. This repo assumes that you already setup CycleCloud cluster with Slurm and this repo shows how to build a LLM inference workload on top of the Slurm compute node (worker node). I used Text Generation Inference (TGI) (https://github.com/huggingface/text-generation-inference) from HuggingFace as an LLM inference example. As this example is NOT VERIFIED on CycleCloud instance, you might need to manually modify some codes in tgi_conda_setup.sh.

<a ><img src="assets/cyclecloud_llm.jpg" width="100%"></a>

The image is modified from (https://techcommunity.microsoft.com/t5/azure-high-performance-computing/enabling-job-accounting-for-slurm-with-azure-cyclecloud-8-2-and/ba-p/3413803)

## Prerequisites

1. To setup CycleCLoud + Slurm, refer to the official Azure's documentation and tutorials. 
https://learn.microsoft.com/en-us/azure/cyclecloud/qs-install-marketplace?view=cyclecloud-8

2. This repo is tested with ROCm5.7 and please install ROCm runtime/libraries to the compute node (worker node).
https://rocm.docs.amd.com/en/docs-5.7.0/deploy/linux/installer/install.html#

## LLM Inference deployment on the Slurm compute node

### Option A: use Nvidia's Pyxis plug-in to use LLM docker image. 

After installation of Pyxis (https://github.com/NVIDIA/pyxis), launch a prebuilt ROCm-enabled TGI LLM Inference from HuggingFace. 

```bash
srun --container-image=ghcr.io/huggingface/text-generation-inference:1.4-rocm
```

### Option B: manual installation of LLM application on CycleCloud compute node.

First, install anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

Run the followings to setup TGI application from source code

```bash
source ~/.bashrc
conda init
conda create -n tgi python=3.9 
conda activate tgi

cd $HOME
git clone https://github.com/huggingface/text-generation-inference.git 
cd text-generation-inference/

export PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
cargo build --release

sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    curl \
    git \
    make \
    libssl-dev \
    g++ \
    rocthrust-dev \
    hipsparse-dev \
    hipblas-dev && \
    sudo rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/test/rocm5.7
python -m pip install einops pandas ray pyarrow

export PYTORCH_ROCM_ARCH="gfx90a"
export HUGGINGFACE_HUB_CACHE=./
export HF_HUB_ENABLE_HF_TRANSFER=1 
export PORT=9090
export PATH=$PATH:$HOME/text-generation-inference/target/release

mkdir build_kernels
cd build_kernels 

cp ../server/Makefile-vllm Makefile
sed -i 's/pip /python -m pip /g' Makefile
make install-vllm-rocm

cp ../server/Makefile-flash-att-v2 Makefile
sed -i 's/pip /python -m pip /g' Makefile
make install-flash-attention-v2-rocm

cp -rf ../server/custom_kernels/ .
cd custom_kernels/
python setup.py install
cd ..

cp -rf ../server/exllamav2_kernels/ .
cd exllamav2_kernels/
python setup.py install
cd ..

cd ../server
rm -rf text_generation_server/pb
sed -i 's/pip /python -m pip /g' Makefile
make gen-server

python -m pip install -r requirements_rocm.txt
python -m pip install ".[accelerate, peft]"
cd ..

./target/release/text-generation-launcher --model-id TheBloke/Llama-2-7B-Chat-fp16 &
```

To test the TGI in the compute node (worker node)

```bash
curl 127.0.0.1:9090/generate X POST d '{"inputs":"Chocolate is good for","parameters":{"max_new_tokens":100}}' H 'Content-Type: application
```

And the corresponding response from TGI server will be like this

```bash
2024-02-06T03:08:33.664018Z  INFO generate{parameters=GenerateParameters { best_of: None, temperature: None, repetition_penalty: None, top_k: None, top_p: None, typical_p: None, do_sample: false, max_new_tokens: Some(100), return_full_text: None, stop: [], truncate: None, watermark: false, details: false, decoder_input_details: false, seed: None, top_n_tokens: None } total_time="2.520979735s" validation_time="308.477µs" queue_time="70µs" inference_time="2.520601569s" time_per_token="25.206015ms" seed="None"}: text_generation_router::server: router/src/server.rs:298: Success
{"generated_text":" your heart, but not for your waistline\n\nChocolate is a tasty treat that has been shown to have several health benefits, including improving heart health. Studies have found that chocolate can help lower blood pressure, improve blood flow to the heart, and even reduce the risk of heart disease. However, it's important to remember that chocolate is high in calories and sugar, which can lead to weight gain if consumed in excess. So while"}(tgi)
```

## LLM Inference uxui lanuch

Once LLM Inference launched on the Slurm compute node, you can launch an additional uxui application so that users can access to the LLM instance. This demo is from (https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#gradio)

```bash
python tgi_uxui.py
```

<a ><img src="assets/tgi_uxui_demo.jpg" width="100%"></a>
