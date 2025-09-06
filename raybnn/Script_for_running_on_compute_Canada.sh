#!/bin/bash -l 
#SBATCH --job-name=arrayfire   # Name of job 
#SBATCH --account=def-xdong    # adjust this to match the accounting group you are using to submit jobs 
#SBATCH --time=0-5:18:00          # 2 hours 
#SBATCH --cpus-per-task=6         # CPU cores/threads 
#SBATCH --gpus-per-node=a100:1 
#SBATCH --mem=128G 
#SBATCH --mail-user=hbui@uvic.ca
#SBATCH --mail-type=ALL
 
# Load modules 
module load StdEnv/2020 gcc/9.3.0 cuda/12.2 fmt/9.1.0 spdlog/1.9.2 arrayfire/3.9.0 rust/1.70.0 python/3.11.2   
 
 
nvidia-smi 
 
 
 
rm -rf /scratch/hbui1/arrayfire/ 
mkdir /scratch/hbui1/arrayfire/ 
 
 
export AF_JIT=20 
 
export AF_JIT_KERNEL_CACHE_DIRECTORY=/scratch/hbui1/arrayfire/ 
 
export AF_OPENCL_MAX_JIT_LEN=$AF_JIT 
 
export AF_CUDA_MAX_JIT_LEN=$AF_JIT 
 
export AF_CPU_MAX_JIT_LEN=$AF_JIT 
 
export AF_DISABLE_GRAPHICS=1 
 
export RUSTFLAGS="-Awarnings -C target-cpu=native" 
 
 
 
cd /home/hbui1/Desktop/raybnn 
cargo clean 

#cargo run --example example --release
cargo run --example toy_example --release 
#cargo run --example toy_example_with_PN_dataset --release
#cargo run --example figure1b --release 
#cargo run --example figure2a --release 
#cargo run --example figure2b --release 
#cargo run --example figure2c --release 
#cargo run --example figure2d --release 
#cargo run --example figure2e --release 
#cargo run --example figure2f --release 
#cargo run --example figure3a --release 
#cargo run --example figure3b --release 
#cargo run --example figure3d --release 
#cargo run --example figure4_raybnn --release

 
#for i in {0..56}       
#do 
#    cargo run --example figure6_raybnn2   --release  $i 
#done

cargo clean 
cd /home/hbui1/Desktop/raybnn 
#zip -r  RayBNN.zip ./RayBNN/ 
 
 

