#!/bin/bash -l
#SBATCH --job-name=test_raybnn_python   # Name of job
#SBATCH --account=def-xdong    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-00:14          # 2 hours
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G
#SBATCH --mail-user=hbui@uvic.ca
#SBATCH --mail-type=ALL

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 arrayfire/3.9.0 rust/1.85.0 python/3.11.5 openblas scipy-stack/2024b

module load numpy/2.1.1 maturin patchelf

nvidia-smi
DIR="/home/hbui1/Desktop/RayBNN/RayBNN_Python/"
cd "$DIR"

if [ ! -d "$DIR/venv" ]; then
         echo "venv not found, create new one"
python3 -m venv --system-site-packages venv # This is essential for venv to be able to load modules from StdEnv
else
         echo "venv exist!"
 fi
source "$DIR/venv/bin/activate"
rm -rf "$DIR/Rust_Code/target/"

cd "$DIR/Rust_Code"

maturin develop
cd "$DIR/Python_Code"
#python3 example.py
python3 end_to_end_model.py
