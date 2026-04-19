#!/bin/bash -l
#SBATCH --job-name=run_mwt_combine_model   # Name of job
#SBATCH --account=def-xdong    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=8:00:00          # 2 hours
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --mail-user=hbui@uvic.ca
#SBATCH --mail-type=ALL


# Load modules
module --force purge

DIR="/home/hbui1/projects/def-xdong/hbui1"
set -e

module load StdEnv/2023 gcc/12.3 cuda/12.6  arrayfire/3.10.0 rust/1.85.0 python/3.11.5 openblas arrayfire/3.10.0
#fmt/9.1.0 spdlog/1.9.2

# export AF_PATH=$DIR/RayBNN_Python/ArrayFire-3.10.0-Linux
# export LIBRARY_PATH="$DIR/RayBNN_Python/ArrayFire-3.10.0-Linux/lib64:$LIBRARY_PATH"
# export LD_LIBRARY_PATH="$DIR/RayBNN_Python/ArrayFire-3.10.0-Linux/lib64:$LD_LIBRARY_PATH"

nvidia-smi

source $DIR/RayBNN_Python/venv/bin/activate

PACKAGES=(
    "maturin:maturin"
    "numpy<2:numpy"
    "patchelf:patchelf"
    "matplotlib:matplotlib"
    "scikit-learn:sklearn"
    "pandas:pandas"
    "torch:torch"
    "psutil:psutil"
    "torchvision:torchvision"
)

MISSING_INSTALLS=()
for entry in "${PACKAGES[@]}"; do
    install_name="${entry%%:*}"
    import_name="${entry##*:}"
    if ! python -c "import $import_name" &> /dev/null; then
        MISSING_INSTALLS+=("$install_name")
    fi
done

if [ ${#MISSING_INSTALLS[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_INSTALLS[@]}"
    pip install "${MISSING_INSTALLS[@]}"
else
    echo "Build dependencies already satisfied. Skipping pip install."
fi
maturin develop --release

cd $DIR/RayBNN_Python/Python_Code

# echo $LD_LIBRARY_PATH
# ldd $DIR/RayBNN_Python/venv/lib/python3.11/site-packages/raybnn_python/raybnn_python.cpython-311-x86_64-linux-gnu.so 2>&1 | grep "not found"
#python original_run_network.py > debug_seg.txt 2>&1

python mwt_test_backward_cnn+raybnn_testing.py > debug_seg.txt 2>&1


