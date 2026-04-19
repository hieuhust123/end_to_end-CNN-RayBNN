#!/bin/bash -el


source /home/hbui/Downloads/RayBNN_Python/.venv/bin/activate

# pairs: "install_name:import_name"
PACKAGES=(
    "maturin:maturin"
    "numpy<2:numpy"
    "patchelf:patchelf"
    "matplotlib:matplotlib"
    "scikit-learn:sklearn"
    "pandas:pandas"
    "torch:torch"
    "psutil:psutil"
    "keras:keras"
    "tensorflow==2.18.0:tensorflow"
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

RAYBNN_DIR="/home/hbui/Downloads/RayBNN_Python/Rust_Code"
if ! python -c "import raybnn_python" &> /dev/null; then
    echo "Building raybnn_python..."
    cd "$RAYBNN_DIR"
    maturin develop --release
    cd -
else
    echo "raybnn_python already installed. Skipping build."
fi

SCRIPT_DIR="/home/hbui/Downloads/Xuan_Chen_code/c/CNN+RayBNN"

python "$SCRIPT_DIR/train_test.py"
python "$SCRIPT_DIR/train_feature_part1.py"
python "$SCRIPT_DIR/train_feature_part2.py"
python "$SCRIPT_DIR/train_feature_part3.py"
python "$SCRIPT_DIR/train_feature_part4.py"
python "$SCRIPT_DIR/run_network_RayBNN_SLEEP_copy.py"
