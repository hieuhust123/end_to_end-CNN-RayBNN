# RayBNN_Python
Python Bindings for Rust RayBNN

This branch is used for testing of end-to-end model forward pass

```
## Activate venv
cd RayBNN_Python/
source ~/venvs/torch113/bin/activate
cd ./Rust_Code/
cargo clean
RUSTFLAGS="-Awarnings" maturin develop
cd ../Python_Code
python3 ./test_forward_cnn+raybnn.py

```




