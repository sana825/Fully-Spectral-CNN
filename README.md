# Fully-Spectral-CNN

This repository contains the High-Level Synthesis (HLS) source codes and testbenches for the Spectral Convolutional Neural Network (SpCNN) accelerator proposed in the paper. To ensure full transparency and reproducibility, we share the optimized codes for the evaluated SpCNN configurations plotted in the ME and PE frontiers, making the pragma placement and optimization strategies clear and transparent. It includes the energy-optimized D3 and D4 configurations of the FD-CNN and CD-CNN architectures trained on the MNIST dataset.

## Hardware & Software Environment

* **Target FPGA Board:** Xilinx Kintex UltraScale+ (`XCKU5P-FFVA-1`)
* **HLS Tool:** AMD Vitis HLS 2025.2

## Repository Structure

```text
├── CD_CNN/
│   ├── source/
│   │   ├── inference.h             # Support header file
│   │   ├── inference.cpp           # Main HLS source file
│   │   └── dct_mnist_weights.h     # Trained model weights in C header format
│   └── tb/
│       └── tb.cpp                  # Testbench file for single-image inference using HLS Co-simulation
├── FD_CNN/
│   ├── source/
│   │   ├── inference.h             # Support header file
│   │   ├── inference.cpp           # Main HLS source file
│   │   └── model_weights_mnist.h   # Trained model weights in C header format
│   └── tb/
│       └── tb.cpp                  # Testbench file for single-image inference using HLS Co-simulation
└── MNIST_Cheaders.zip              # MNIST test dataset converted into C header files
```
*(Note: If the HiSpec-DSE results are stored in a specific folder or file—such as `HiSpec_DSE/` or a `.csv`/`.xlsx` file—be sure to add that entry to the tree above.)*

## Getting Started & Usage

Follow these steps to set up and run the HLS co-simulation for either architecture:

1. **Extract the Dataset**
   Unzip `MNIST_Cheaders.zip` to access the MNIST dataset converted into C header files.

2. **Place Dataset in Testbench Folder**
   Copy or upload the extracted dataset files into the `tb/` folder of the architecture you wish to test (`CD_CNN/tb/` and/or `FD_CNN/tb/`).

3. **Configure the Testbench**
   Open `tb/tb.cpp` and update the `#include` path of the test image header to point to your chosen MNIST image file from the extracted dataset.

4. **Create Project & Run HLS Co-Simulation**
   * Open **Vitis HLS 2025.2** and create a new project targeting the Kintex UltraScale+ **`xcku5p`** part.
   * Add the files from `source/` as design files and `tb/tb.cpp` as the testbench.
   * Run **C Synthesis** followed by **C/RTL Co-simulation** to verify inference accuracy and inspect waveforms.
   * Run **Package IP** and import IP in **Vivado Design block** for total power and WNS.
