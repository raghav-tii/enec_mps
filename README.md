# enec_mps

Main repository for all ENEC-project related files, including a copy of `tensnet` library.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

---

## Introduction

This repository contains the latest version of the code related to the ENTC-TII-ENEC collaboration. This files contains all the instructions needed to launch simulations using the `tensnet` library developed at QRC, TII.

The `scripts/` folder contains scripts for various high level tasks such as mask generation, time evolution, and post-processing and visualization.

The `tensnet/` module is the core python package that consists of data structures and algorithms to manipulate the MPSs.

## Requirements

List the requirements needed to run the module. It can also be found in the file `requirements.txt`.

```bash
Python >= 3.10
Required Libraries:
- numpy
- scipy
- torch
- tntorch
- opt_einsum
- einops
- tqdm
- matplotlib
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/raghav-tii/enec_mps.git
    cd enec_mps
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    # For Linux/MacOS
    python3 -m venv <name_of_venv>
    source <name_of_venv>/bin/activate

    # For Windows
    python -m venv <name_of_venv>
    <name_of_venv>\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure necessary directories exist to save results:

    The contents of `data/` folder are untracked by git by construction and will be used to store all the results, temporary files and auxillary files such as MPS masks and simulation parameters.

    ```bash
    cd data/
    mkdir results results/temp mps_masks
    ```

## Usage

Provide a step-by-step guide on how to use the module.
