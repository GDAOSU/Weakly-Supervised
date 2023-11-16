
# Weakly-Supervised Land-Cover Classification

## Overview
Welcome to the repository for our paper, "Weakly Supervised Land-Cover Classification of High-Resolution Images with Low-Resolution Labels through Optimized Label Refinement." In this project, we introduce a novel approach to enhance semantic segmentation models using optimized label refinement. Our method effectively transforms low-resolution (LR) noisy labels into refined high-resolution (HR) labels, significantly improving the accuracy of land-cover classification.

### Key Features:
- **Double Filtering of LR Labels**: We filter out noise in LR labels during both label selection and assignment stages.
- **Graph Cut Method**: Utilizes an energy function minimization task to select correct LR labels.
- **Label Refinement**: Incorporates Forest and Water indices and a Random Forest (RF) classifier to refine labels.
- **Improved Accuracy**: Achieves 2-14% higher average accuracy (AA) on DFC2020 datasets compared to models trained on original LR labels.

## Installation

### Prerequisites
- Python 3.8.18
- PyTorch 2.0.1
- GDAL (for Linux users, use 'tifffile' instead)

### Installation Commands
```bash
conda install torch==2.0.1
```
```
# For Windows
conda install gdal
# For other OS
conda install tifffile
```

## Repository Structure

```
Weakly-Supervised/
│
├── training/                   # Folder to store training data
├── validation/                 # Folder to store validation data
│
├── Evaluation.py               # Script for evaluating model performance
├── GraphCut.py                 # Processes hyperspectral imagery and LR labels
├── RF.py                       # Trains Random Forest model
├── Refine.py                   # Combines indexing and RF for HR label prediction
├── indexStat.py                # Statistical analysis of indices
├── indexPlot.py                # Plotting tool for indices
│
└── README.md                   # You are here!
```

## Usage

### GraphCut.py
Processes HR multispectral imagery and LR labels to select relative correct labels, excluding potentially incorrect label areas.

### RF.py
Trains a Random Forest model using the labels select from GraphCut.py and HR multispectral imagery for predicting HR labels.

### Refine.py
Utilizes spectral indice and Random Forest predictions to assign a new HR label.

## Dataset
The dataset utilized in this study is DFC2020, available at [IEEE Dataport](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest).

## Contributing
We welcome contributions to improve our methods or extend the applications of our work. Please feel free to submit pull requests or open issues for discussion.

## Citation
If you find our work useful in your research, please consider citing:

```
TBD
```

## License
© 2022 The Ohio State University. All rights reserved.

END-USER LICENSE AGREEMENT

IMPORTANT! READ CAREFULLY: This End-User License Agreement (“EULA”) is a legal agreement between you (either an individual or a single entity) (“You”) and The Ohio State University (“OSU”) for the use of certain “Works” that have been made in the course of research at OSU and any associated materials provided therewith (collectively, the “Works”).

ACCEPTANCE OF THIS EULA. YOU AGREE TO BE BOUND BY THE TERMS OF THIS EULA UPON THE EARLIER OF: (A) RECEIVING, REPRODUCING, OR OTHERWISE USING ANY OF THE WORKS OR (B) ACKNOWLEDGING THAT YOU HAVE READ THIS EULA AND CLICKING BELOW TO ACCEPT THE TERMS OF THE EULA. IF YOU DO NOT AGREE TO THE TERMS OF THIS EULA, DO NOT COPY OR USE THE WORKS, AND DO NOT CLICK TO INDICATE ACCEPTANCE OF THIS EULA.

License Grant. Subject to the restrictions herein, OSU hereby grants to You a non-exclusive, worldwide license to use and reproduce the Works solely for Your internal, non-commercial purposes. OSU shall provide You copies of the Works in electronic, printable or reproducible form.

Restrictions on Reproduction and Distribution. You shall solely reproduce the Works for the limited purpose of generating sufficient copies for Your use internally and for non-commercial purposes, and shall in no event distribute copies of, transmit or display the Works to third parties by sale, rental, lease, lending, or any other means. You shall not modify, translate, adapt, merge, or make derivative works of the Works.

The Works are distributed "as-is." OSU expressly disclaims any warranties of any kind, either express or implied, including but not limited to implied warranties of merchantability, fitness for a particular purpose, or noninfringement. OSU does not assume any responsibility or liability for damages of any kind that may result, directly or indirectly, from the use of the Works.
