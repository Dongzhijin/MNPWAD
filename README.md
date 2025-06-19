# MNPWAD: Multi-Normal Prototypes Learning for Weakly Supervised Anomaly Detection

The official code implementation for the paper ["Multi-Normal Prototypes Learning for Weakly Supervised Anomaly Detection"](https://arxiv.org/abs/2408.14498).

## Overview

Anomaly detection is a crucial task in various domains. Most existing methods assume that normal sample data clusters around a single central prototype, while real data may consist of multiple categories or subgroups. Additionally, existing methods always assume all unlabeled samples are normal while some of them are inevitably anomalies. 

To address these issues, we propose a novel anomaly detection framework that can efficiently work with limited labeled anomalies. Specifically:

- **Multi-Normal Prototypes**: We assume normal sample data may consist of multiple subgroups and propose to learn multi-normal prototypes to represent them using deep embedding clustering and contrastive learning.
- **Likelihood Estimation**: We propose a method to estimate the likelihood of each unlabeled sample being normal during model training, which helps learn more efficient encoders and normal prototypes for anomaly detection.
- **Weakly Supervised Learning**: Our method works effectively with limited labeled anomalies, making it practical for real-world scenarios.

Extensive experiments on various datasets demonstrate the superior performance of our method compared to state-of-the-art methods.

## Requirements

- Python 3.7+
- PyTorch 1.13.0
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm
- fastai
- PyYAML
- ipdb

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dongzhijin/MNPWAD.git
cd MNPWAD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Place your dataset files in CSV format in the `./data` directory. The CSV files should have:
- Features in all columns except the last one
- Labels in the last column (0 for normal, 1 for anomaly)

## Usage

### Quick Start

Run the training with default parameters:
```bash
bash train.sh
```

### Custom Training

You can customize the training by modifying the parameters in `train.sh` or running `train.py` directly:

```bash
python train.py --datapath ./data \
                --datasets IoT23_CandC,IoT23_Attack,IoT23_DDoS,IoT23_Okiru \
                --trainflag origin \
                --labeled_ratio 0.01 \
                --runs 5 \
                --base_model MNP \
                --gpu 0,1 \
                --batch_size 128 \
                --lr 0.005 \
                --n_emb 8 \
                --m1 0.02 \
                --lambda_kl 1.0 \
                --debug False \
                --dataset2n_prototypes dataset2n_prototypes
```

### Parameters

- `--datapath`: Path to the data directory (default: `./data`)
- `--datasets`: Comma-separated list of dataset names
- `--labeled_ratio`: Ratio of labeled anomalies in training data (default: 0.01)
- `--runs`: Number of experimental runs (default: 5)
- `--base_model`: Base model type (default: `MNP`)
- `--gpu`: GPU devices to use (default: `0`)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.005)
- `--n_emb`: Embedding dimension (default: 8)
- `--m1`: Margin parameter for contrastive learning (default: 0.02)
- `--lambda_kl`: Weight for KL divergence loss (default: 1.0)
- `--n_prototypes`: Number of normal prototypes (default: 0, auto-determined)
- `--pretrainAE`: Whether to pretrain autoencoder (default: True)

## Model Architecture

The MNPWAD framework consists of several key components:

1. **Denoising Autoencoder**: Pre-trained layer-wise to learn robust feature representations
2. **Multi-Normal Prototypes (MNP)**: Learn multiple prototypes to represent different normal subgroups
3. **Anomaly Score Network**: Combines reconstruction error, embedding features, and prototype similarity
4. **Contrastive Learning**: Uses labeled anomalies to improve prototype learning

## Output

The model outputs:

- Training logs in `output/{trainflag}-{timestamp}/log/`
- Model checkpoints in `output/{trainflag}-{timestamp}/checkpoints/`
- Results in `output/{trainflag}-{timestamp}/middle_result/` and `all_result.csv`

Results include:

- AUC-ROC and AUC-PR scores
- Training time
- Number of prototypes used

## File Structure

```text
MNPWAD/
├── MNPWAD.py           # Main model implementation
├── train.py            # Training script
├── train.sh            # Training shell script
├── model.py            # Neural network architectures
├── utils.py            # Utility functions
├── utils_pretrain.py   # Autoencoder pretraining utilities
├── Dataset.py          # Data loading and batch generation
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{dong2024multinormalprototypeslearningweakly,
      title={Multi-Normal Prototypes Learning for Weakly Supervised Anomaly Detection}, 
      author={Zhijin Dong and Hongzhi Liu and Boyuan Ren and Weimin Xiong and Zhonghai Wu},
      year={2024},
      eprint={2408.14498},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2408.14498}, 
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
