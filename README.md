# Project Name

The offcial code for paper "Multi-Normal Prototypes Learning for Weakly Supervised Anomaly Detection".

## Overview

Anomaly detection is a crucial task in various domains. Most of the existing methods assume the normal sample data clusters around a single central prototype while the real data may consist of multiple categories or subgroups. In addition, existing methods always assume all unlabeled samples are normal while some of them are inevitably being anomalies. To address these issues, we propose a novel anomaly detection framework that can efficiently work with limited labeled anomalies. Specifically, we assume the normal sample data may consist of multiple subgroups, and propose to learn multi-normal prototypes to represent them with deep embedding clustering and contrastive learning. Additionally, we propose a method to estimate the likelihood of each unlabeled sample being normal during model training, which can help to learn more efficient encoder and normal prototypes for anomaly detection. Extensive experiments on various datasets demonstrate the superior performance of our method compared to state-of-the-art methods.


## Installation and Training

To get started with Project Name, follow the steps below to set up the development environment.

```bash
# Clone the repository
git clone https://github.com/Dongzhijin/MNPWAD.git

# Navigate to the project directory
cd MNPWAD

# Install dependencies
pip install -r requirements.txt

# Start the project
bash train.sh
