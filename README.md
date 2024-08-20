# Project Name

The offcial code for paper "Reconstruction-based Multi-Normal Prototypes Learning for Weakly Supervised Anomaly Detection".

## Overview

Anomaly detection is a crucial task in various domains. Most of the existing methods assume the normal sample data clus- ters around a single central prototype while the real data may consist of multiple categories or subgroups. In addition, ex- isting methods always assume all unlabeled data are nor- mal while they inevitably contain some anomalous samples. To address these issues, we propose a reconstruction-based multi-normal prototypes learning framework that leverages limited labeled anomalies in conjunction with abundant un- labeled data for anomaly detection. Specifically, we assume the normal sample data may satisfy multi-modal distribution, and utilize deep embedding clustering and contrastive learn- ing to learn multiple normal prototypes to represent it. Addi- tionally, we estimate the likelihood of each unlabeled sample being normal based on the multi-normal prototypes, guiding the training process to mitigate the impact of contaminated anomalies in the unlabeled data. Extensive experiments on various datasets demonstrate the superior performance of our method compared to state-of-the-art techniques.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## Installation and Training

To get started with Project Name, follow the steps below to set up the development environment.

```bash
# Clone the repository
git clone https://github.com/yourusername/yourprojectname.git

# Navigate to the project directory
cd yourprojectname

# Install dependencies
pip install -r requirements.txt

# Start the project
bash train.sh
