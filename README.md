# GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration

[![OpenReview](https://img.shields.io/badge/OpenReview-232VcN8tSx-3f3f3f.svg)](https://openreview.net/forum?id=232VcN8tSx&referrer=%5Bthe%20profile%20of%20Ruoxi%20Jia%5D(%2Fprofile%3Fid%3D~Ruoxi_Jia1))

Jiachen T. Wang, Tong Wu, Dawn Song, Prateek Mittal, Ruoxi Jia

## Overview

GREATS is a framework for efficient LLM training through online data selection. It adaptively selects high-quality training samples in every iteration to improve training efficiency and model performance.

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
```bash
pip install peft==0.7.1
pip install transformers==4.36.2
pip install torch
```

## Quick Start

Run experiments using:
```bash
sh warmup_train.sh Regular 4 0.05 4 world_religions
```

### Parameters

```bash
sh warmup_train.sh \
    <selection_method>  # Regular, GradNorm, MaxLoss, TracIN-AdaptiveSelect-PerBatch(GREATS)
    <batch_size>        # 4, 8, 16, 32
    <data_percentage>   # Percentage of full data to train (e.g., 0.05)
    <validation_size>   # Size of validation set
    <subject>          # Dataset subject (e.g., world_religions)
```

## Citation

```bibtex
@inproceedings{
    wang2024greats,
    title={{GREATS}: Online Selection of High-Quality Data for {LLM} Training in Every Iteration},
    author={Jiachen T. Wang and Tong Wu and Dawn Song and Prateek Mittal and Ruoxi Jia},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=232VcN8tSx}
}
```

## Acknowledgments

This project builds upon [LESS](https://github.com/princeton-nlp/LESS). We thank the authors for their valuable contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
