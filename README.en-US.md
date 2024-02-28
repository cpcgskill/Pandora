# Pandora

Pandora is an independently developed LLM project, currently in its early stages, primarily for personal research and exploration. This project utilizes Rotating Position Encoding (RoPE), the GPT architecture, and the SkipGram word embedding model, employing a context length of 512 during training. Although Pandora is not yet fully mature, it offers an intriguing opportunity for interested developers and researchers to explore.

- [English](./README.en-US.md)
- [中文](./README.md)

## Table of Contents

- [Quick Start](#quick-start)
    - [Clone the Project](#clone-the-project)
    - [Install Dependencies](#install-dependencies)
    - [Train the Model](#train-the-model)
- [Current Performance](#current-performance)
    - [Pandora Loss Graph](#pandora-loss-graph)
    - [SkipGram Loss Graph](#skipgram-loss-graph)
- [License](#license)

## Quick Start

### Clone the Project

```bash
git clone git@github.com:cpcgskill/Pandora.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

For Python 3.10 users, due to compatibility issues with `multiprocess` in the `datasets` dependency, a specific version of the dependencies file is required:

```bash
pip install -r requirements_3_10.txt
```

### Train the Model

- Train the Tokenizer:

```bash
python train_tokenizer.py
```

- Train the SkipGram model:

```bash
python train_SkipGram.py
```

- Train the Pandora model:

```bash
python train_Pandora.py
```

## Current Performance

Due to limitations in model scale, Pandora currently has deficiencies in its knowledge reserve and may inaccurately or fail to answer some questions that are beyond its knowledge scope.

### Pandora Loss Graph

![Pandora](./images/pandora_loss.png)

### SkipGram Loss Graph

![SkipGram](./images/skip_gram_loss.png)

## License

This project is licensed under the Apache-2.0 license. For detailed information about the license, please refer to the LICENSE file in the project.