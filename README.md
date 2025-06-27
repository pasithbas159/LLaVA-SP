# [ICCV 2025] The official pytorch implement of "LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs".

## The implementation changes of LLaVA-SP are in llava/model/llava_arch.py and llava/model/multimodal_encoder/clip_encoder.py.


## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```


## LLaVA-SP Weights
Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)(:to do) for all public LLaVA checkpoints, and the instructions of how to use the weights.


## Citation

If you find LLaVA-SP useful for your research and applications, please cite using this BibTeX:
```bibtex

```
