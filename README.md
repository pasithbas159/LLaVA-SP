## The official pytorch implement of "LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs" [[Paper](https://arxiv.org/abs/2507.00505)]

The implementation changes of LLaVA-SP are in **llava_arch.py, clip_encoder.py, llava_trainer.py** and **train.py**.


## Install

Please see instructions for https://github.com/haotian-liu/LLaVA/



## LLaVA-SP Weights
Please check out https://huggingface.co/Levideus/models for all public LLaVA-SP checkpoints.

## Quick Start  
```
python llava/eval/run_llava.py
--model_path /path/llava-sp-cropping-lora
--model_base /path/vicuna-1.5-7b
```

```
python llava/eval/run_llava.py --model-path Levideus/llava-sp-pooling-lora --model-base lmsys/vicuna-7b-v1.5 --image-file "https://llava-vl.github.io/static/images/view.jpg" --query "What are the things I should be cautious about when I visit here?"
```

## Citation

If you find LLaVA-SP useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{lou2025llavasp,
    title={LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs},
    author={Lou, Haoran and Fan, Chunxiao and Liu, Ziyan Liu and Wu, Yuexin Wu and Wang, Xinliang},
    publisher={arXiv:2507.00505},
    year={2025}
}
```
https://stackoverflow.com/questions/79273647/cannot-import-name-encoderdecodercache-from-transformers