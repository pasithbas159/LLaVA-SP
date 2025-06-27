## The official pytorch implement of "LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs"

The implementation changes of LLaVA-SP are in **llava_arch.py, clip_encoder.py, llava_trainer.py and train.py**.


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

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)

## Citation

If you find LLaVA-SP useful for your research and applications, please cite using this BibTeX:
```bibtex

```
