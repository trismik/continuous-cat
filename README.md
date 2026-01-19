# Continuous CAT for LLM Evaluation

Continuous Computerized Adaptive Testing (CAT) is a method for efficiently ranking LLMs using continuous evaluation metrics like ROUGE and BERTScore. By adaptively selecting which test items to administer to each model, it produces confident rankings while evaluating only a fraction of the full benchmark.

This repository contains the implementation and experiments for the paper ["Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores"](https://arxiv.org/abs/XXXX.XXXXX).

## Quick Start

```bash
# Create and activate conda environment
conda create -n continuous-cat python=3.11
conda activate continuous-cat

# Install dependencies
pip install -r requirements.txt

# Run all experiments (generates Tables 2-6)
python experiments/run_all.py --n-seeds 20 --output-dir experiments/results
```

## Citation

```bibtex
@article{balkir2026continuous,
  title={Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores},
  author={Balkir, Esma and Pernthaller, Alice and Basaldella, Marco and Hern{\'a}ndez-Orallo, Jos{\'e} and Collier, Nigel},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
