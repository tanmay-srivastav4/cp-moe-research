# CP-MoE Research Reproduction

This repo is for reproducing CP-MoE first, then adding router-aware and adaptive probing improvements.

## Strategy

Do not download large model weights into this repo. Hugging Face and Colab should cache them outside the project:

- Local smoke tests: use a tiny causal LM to verify code paths.
- Real reproduction: run on Colab Pro/Pro+, Kaggle GPU, RunPod, Lambda, or a university GPU.
- LLaMA-2-7B requires Hugging Face access approval and an `HF_TOKEN`.

## Stages

1. Smoke test with a small model and 2-3 tiny tasks.
2. Reproduce CP-MoE on SuperNI Order 1.
3. Add paper ablations.
4. Add router-drift diagnostics.
5. Implement router-aware CP-MoE improvements.

## Environment

Install dependencies:

```bash
pip install -e ".[dev]"
```

For GPU runs, install a PyTorch build matching your CUDA version before installing this package.

## Model Loading

The training scripts use `transformers.from_pretrained(...)`. This means:

- On local CPU, use a tiny open model for smoke testing.
- On Colab, mount Drive for checkpoints and let Hugging Face cache model weights in `/root/.cache/huggingface`.
- For gated LLaMA models, run `huggingface-cli login` or set `HF_TOKEN`.

Example:

```bash
export HF_TOKEN=...
python scripts/train_continual.py --config configs/smoke_tiny.yaml
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="..."
python scripts/train_continual.py --config configs/smoke_tiny.yaml
```

## Current Status

This is the first implementation scaffold. The first target is a local smoke run, not CP-MoE paper numbers.

