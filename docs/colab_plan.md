# Colab Plan

## Do We Train Locally Or On Colab?

Use this machine for code editing and smoke tests. Use Colab or another GPU host for the real 7B experiments.

Local machine:

- Good for syntax checks.
- Good for tiny model smoke runs.
- Not suitable for LLaMA-2-7B CP-MoE training unless you have a large NVIDIA GPU.

Colab/GPU host:

- Required for LLaMA-2-7B reproduction.
- Store checkpoints in Google Drive or a mounted volume.
- Let Hugging Face cache model files outside the repo.

## Model Weight Handling

Do not commit model weights. Do not copy model weights into the repo.

Hugging Face will download models into a cache directory:

- Colab default: `/root/.cache/huggingface`
- Windows default: `C:\Users\<you>\.cache\huggingface`

For LLaMA-2-7B:

1. Request model access on Hugging Face.
2. Create a Hugging Face token.
3. In Colab, run:

```bash
pip install huggingface_hub
huggingface-cli login
```

or:

```python
import os
os.environ["HF_TOKEN"] = "..."
```

## First Run Order

1. `configs/smoke_tiny.yaml` locally.
2. `configs/smoke_tiny.yaml` on Colab GPU.
3. A 1-task SuperNI mini config.
4. Full SuperNI Order 1.
5. Full SuperNI Order 2.

## Expected Compute

The paper reports SuperNI CP-MoE at roughly 114 minutes per epoch on H200-class hardware. Colab will likely be slower.
Start small and only scale after the smoke path works.

