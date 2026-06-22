# Step 01: Reproduction Strategy

## Answer: Where Do We Load And Train The Model?

We do not manually download model files into this repo.

The code calls:

```python
AutoModelForCausalLM.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
```

Hugging Face downloads and caches the model automatically outside the repo.

For real CP-MoE reproduction, train on a GPU service:

- Colab Pro/Pro+ for initial experiments.
- RunPod/Lambda/AWS/university GPU for full runs.
- Local machine only if you have a strong NVIDIA GPU with enough VRAM.

## Why Not Start Directly With LLaMA-2-7B?

Because a 7B CP-MoE continual run is expensive and slow. First we need to prove:

- model wrapping works,
- MoE-LoRA modules train,
- checkpoints save,
- task loop works,
- evaluation after each task works,
- metrics are logged.

That is what the tiny smoke config is for.

## Milestone A: Local Or Colab Smoke Test

Run:

```bash
pip install -e ".[dev]"
python scripts/train_continual.py --config configs/smoke_tiny.yaml
```

Expected result:

- the tiny model loads,
- two toy tasks train,
- `outputs/smoke_tiny/metrics.json` appears,
- checkpoints appear after each task.

This does not validate CP-MoE paper performance. It validates the code path.

## Milestone B: Add Exact Transient Expert

Replace the temporary dummy importance accumulation with:

1. transient LoRA expert initialization,
2. warm-up on `warmup_tokens`,
3. path-integral importance computation,
4. CKA between transient and stable experts,
5. CP bias injection,
6. CKA-weighted importance accumulation.

Only after this do we call it a CP-MoE reproduction.

## Milestone C: SuperNI Mini

Before full Order 1, run only one or two SuperNI tasks:

- verify JSON loading,
- verify prompts,
- verify ROUGE/accuracy evaluation,
- verify memory use.

## Milestone D: SuperNI Order 1

Use:

```bash
python scripts/train_continual.py --config configs/superni_cpmoe_order1.yaml
```

Expected paper target:

```text
AP: about 50.84
Forgetting: about 0.62
Zero-shot transfer: about 35.80
```

Acceptable first reproduction:

```text
Within 1-2 points: good.
Within 3 points: usable but investigate.
Worse than 3 points: debug before improving.
```

## Colab Skeleton

```bash
git clone <your-repo-or-uploaded-folder> cp-moe-research
cd cp-moe-research
pip install -U pip
pip install -e ".[dev]"
huggingface-cli login
python scripts/train_continual.py --config configs/smoke_tiny.yaml
```

For LLaMA-2:

```bash
python scripts/train_continual.py --config configs/superni_cpmoe_order1.yaml
```

## Storage Rules

Keep these out of git:

- Hugging Face cache,
- checkpoints,
- raw SuperNI data,
- tensorboard or wandb logs,
- `.pt`, `.bin`, `.safetensors` files.

The `.gitignore` is already set up for this.

