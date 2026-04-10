# Vexoo Labs – AI Engineer Assignment

**Submitted by:** Shivaprasad Chinthoju

---

## Overview

This repository contains the solution to the Vexoo Labs AI Engineer Assignment, covering:

- **Part 1:** Document ingestion using a Sliding Window + 4-layer Knowledge Pyramid
- **Part 2:** Fine-tuning LLaMA 3.2 1B on GSM8K using LoRA-based SFT
- **Bonus:** Plug-and-play Reasoning-Aware Adapter

---

## Repository Structure

```
vexoo-labs-ai-assignment/
├── part1_ingestion.py            # Sliding Window + Knowledge Pyramid system
├── part2_gsm8k_training.py       # GSM8K LoRA fine-tuning pipeline
├── bonus_reasoning_adapter.py    # Reasoning-aware routing adapter
├── requirements.txt              # Python dependencies
├── summary_report.docx           # One-page summary report
├── pyramid_output.json           # Sample output from Part 1
└── simulation_results.json       # Sample output from Part 2 dry-run
```

---

## Part 1 – Document Ingestion + Knowledge Pyramid

### Approach

Ingests any plain-text document through a **2-page sliding window** (2,000 chars/page, 50% overlap) to produce overlapping chunks that preserve cross-boundary context. Each chunk is enriched into a **4-layer Knowledge Pyramid**:

| Layer | Name | Method |
|-------|------|--------|
| 0 | Raw Text | Verbatim window content |
| 1 | Chunk Summary | First + last sentence + word count (placeholder) |
| 2 | Category / Theme | Rule-based keyword scoring across 6 domains |
| 3 | Distilled Knowledge | Top-10 keywords + 8-dim mock embedding |

At query time, **cosine similarity** over word-frequency vectors selects the best-matching chunk and pyramid layer.

### Run

```bash
python part1_ingestion.py
```

> No external dependencies required — uses Python standard library only.

### Output

- Console: per-chunk pyramid summary + ranked query results
- `pyramid_output.json`: full pyramid structure for inspection

---

## Part 2 – GSM8K LoRA Fine-Tuning

### Approach

Fine-tunes **LLaMA 3.2 1B Instruct** on 3,000 GSM8K training samples using:
- **4-bit NF4 quantisation** (bitsandbytes) to fit on a single GPU
- **LoRA adapters** (rank 8, targets `q_proj` and `v_proj`)
- **Chain-of-thought prompt format** with `#### <answer>` suffix for exact-match evaluation

### Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # required to download LLaMA 3.2 weights
```

### Run (real training – requires CUDA GPU ≥ 16 GB VRAM)

```bash
python part2_gsm8k_training.py
```

### Dry-run (no GPU needed)

Validates data loading, formatting, and tokenisation. Simulates loss curves.

```bash
SIMULATE_TRAINING=true python part2_gsm8k_training.py
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | meta-llama/Llama-3.2-1B-Instruct |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Effective batch size | 16 (batch 4 × grad accum 4) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Train samples | 3,000 |
| Eval samples | 1,000 |
| Evaluation metric | Exact-match accuracy |

### Outputs

| File | Contents |
|------|----------|
| `gsm8k_lora_output/training_log.json` | Loss + LR per optimisation step |
| `gsm8k_lora_output/eval_results.json` | Final accuracy + hyperparameters |
| `gsm8k_lora_output/adapter_model.bin` | Saved LoRA weights |

---

## Bonus – Reasoning-Aware Adapter

### Approach

A lightweight plug-and-play routing component that:

1. **Classifies** the input query into one of 6 types using regex-based keyword scoring:
   - Math, Legal, Medical, Code, Factual, Conversational
2. **Activates** the appropriate reasoning strategy:
   - Chain-of-Thought → Math
   - Rule-Lookup + Citation → Legal
   - Evidence-Based Reasoning → Medical
   - Plan-then-Code Scratchpad → Code
   - Retrieve-then-Verify → Factual
3. **Returns** the answer alongside a step-by-step reasoning trace and confidence flag.

### Extending

New domains can be added in a single line:

```python
adapter.register(QueryType.SCIENCE, my_science_handler)
```

### Run

```bash
python bonus_reasoning_adapter.py
```

> No external dependencies required — uses Python standard library only.

---

## Requirements

Install all dependencies for Part 2:

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| datasets | ≥ 2.19.0 |
| transformers | ≥ 4.44.0 |
| peft | ≥ 0.11.0 |
| bitsandbytes | ≥ 0.43.0 |
| torch | ≥ 2.2.0 |
| accelerate | ≥ 0.30.0 |

> Part 1 and the Bonus adapter use Python standard library only — no extra packages needed.

---

## Notes

- Part 1 uses Python standard library only (no ML frameworks required).
- Part 2 requires a CUDA GPU for real training; use `SIMULATE_TRAINING=true` for a dry-run on any machine.
- The Bonus adapter uses simulated LLM responses; replace `_simulate_llm()` with a real API call (e.g. Anthropic, OpenAI) for production use.
