# Intelliton

Intelliton is a research project that investigates whether transformer neural networks support
stable, quasi-particle-like collective excitation modes — **Intellitons** — inside their residual
streams.

## GitHub Pages Blog (bilingual)

The project includes a **bilingual popular science blog** hosted on GitHub Pages:

**<https://xiongzhp.github.io/Intelliton/>**

The blog is written in both **English and Chinese** (中英双语). Use the language toggle in the
top-right corner to switch. Your preferred language is remembered locally in the browser.

### Blog articles

| # | English title | 中文标题 |
|---|---|---|
| 1 | What Are Intellitons? | 什么是 Intelliton？ |
| 2 | Inside Qwen3-4B-Base | 走进 Qwen3-4B-Base |
| 3 | Scaling and Alignment Through the Intelliton Lens | 用 Intelliton 视角看规模扩展与对齐 |
| 4 | Hallucination as Internal Instability | 把幻觉理解为内部不稳定性 |

The blog source lives in `docs/` and is powered by Jekyll with a custom bilingual layout.

---

## Research code

This `src/` directory contains the paper-oriented experiment subset for the Intelliton project.

## Scope

The code here is organized around the main paper narrative:

1. **Discovery**
   - residual-stream spectral analysis
   - propagator mass extraction
   - lattice dispersion relation
2. **Characterization**
   - EFT / RG flow
   - Intelliton species classification
3. **Applications**
   - hallucination diagnostics
   - generation-time trajectory analysis

## Main entrypoint

Run all five paper models in sequence:

```bash
python run_paper.py --all-models
```

Run a single model:

```bash
python run_paper.py --model-path /data/users/xiongzhaoping/Intel2/Qwen3-4B-Base
```

### Supported models

| Model | Type | Path |
|-------|------|------|
| Qwen3-4B-Base | base | `/data/users/xiongzhaoping/Intel2/Qwen3-4B-Base` |
| Qwen3-8B-Base | base | `/data/users/xiongzhaoping/Intel2/Qwen3-8B-Base` |
| Mistral-7B-v0.3 | base | `/data/users/xiongzhaoping/Intel2/Mistral-7B-v0.3` |
| Qwen3-4B | instruct | `/data/users/xiongzhaoping/Intel2/Qwen3-4B` |
| Qwen3-8B | instruct | `/data/users/xiongzhaoping/Intel2/Qwen3-8B` |

## Important files

- `run_paper.py` — standalone paper runner (multi-model)
- `paper_pipeline.py` — orchestration of the paper experiment subset
- `config.py` — model-agnostic config (auto-configured from config.json)
- `datasets.py` — prompt sets used by the paper path
- `lattice_field.py` — spectral and quasiparticle observables
- `eft_renormalization.py` — RG/EFT analysis
- `intelliton_classifier.py` — Intelliton catalog construction
- `hallucination_diagnostic.py` — grounded vs hallucination-prone diagnostics
- `intelliton_dynamics.py` — generation-time dynamics tracking
- `visualization.py` — paper figures
- `intelliton_analyzer.py` — local analyzer used by the paper path

## Outputs

Typical outputs include:

- `intelliton_catalog.csv`
- `hallucination_diagnostics.png`
- `intelliton_trajectory_detail.csv`
- `intelliton_trajectory_summary.csv`
- `intelliton_trajectory.png`
- spectrum / RG / EFT / summary figures

## Dependency note

The goal of `src/` is to make the **paper experiment path** self-contained and independent from the root-level project modules.

## Suggested workflow

1. Activate the Python environment
2. Run `run_paper.py`
3. Inspect generated CSV and figure outputs
4. Use those outputs to assemble paper figures and tables
