# 🌿 LEAF — Learning Engine for Autonomous Frameworks

**An agentic ML framework that uses tree search to iteratively generate, evaluate, and improve ML solutions for Kaggle-style competitions.**

LEAF implements a Monte Carlo Tree Search (MCTS) over code solutions, where each node represents a candidate ML pipeline. The agent autonomously drafts, debugs, improves, explodes (FWA-inspired), and merges solutions — guided by LLM-powered code generation and evaluation.

## ✨ Key Features

- **🌳 Tree Search over Code**: MCTS with draft/improve/debug/explode/merge operators
- **🧠 Three-Layer Memory**: Persistent strategy library, stage summaries, and raw experiment buffer
- **🔥 Fireworks Algorithm (FWA)**: Amplitude-adaptive explosion for escaping local optima
- **🔍 Hybrid Retrieval**: BM25 + FAISS semantic search for finding similar past experiments
- **🤖 Multi-Backend LLM Support**: OpenAI, Anthropic, Gemini, Qwen, and any OpenAI-compatible API
- **📊 MLEBench Integration**: Optional submission validation against official MLEBench grading rules
- **🛡️ Duplicate Detection**: LLM-based semantic deduplication prevents redundant exploration
- **⚡ Parallel Execution**: Multi-process code execution with CPU pinning and GPU isolation

## 📁 Project Structure

```
leaf/
├── run.py                 # Entry point: python -m leaf.run
├── agent.py               # Core agent: search policy, draft/improve/debug/explode/merge
├── interpreter.py         # Subprocess-based Python code executor
├── journal.py             # Solution tree data structure (nodes, metrics, MCTS)
├── grade_server.py        # Optional MLEBench submission validation server
├── backend/               # LLM API backends
│   ├── __init__.py        # Provider routing (auto-detects OpenAI/Anthropic/Gemini/Qwen)
│   ├── backend_openai.py  # OpenAI Responses API
│   ├── backend_anthropic.py # Anthropic Claude API
│   ├── backend_gemini.py  # Google Gemini API (via OpenAI-compatible interface)
│   ├── backend_qwen.py    # Qwen API (OpenAI-compatible)
│   ├── backend_utils.py   # Shared utilities (retry, prompt compilation, FunctionSpec)
│   └── call.py            # Alternative LLM calling interface
├── memory/                # Memory and retrieval systems
│   ├── buffer.py          # Three-layer memory manager (L1→L2→L3)
│   ├── global_memo.py     # Global deduplication with semantic similarity
│   └── retriever.py       # BM25 + FAISS hybrid retriever
├── prompt/                # Prompt templates
│   ├── impl_guideline.py  # Implementation guidelines for code generation
│   ├── scot.md            # Structured chain-of-thought patch guide
│   └── validation_template_prompts.py
├── utils/                 # Utilities
│   ├── config.py          # Configuration loading (OmegaConf + CLI args)
│   ├── config.yaml        # Default configuration
│   ├── evaluator.py       # LLM-based code evaluation and metric extraction
│   ├── skill.py           # Skill guidance provider (user-configurable)
│   ├── skill_conventional.py  # Advanced skill evolution with classification
│   ├── data_preview.py    # Automatic data preview generation
│   ├── response.py        # Code extraction and formatting
│   ├── metric.py          # Metric value comparison
│   ├── serialize.py       # JSON serialization for journals
│   ├── bm25_faiss.py      # Node retrieval via hybrid search
│   └── llm_caller.py      # Low-level LLM API wrapper
├── kaggle_prompt/         # Domain-specific ML guidance (per task type)
├── kaggle_prompt_conventional/  # Conventional ML guidance
├── eva_prompt/            # Per-competition evaluation prompts
└── webui/                 # Optional Streamlit web interface
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd leaf

# Install dependencies
pip install -r requirements.txt

# For MLEBench integration (optional):
pip install flask mlebench

# For Web UI (optional):
pip install streamlit python-dotenv
```

### 2. Configure API Keys

Set your LLM API key via environment variable or CLI argument:

```bash
# Option A: Environment variable
export LEAF_API_KEY="sk-your-api-key"
export LEAF_BASE_URL="https://api.openai.com/v1"  # or any OpenAI-compatible URL

# Option B: CLI argument (passed to OmegaConf)
python -m leaf.run agent.api_key=sk-xxx agent.base_url=https://...
```

For multiple API keys (rotation pool), use comma-separated values:
```bash
export LEAF_API_KEYS="sk-key1,sk-key2,sk-key3"
```

For Anthropic:
```bash
export ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2"
```

### 3. Prepare Your Data

Organize your competition data in MLEBench format:
```
data/
└── your-competition/
    └── prepared/
        └── public/
            ├── description.md    # Task description
            ├── train.csv         # Training data
            ├── test.csv          # Test data
            └── sample_submission.csv
```

### 4. Run the Agent

```bash
# Basic run
python -m leaf.run \
    data_dir=./data/your-competition/prepared/public \
    desc_file=./data/your-competition/prepared/public/description.md \
    exp_name=my_experiment

# With skill file (recommended: provide your current best solution to accelerate search)
python -m leaf.run \
    data_dir=./data/your-competition/prepared/public \
    desc_file=./data/your-competition/prepared/public/description.md \
    skill_file=./skills/my_best_solution.md \
    exp_name=my_experiment

# With eval prompt (recommended: provide task-specific evaluation rules)
python -m leaf.run \
    data_dir=./data/your-competition/prepared/public \
    desc_file=./data/your-competition/prepared/public/description.md \
    mle_data_dir=./eval_prompts \
    exp_name=my_experiment

# Override any config parameter via CLI
python -m leaf.run \
    data_dir=./data/... \
    agent.steps=100 \
    agent.draft.model=gpt-4 \
    agent.feedback.model=gpt-4 \
    exec.timeout=7200

# Save logs to file (while still printing to screen)
mkdir -p logs
python -m leaf.run \
    data_dir=./data/your-competition/prepared/public \
    exp_name=my_experiment \
    2>&1 | tee logs/my_experiment.log

# Save logs to file only (no screen output, run in background)
nohup python -m leaf.run \
    data_dir=./data/your-competition/prepared/public \
    exp_name=my_experiment \
    > logs/my_experiment.log 2>&1 &
```

### 5. MLEBench Grade Server (Optional)

If you want submission format validation against MLEBench:

```bash
# Start the grade server
python -m leaf.grade_server --data_dir /path/to/mle-bench/data --port 5005

# Enable in config.yaml or via CLI:
python -m leaf.run \
    data_dir=... \
    use_grade_server=true \
    grade_server_port=5005 \
    mle_data_dir=/path/to/mle-bench/data
```

> **📊 Scoring MLEBench Submissions**
>
> LEAF generates `submission.csv` files in the workspace. To score these against MLEBench ground truth (e.g., for benchmarking or medal threshold comparison), please refer to the [MLEBench documentation](https://github.com/openai/mle-bench) for the official grading pipeline and scoring scripts.

### 6. Batch Scheduling (Multi-GPU)

For running multiple competitions in parallel:

```bash
# Edit scheduler_leaf.sh with your tasks and GPU configuration
bash scheduler_leaf.sh
```

## ⚙️ Configuration

All configuration is in `utils/config.yaml` and can be overridden via CLI arguments (OmegaConf).

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `null` | Path to competition data directory |
| `desc_file` | `null` | Path to task description markdown file |
| `exp_name` | `null` | Experiment name (auto-generated if null) |
| `log_dir` | `./logs/run` | Output directory for logs and results |
| `workspace_dir` | `./workspaces/run` | Agent working directory |
| `use_grade_server` | `false` | Enable MLEBench format validation |
| `skill_file` | `null` | Path to skill markdown file (see [Writing Custom Skills](#-writing-custom-skills)) |
| `mle_data_dir` | `null` | Path to eval prompt directory (see [Eval Prompts](#-eval-prompts)) |
| `agent.steps` | `400` | Maximum search iterations |
| `agent.draft.model` | `gpt-5.5` | LLM model for code generation |
| `agent.feedback.model` | `gpt-5.5` | LLM model for evaluation |
| `agent.cheap.model` | `qwen3.6-max-preview` | Lightweight model for auxiliary tasks |
| `agent.search.num_drafts` | `3` | Number of initial draft solutions |
| `agent.search.max_debug_depth` | `20` | Maximum consecutive debug attempts |
| `exec.timeout` | `32400` | Code execution timeout (seconds) |

### LLM Backend Selection

The backend is auto-detected from the model name:
- `gpt-*` → OpenAI
- `gemini-*` → Gemini
- `claude-*` or `us.anthropic.*` → Anthropic
- Everything else → Qwen (OpenAI-compatible)

## 📝 Writing Custom Skills

Skills are Markdown files that provide domain-specific guidance to the agent. **By passing a skill file containing your current best solution or known-effective approaches, the agent can start evolving from a strong baseline instead of exploring from zero-shot**, significantly accelerating the search process.

A skill file can include:
- **Known-effective solutions**: Paste your current best code or approach so the agent uses it as a starting point
- **Domain insights**: Data characteristics, feature engineering tricks specific to this task
- **Proven strategies**: Model architectures, ensemble methods, or hyperparameter ranges that work well
- **Pitfalls to avoid**: Things you've tried that didn't work

```markdown
# Skill: Leaf Classification

## Current Best Approach (Score: 0.985)
- Ensemble of LightGBM + XGBoost + MLP with weighted averaging
- Feature engineering: group statistics (mean/std/skew/kurtosis) on margin/shape/texture
- PCA components (n=10) per feature group
- 5-fold StratifiedKFold CV, optimize ensemble weights via Nelder-Mead

## Key Observations
- 192 pre-extracted features (64 margin + 64 shape + 64 texture)
- StandardScaler is critical for the neural network branch
- Cross-group dot products (margin×shape, etc.) add marginal gain

## What Didn't Work
- Pure CNN on raw leaf images (overfits with limited data)
- Single model without ensemble (3-5% worse log_loss)
```

Pass it via config:
```bash
python -m leaf.run skill_file=./skills/leaf_classification.md ...
```

> **💡 Custom Pretrained / Deep Learning Models**
>
> This repository is the **general-purpose base version** of LEAF. If you want the agent to use specific pretrained models (e.g., domain-specific HuggingFace models) or specialized deep learning architectures, you need to:
>
> 1. **Edit `agent._prompt_environment`** in `agent.py` — update the installed packages list and add usage hints for your models so the LLM knows they are available.
> 2. **Install the required dependencies** — add the corresponding packages to `requirements.txt` or install them in your environment.
>
> The agent can only use models and libraries it knows about through the prompt. Without explicit guidance, it will default to common packages (PyTorch, transformers, timm, etc.).

## 📋 Eval Prompts

Eval prompts are per-competition markdown files that provide **task-specific evaluation rules** to the agent's evaluator. They tell the LLM reviewer exactly what metric to look for, how to interpret it, and what constitutes a bug for this particular competition.

**When `mle_data_dir` is not set (default)**: The evaluator relies on the LLM's own knowledge to assess code quality and extract metrics. This works for most standard competitions.

**When `mle_data_dir` is set**: The evaluator loads `<mle_data_dir>/<exp_name>.md` and injects the content into the evaluation prompt. This enables precise, domain-aware evaluation (e.g., "the metric is Pearson correlation, higher is better, watch for NaN predictions").

### Directory Structure

```
eval_prompts/
├── leaf-classification.md      # loaded when exp_name=leaf-classification
├── housing-prices.md
└── time-series-forecasting.md
```

### Example Eval Prompt (`leaf-classification.md`)

```markdown
## Evaluation Rules
- **Metric**: Multi-class Log Loss (lower is better)
- **Expected output**: The code must print a single float representing the validation log_loss
- **Submission format**: CSV with `id` column + 99 species probability columns (must sum to 1.0 per row)

## Common Bugs to Watch For
1. Probabilities not clipped to [1e-15, 1-1e-15] → log_loss explodes to infinity
2. Missing species columns in submission → format validation fails
3. LabelEncoder mismatch between train and test → wrong column ordering
```

### Usage

```bash
python -m leaf.run \
    data_dir=./data/leaf-classification \
    exp_name=leaf-classification \
    mle_data_dir=./eval_prompts
```

## 📊 Output Structure

After a run, results are saved to `log_dir/<exp_name>/`:

```
logs/run/my_experiment/
├── journal.json              # Full solution tree with all nodes
├── config.yaml               # Configuration snapshot
└── best_submission/
    ├── submission01.csv       # Best submission
    ├── submission02.csv       # 2nd best
    ├── submission03.csv       # 3rd best
    ├── best_solution01.py     # Best solution code
    ├── best_solution02.py
    └── best_solution03.py
```

## 🔧 Advanced Usage

### Forced Explosion (Stagnation Breaker)

When the search stagnates, LEAF triggers high-temperature explosions on the best UCT node:

```yaml
agent:
  search:
    forced_explode:
      enabled: true
      patience: 20          # Steps before triggering
      temperature: 1.2      # Higher = more diverse
      num_sparks: 5         # Parallel solutions to generate
      cooldown: 15          # Minimum steps between triggers
```

### Three-Layer Memory System

LEAF maintains a hierarchical memory:
- **L1 (Raw Buffer)**: Recent experiment records (configurable limit)
- **L2 (Stage Summaries)**: LLM-compressed experience summaries
- **L3 (Refined Strategies)**: Distilled positive/negative patterns

### Embedding Model

For semantic search, LEAF uses `thenlper/gte-small`. Configure the local path:

```bash
# Via environment variable
export LEAF_FAISS_DIR=/path/to/gte-small

# Via config
python -m leaf.run faiss_dir=/path/to/gte-small ...
```

If the local model is not found, it falls back to downloading from HuggingFace.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                    Agent (MCTS)                  │
│  ┌───────┐ ┌─────────┐ ┌───────┐ ┌──────────┐ │
│  │ Draft │ │ Improve │ │ Debug │ │  Explode  │ │
│  └───┬───┘ └────┬────┘ └───┬───┘ └─────┬─────┘ │
│      └──────────┴──────────┴───────────┘        │
│                    Journal                       │
│  ┌──────────────────────────────────────────┐   │
│  │  Solution Tree (Nodes + Metrics + MCTS)  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Memory    │  │ Retriever│  │   Evaluator  │ │
│  │  (L1/L2/L3)│  │(BM25+FAISS)│ │  (LLM-based) │ │
│  └────────────┘  └──────────┘  └─────────────┘ │
└─────────────────────────────────────────────────┘
         │                          │
    ┌────▼─────┐             ┌──────▼──────┐
    │Interpreter│             │  Backend     │
    │(subprocess)│            │(OpenAI/etc.) │
    └──────────┘             └──────────────┘
```

## 📄 License

This project is based on [AIDE](https://github.com/WecoAI/aideml) by WecoAI, extended with MCTS-based search, three-layer memory, fireworks algorithm, and MLEBench integration.

## 🙏 Acknowledgments

- [AIDE ML](https://github.com/WecoAI/aideml) — Original agentic ML framework
- [MLEBench](https://github.com/openai/mle-bench) — ML engineering benchmark by OpenAI
- [Fireworks Algorithm](https://en.wikipedia.org/wiki/Fireworks_algorithm) — Inspiration for the explosion operator
