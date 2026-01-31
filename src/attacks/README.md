# ToolFlood Attack

ToolFlood is a **candidate-capture attack** on tool-using agents. It generates adversarial tools that are retrieved instead of benign tools when the victim agent answers user queries.

- **Phase 1:** Generate tool candidates from sampled target queries (parallelized).
- **Phase 2:** Greedily select tools that maximize query coverage (embedding distance threshold).

---

## Install

From the project root:

```bash
# With pip
pip install -r requirements.txt

# Or with uv
uv sync
```
---

## Configuration

### 1. Main config (`config/config.yaml`)

**ToolFlood attack** (`toolflood` section):

| Option | Description | Example |
|--------|-------------|--------|
| `num_tools_per_query` | Target tools per query (top-k) | `5` |
| `query_sample_size` | Queries per Phase 1 sample | `20` |
| `num_tools_per_sample` | Tools generated per sample before filtering | `10` |
| `max_generation_iterations` | Phase 1 iteration limit | `20` |
| `max_embedding_distance` | Max cosine distance query↔tool (candidate filter) | `0.3` |
| `total_tool_budget` | Cap total attacker tools (`null` = no cap) | `null` |
| `max_concurrent_tasks` | Parallel tasks in Phase 1 | `20` |
| `embedding_model` | Embedding model name (see `models.yaml`) | `"text-embedding-3-small"` |
| `llm_optimizer_model` | LLM used to generate tool descriptions | `"gpt-4o-mini"` |
| `attacker_tools_output_path` | Where to write attacker tools (optional) | `"./output/attacker_tools.json"` |

**Experiment** (`experiment` section):

| Option | Description |
|--------|-------------|
| `benign_data_directory` | Dir with `tasks/` and `tools.json` (e.g. `./data/ToolE`) |
| `output_directory` | Results, merged tools, vectorstores (e.g. `./outputs/toolflood/toole`) |
| `max_train_queries` | Max train queries for attack generation |
| `max_test_queries` | Max test queries for evaluation |
| `victim_models` | Victim LLM names to evaluate |
| `attack_embedding_models` | Embedding model(s) for attack phase |
| `victim_embedding_models` | Embedding model(s) for victim retrieval |
| `task_names` | Task names under `tasks/` (or omit for all tasks) |
| `hard_reset` | If `true`, ignore previous results and start fresh |

**Agent** (`agent` section): `top_k` is the number of tools retrieved per query (e.g. `5`).

### 2. Models (`config/models.yaml`)

Define LLMs and embeddings used by config names:

```yaml
models:
  gpt-4o-mini:
    model: "gpt-4o-mini"
    model_provider: "openai"
    api_key: "your-api-key"
    temperature: 0.0

embeddings:
  text-embedding-3-small:
    model: "text-embedding-3-small"
    provider: "openai"
    api_key: "your-api-key"
```

Set `OPENAI_API_KEY` (or equivalent) in the environment, or put keys in this file.

### 3. Data layout

`benign_data_directory` must contain:

- **`tools.json`** — benign tools (name → description or list of tool objects).
- **`tasks/`** — per-task JSON files with train/test queries (e.g. `task_1_space_images.json`).

Task files are loaded by `load_queries_from_tasks`; task names in `experiment.task_names` should match your task naming (e.g. scenario names derived from filenames or content).

---

## Running

From the **project root**:

```bash
python -m src.experiments.run_toolflood --config config/config.yaml --models config/models.yaml
```

- **`--config`** — path to main YAML (default: `config/config.yaml`).
- **`--models`** — path to models YAML (default: `config/models.yaml`).

The script will:

1. For each task (or all tasks), generate attacker tools with the attack embedding model(s).
2. Merge attacker + benign tools and build a vector store per (attack_emb, victim_emb) pair.
3. Evaluate each victim model on train and test queries.
4. Append results and write them incrementally.

Results are written under `experiment.output_directory`:

- **`results_full.json`** — full per-run results.
- **`results_table.csv`** — summary table (ASR, TDR, mean domination, etc.).
- **`<task>/attack_emb_<name>_victim_emb_<name>/`** — `merged_tools.json`, `attack_tools.json`, `vectorstore/` for that combination.

To rerun from scratch, set `experiment.hard_reset: true` in `config/config.yaml`.

---

## Metrics

- **ASR (Attack Success Rate):** fraction of queries where the selected tool was an attacker tool.
- **TDR (Top-k Domination Rate):** fraction of queries where attacker tools dominate the top-k.

These are computed for both train and test splits and reported in `results_table.csv`.
