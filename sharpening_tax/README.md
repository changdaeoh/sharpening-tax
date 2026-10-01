# pass@k evaluation and the Sharpening Tax

Contents:

- `harness.py`: the text harness that lets a base model call tools;
- `policy.py`, `run.py`, `benchmarks/`: rollout runners for BFCL v4 (`multi_turn_base`, 200
  tasks), WebShop (500 goals) and ACEBench (770 tasks);
- `metrics.py`: pass@k, pass^k, scalability and the Sharpening Tax with paired-bootstrap CIs;
- `outcomes.py`: merging rollout files and building a per-task counts table;
- `config.py`: the 14 model pairs and the decoding settings.

## Metrics

With `c_i` successes out of `n_i` rollouts on task `i` (unbiased estimators), averaged over tasks:

```
pass@k   = 1 - C(n-c, k) / C(n, k)                    coverage
pass^k   = C(c, k) / C(n, k)                          consistency
A(K)     = sum_{k=1}^{K-1} [pass@K - pass@k]          raw area scalability
S(K)     = A(K) / ((K - 1) (1 - pass@1))              calibrated scalability
Tax_X(K) = X_base(K) - X_post(K),  X in {A, S}        Sharpening Tax
```

For your own rollouts, write one JSONL row per task with `task_id` and `pass_flags` (a list of
0/1), then run

```bash
python -m sharpening_tax.metrics --base base.jsonl --rl post.jsonl --K 128
```

`--rl` is the post-trained model. The CIs are percentile intervals of a paired task bootstrap
(1,000 resamples by default; `Pair(...).tax(Ks, n_boot=..., seed=...)` changes both).

## Example outcomes

`data/outcomes_42cells.csv.gz` holds one row per task for each of the 14 model pairs, three
benchmarks and two arms (`base` for the base model with the harness, `rl` for the post-trained
model with its native chat interface), with columns `benchmark, pair, arm, model, task_id, n, c`.
Every task has `n = 128` rollouts and `c` successes.

## Collecting rollouts (GPU)

Install the rollout requirements (they list the benchmarks' runtime dependencies; do not
install the benchmarks' own pinned requirements into the vLLM environment):

```bash
pip install -r requirements-rollout.txt
```

Then check out the benchmarks:
- [gorilla/BFCL](https://github.com/ShishirPatil/gorilla) @ `6ea5797`, with
  `BFCL_ROOT=<gorilla>/berkeley-function-call-leaderboard`;
- [WebShop](https://github.com/princeton-nlp/WebShop) @ `64fa2a5` with its 1,000-product index
  (`setup.sh -d small`), with `WEBSHOP_DIR=<WebShop>`;
- [ACEBench](https://github.com/chenchen0103/ACEBench) @ `56dd66c`, with `ACEBENCH_DIR=<ACEBench>`.

One base / post-trained pair on one benchmark:

```bash
# Base model, queried through the text harness
bash scripts/serve_vllm.sh google/gemma-4-31B 0,1 8000 &
# (wait until http://127.0.0.1:8000/v1/models answers; loading takes a few minutes)
python -m sharpening_tax.run --benchmark bfcl --model google/gemma-4-31B --mode harness \
    --base-url http://127.0.0.1:8000/v1 --out runs/bfcl/gemma-4-31B.base.jsonl

# Post-trained model, queried through its native chat interface
MAX_MODEL_LEN=32768 bash scripts/serve_vllm.sh google/gemma-4-31B-it 2,3 8001 &
python -m sharpening_tax.run --benchmark bfcl --model google/gemma-4-31B-it --mode chat \
    --base-url http://127.0.0.1:8001/v1 --out runs/bfcl/gemma-4-31B.rl.jsonl

python -m sharpening_tax.metrics --base runs/bfcl/gemma-4-31B.base.jsonl --rl runs/bfcl/gemma-4-31B.rl.jsonl

# All runs -> one counts table in the format of data/outcomes_42cells.csv.gz
python -m sharpening_tax.outcomes collect runs -o my_outcomes.csv.gz
```

Runs resume where they stopped. WebShop's environment is not thread-safe, so split it over
processes with `--shard i/8` (i = 0..7, one `--out` per shard) and join the shards with
`python -m sharpening_tax.outcomes merge`.

## Notes

- **Decoding** (`config.py`, the same for base and post-trained models): temperature 0.4 / 0.7 /
  0.7, top-p 0.95, max tokens 1024 / 256 / 1200 for BFCL / WebShop / ACEBench, N = 128.
  Unsent fields come from each checkpoint's `generation_config.json`, which is not symmetric:
  `top_k=64` for both arms of every Gemma-4 pair; `top_k=20, repetition_penalty=1.05` for the
  Qwen2.5 post-trained models; `top_k=20` for Qwen3.5-35B-A3B post-trained.
- **Context length** 16384 by default (`MAX_MODEL_LEN`). Long multi-turn transcripts, e.g.
  post-trained models on BFCL, may need 32768; rollouts that outgrow the context count as
  failures. Tested with vLLM 0.19.0; gemma-4-12B needs vLLM >= 0.23.
- **Ministral-3 Instruct** checkpoints are FP8 (their bases are BF16). On GPUs without native
  FP8, check that the outputs are coherent before a long run.
- **Qwen3.5**: the base models use a closing-fence stop on BFCL, and thinking is off for every
  Qwen3.5 model queried through the chat endpoint (both applied automatically, see `config.py`).
- **ACEBench `agent_multi_step`** (20 tasks) is played with ACEBench's own agent client
  (temperature 0.001, top-p 1, 1000 tokens, its Chinese agent prompt), as upstream does.
- **WebShop** uses its synthetic goals (`human_goals=False`; the first 500 after WebShop's fixed
  shuffle). Their price caps come from an unseeded RNG unless `WEBSHOP_ENV_SEED` is set.
- **Seeds**: BFCL and ACEBench single-shot requests send `--seed` + rollout index. Extra rollout
  batches of the same tasks need a disjoint `--seed`.
- **Without the harness**: to query a base model through the chat endpoint, serve it with its
  post-trained sibling's chat template (`CHAT_TEMPLATE_FILE=...`) and run `--mode chat`.
  Post-trained models also accept `--mode harness`. `--categories` restricts ACEBench to a
  subset, e.g. the 13 single-shot categories in `SINGLE_SHOT` (`benchmarks/acebench.py`).
