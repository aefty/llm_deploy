# LLM Server

Local LLM server with an OpenAI-compatible API. Uses **mlx_lm** on Mac and **vLLM** on Linux.

---

## Setup

```bash
git clone <repo-url>
cd apertus
```

Credential files are not included. They are auto-created with instructions on first run.

- `HF.token` — HuggingFace read token (Linux only)
- `API.keys` — API keys for the server, one per line

---

## Run

**Mac**
```bash
pip install mlx_lm
./run_mac.sh <model> <port>
```

**Linux**
```bash
./run_linux.sh <model> <port>
```

Available models:

| Model | Mac | Linux |
|---|---|---|
| `qwen_0.5b` | qwen2.5:0.5b (4bit) | Qwen/Qwen2.5-0.5B-Instruct |
| `apertus_8b` | Apertus-8B-bf16 | swiss-ai/Apertus-8B-Instruct-2509 |
| `apertus_8b_4bit` | Apertus-8B-4bit | — |
| `apertus_70b` | — | swiss-ai/Apertus-70B-Instruct-2509 |
| `meta_llama_8b` | Llama-3.1-8B (4bit) | — |

---

## API

OpenAI-compatible endpoint at `http://127.0.0.1:<port>/v1`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="test_key")

response = client.chat.completions.create(
    model="default_model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```
**for mlx_lm there is no need for `API.keys` nor `HF.token`**
---

## Testing

```bash
# for mac
sh run_mac.sh qwen_0.5b 8000
python3.12 test.py                  # basic connectivity test
python3.12 test_finance_eval.py     # MMLU finance benchmark (requires pip install deepeval datasets)

# for linux
sh run_linux.sh qwen_0.5b 8000
python3.12 test.py                  # basic connectivity test
python3.12 test_finance_eval.py     # MMLU finance benchmark (requires pip install deepeval datasets)
```
