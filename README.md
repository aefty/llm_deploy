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

**Mac**
- `qwen_0.5b` — mlx-community/Qwen2.5-0.5B-Instruct-4bit
- `apertus_8b` — mlx-community/Apertus-8B-Instruct-2509-bf16
- `apertus_8b_8bit` — mlx-community/Apertus-8B-Instruct-2509-8bit
- `apertus_8b_6bit` — mlx-community/Apertus-8B-Instruct-2509-6bit
- `apertus_8b_4bit` — mlx-community/Apertus-8B-Instruct-2509-4bit
- `meta_llama_8b` — mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

**Linux**
- `qwen_0.5b` — Qwen/Qwen2.5-0.5B-Instruct
- `apertus_8b` — swiss-ai/Apertus-8B-Instruct-2509
- `apertus_70b` — swiss-ai/Apertus-70B-Instruct-2509

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
For mlx_lm there is no need for `API.keys` nor `HF.token`

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
