from openai import OpenAI
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models import DeepEvalBaseLLM

HOST    = "http://127.0.0.1"
PORT    = 8000
API_KEY = "ignore"


# ── Local LLM wrapper ─────────────────────────────────────────────────

class LocalLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            base_url=f"{HOST}:{PORT}/v1",
            api_key=API_KEY,
        )
        self.model_id = self.client.models.list().data[0].id
    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="default_model",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "local-llm"


# ── Run MMLU finance benchmark ────────────────────────────────────────

benchmark = MMLU(
    tasks=[
        MMLUTask.BUSINESS_ETHICS,
        MMLUTask.HIGH_SCHOOL_MICROECONOMICS,
        MMLUTask.ECONOMETRICS,
        MMLUTask.PROFESSIONAL_ACCOUNTING,
        MMLUTask.MARKETING,
        MMLUTask.HIGH_SCHOOL_MACROECONOMICS,
    ],
    n_shots=5,
)

benchmark.evaluate(model=LocalLLM())
print(benchmark.overall_score)
