import dspy
from pydantic import BaseModel, Field

HOST        = "http://127.0.0.1"
PORT        = 8000
API_KEY     = "test_key"

# -- local (mlx) --
lm = dspy.LM(
    model="openai/default_model",
    api_base=f"{HOST}:{PORT}/v1",
    api_key=API_KEY,
    max_tokens=2048,
)

dspy.configure(lm=lm)

class QA(dspy.Signature):
    """You will answer the question using the provided context."""
    question: str = dspy.InputField(desc="Question asked by the user.")
    context:  str = dspy.InputField(desc="Relevant context to inform the answer.")
    answer:   str = dspy.OutputField(desc="Your direct with relevant dates, concise answer in 100 characters or less.")

class Shorten(dspy.Signature):
    """Shorten the answer to be 100 characters or less. Keep the core meaning."""
    answer:  str = dspy.InputField(desc="Answer that is too long.")
    shorter: str = dspy.OutputField(desc="Shortened version with relevant dates, max 100 characters.")

context = "Your creator is Aryan, you are born in 2025."

qa      = dspy.ChainOfThought(QA)
shorten = dspy.Predict(Shorten)

print("Ask a question (ctrl+c to exit)\n")
while True:
    try:
        question = input(">  ").strip()
        if not question:
            continue

        result = qa(question=question, context=context)
        answer = result.answer

        # retry loop — ask AI to shorten if too long
        attempts = 0
        while len(answer) > 100 and attempts < 3:
            print(f"~ Answer too long ({len(answer)} chars), shortening...\n")
            answer = shorten(answer=answer).shorter
            attempts += 1

        # hard truncate as last resort
        if len(answer) > 100:
            answer = answer[:100]

        print(f"~ Reasoning: {result.reasoning}\n")
        print(f"# Answer: {answer}\n")

    except KeyboardInterrupt:
        print("\nbye")
        break