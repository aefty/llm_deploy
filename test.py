from openai import OpenAI

HOST    = "http://127.0.0.1"
PORT    = 8000
API_KEY = "test_key"

client = OpenAI(
    base_url=f"{HOST}:{PORT}/v1",
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model="default_model",
    messages=[{"role": "user", "content": "who are you?"}],
)

print(response.choices[0].message.content)
