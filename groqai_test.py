from groq import Groq

client = Groq(api_key="")

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain Deep learning in one line"
        }
    ]
)

print(response.choices[0].message.content)