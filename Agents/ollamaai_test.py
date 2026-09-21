import ollama

response = ollama.generate(
    model="gemma3:1b", prompt="Explain machine learning in one line", stream=True
)

print(response["response"])
