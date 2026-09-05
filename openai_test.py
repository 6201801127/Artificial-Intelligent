from openai import OpenAI

OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(model="gpt-4", input="Explain machine learning in one line")

print(response.output_text)
