from google import genai

GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="Refund requests are allowed within 7 days"
)

# print(result)
print(result.embeddings)
vector = result.embeddings[0].values
print(f"Vector : {vector}")