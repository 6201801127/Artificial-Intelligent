import numpy as np
from google import genai

GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

text1 = "Refund requests are allowed within 7 days"
text2 = "Can i get my money back after 5 days?"

result1 = client.models.embed_content(model="gemini-embedding-2", contents=text1)

result2 = client.models.embed_content(model="gemini-embedding-2", contents=text2)
# print(result)
# print(result.embeddings)
vector1 = result1.embeddings[0].values
vector2 = result2.embeddings[0].values

print(f"Vector 1: {len(vector1)}")
print(f"Vector 2: {len(vector2)}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)
