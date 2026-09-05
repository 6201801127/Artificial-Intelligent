import numpy as np
from google import genai

GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

documents = [
    "Refund requests are allowed within 7 days.",
    "Python classes happen Monday to Friday.",
    "Student get 3 interviews opportunities.",
    "Cource access available for 6 months.",
]

# question = "Can i get my money back after 5 days?"
question = "What is the average placement package?"


# 1. Create document embeddings

document_vectors = []

for document in documents:
    result = client.models.embed_content(model="gemini-embedding-2", contents=documents)
    vector = result.embeddings[0].values
    document_vectors.append(vector)

# 2 Create Question embedding
question_result = client.models.embed_content(model="gemini-embedding-2", contents=question)
question_vector = question_result.embeddings[0].values

# 3. Compare similarity

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


scores = []

for vector in document_vectors:
    score = cosine_similarity(question_vector, vector)
    scores.append(score)

# 4. Retrive best chunk

best_index = np.argmax(score)
best_document = documents[best_index]

print("Retrieved Context: ", best_document)

# 5. Send retrieved chunk to LLM

prompt = f"""
Answere the question using only the context below.
Context:
{best_document}

Question:
{question}"""


# 6. generate final answere

response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)

print("Final Answere: ", response.text)
