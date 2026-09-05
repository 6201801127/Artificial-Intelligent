import chromadb

""" ****Method 1: Using ChromaDB (The Easy Way)
"""


# Connect to your local database folder
client = chromadb.PersistentClient(path="./my_chroma_db")
collection = client.get_collection(name="knowledge_base")

# Perform the similarity search
results = collection.query(
    query_texts=["programming language script"],  # Your search query
    n_results=1,  # How many matches to return
)

print("Best Match Text:", results["documents"][0])
print("Distance Score:", results["distances"][0])


""" ****Method 2: Using Sentence Transformers (The Hard Way)

To see how those 384 numbers are compared without using a database, you can use Cosine Similarity. 
Cosine similarity measures the angle between two arrows in a geometric space.
Score of 1.0: The sentences mean the exact same thing.
Score of 0.0: The sentences are completely unrelated.
"""
from sentence_transformers import SentenceTransformer, util

# 1. Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define your stored database documents
stored_documents = [
    "Python is an interpreted programming language.",
<<<<<<< HEAD
    "I saw a huge python snake at the zoo yesterday.",
=======
    "I saw a huge python snake at the zoo yesterday."
]

# 4. Convert everything into 384-dimensional embeddings
doc_embeddings = model.encode(stored_documents)
query_embedding = model.encode(user_query)

print(f"Searching for matches to: '{user_query}'\n")

# 5. Mathematically compare the query vector against all document vectors
similarity_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# 6. Print the breakdown
for i, score in enumerate(similarity_scores):
    percentage = score.item() * 100
    print(f"Document: '{stored_documents[i]}'")
    print(f"➔ Semantic Similarity Match: {percentage:.2f}%\n")
