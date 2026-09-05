# 1. Install ChromaDB
# pip install chromadb


# 2. Initialize Client and Collection
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize a persistent local database
client = chromadb.PersistentClient(path="./my_chroma_db")

# Create or fetch a unique collection
collection = client.get_or_create_collection(name="knowledge_base")


# 3. Add Documents

collection.add(
    documents=["Python is an interpreted programming language.", "ChromaDB uses HNSW for vector indexing."],
    metadatas=[{"category": "dev"}, {"category": "database"}],
    ids=["doc1", "doc2"]
)


# 4. Query Semantically

# results = collection.query(
#     query_texts=["Tell me about database indexes"],
#     n_results=1,
#     where={"category": "database"
#     } # Metadata filtering
# )

results = collection.query(
    query_texts=["Tell me about Python"],
    n_results=1,
)

print(results["documents"])
# Output: [['Python is an interpreted programming language.']]


# Check the physical location of your database folder
absolute_path = os.path.abspath("./my_chroma_db")
print(f"Your folder is physically located here:\n{absolute_path}")

# Method 1: Check Data Programmatically (Python)

# 1. Connect to your database
# client = chromadb.PersistentClient(path="./my_chroma_db")
collection = client.get_collection(name="knowledge_base")

# 2. Fetch data including the raw mathematical vectors
data = collection.get(
    include=["embeddings", "documents", "metadatas"]
)

print("Data fetched successfully from the database.", data)
# 3. Inspect the stored data
# print("Total vectors stored:", len(data["ids"]))
# print("First document ID:", data["ids"][0])
# print("First document text:", data["documents"][0])
# print("First raw vector embedding array:", data["embeddings"])



# 1. Load ChromaDB's default embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define a list of test sentences
sentences = [
    "Python is an interpreted programming language.",
    "I saw a huge python snake at the zoo yesterday.",  # Same word, different context!
    "Java is another coding language."                  # Different words, same context!
]

print("Converting sentences to embeddings behind the scenes...\n")

# 3. Generate the mathematical vectors
embeddings = model.encode(sentences)

# 4. Inspect the results
for i, sentence in enumerate(sentences):
    vector = embeddings[i]
    print(f"Text: '{sentence}'")
    print(f"➔ Vector Data Type: {type(vector)}")
    print(f"➔ Vector Shape: {vector.shape} (Total Dimensions)")
    print(f"➔ First 3 numbers: {vector[:3]}")
    print("-" * 50)
