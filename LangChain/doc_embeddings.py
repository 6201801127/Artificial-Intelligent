import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "My_2nd_RAG_Project"

llm = ChatOllama(
    # model="gemini-3.6-flash",
    model="gemma3:1b",
    # google_api_key="",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)

# docx_loader = Docx2txtLoader("artificial-intelligent/LangChain/docs/apex_logistics_profile.docx")
# docx_documents = docx_loader.load()
# print(f"Number of docx documents: {len(docx_documents)}")
# split_docx_documents = text_splitter.split_documents(docx_documents)
# print(f"Number of split docx documents: {len(split_docx_documents)}")
# print(split_docx_documents[0])

# print(split_docx_documents[1])  # Print the first 500 characters of the first split document
# Print the first 500 characters of the first split document


# FUnction to load a documents from folder and split them into chunks
def load_documents(folder_path: str) -> list[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types
        loaded_documents = loader.load()
        documents.extend(loaded_documents)
    return documents


folder_path = "artificial-intelligent/LangChain/docs"
documents = load_documents(folder_path)
# print(f"Number of documents loaded: {len(documents)}")
# print(f"Number of documents loaded: {documents}")

# Split documents into chunks
split_documents = text_splitter.split_documents(documents)
# print(f"Number of split documents: {len(split_documents)} chunks")

# Createembedding object
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embedding the documents
document_embeddings = embeddings.embed_documents([doc.page_content for doc in split_documents])
print(f"Created embeddings:for {len(document_embeddings)} document chunks")
# print(f"Embedding for first document chunk: {document_embeddings[0]}")


# sentence transformer embedding
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
documents_embeddings = embedding_function.embed_documents(
    [doc.page_content for doc in split_documents]
)
# print(f"Embedding for first document chunk: {documents_embeddings[0]}")

# ChromaDB vectore store
# embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# Initialize a persistent local database
# client = chromadb.PersistentClient(path="./my_chroma_db")

# # Create or fetch a unique collection
# collection = client.get_or_create_collection(name="knowledge_base")


# documents = [doc.page_content for doc in split_documents]
# collection.add(
#     documents=documents,
#     ids=[f"doc_{i}" for i in range(len(documents))]
# )


# results = collection.query(
#     query_texts=["When was apex logistics founded?"],
#     n_results=1,
# )

# print(results["documents"])

# Create and persist chroma vectore store
collection_name = "Company_Documents"
vectorstore = Chroma.from_documents(
    split_documents,
    embedding=embedding_function,
    collection_name=collection_name,
    persist_directory="./chroma_db",
)
print("vectorstore:", vectorstore)


# 5. perform similarity search
query = "When was apex logistics founded?"
search_result = vectorstore.similarity_search(query, k=2)
print("Search result :", search_result)

# 6.

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever.invoke("When was apex logistics founded?")


# Chatpromp template
template = """Answere the question based only on the following 
Context: {context}
Question: {question}
Answere: 
"""
prompt = ChatPromptTemplate.from_template(template)

# Runnable pass through
# from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough

rag_chain = {"Context": retriever, "question": RunnablePassthrough()} | prompt

rag_chain.invoke({"When was apex logistics founded?"})


def doc2str(docs):
    return "\n".join([doc.page_content for doc in docs])


rag_chain = {"Context": retriever, "question": RunnablePassthrough()} | prompt

rag_chain.invoke({"When was apex logistics founded?"})

rag_chain = (
    {"context": retriever | doc2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | strOutputParser()
)

question = "When was apex logistics founded?"
response = rag_chain.invoke(question)
print(response)
