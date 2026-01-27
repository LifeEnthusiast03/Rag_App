from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from dotenv import load_dotenv
import os
import time

load_dotenv()

PERSIST_PATH = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
loader = PyPDFDirectoryLoader('./example')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=300)
docs = loader.load()

split_docs = text_splitter.split_documents(docs)
print(len(split_docs))
uuids = [str(uuid4()) for _ in range(len(split_docs))]

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

try:
    print(f"Starting to add {len(split_docs)} documents to vector store...")
    vector_store.add_documents(documents=split_docs, ids=uuids)
    print("Documents added successfully!")
    
    print(f"Saving vector store to {PERSIST_PATH}...")
    vector_store.save_local(PERSIST_PATH)
    print("Vector store saved successfully!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

