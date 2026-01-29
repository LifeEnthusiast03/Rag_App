from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
from pathlib import Path
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def load_or_create_vector_store(filepath:Path):
    
    batch_path = filepath
    
    # Create faiss_index subdirectory inside the batch directory
    persist_path = batch_path / "faiss_index"
    
    # Load PDFs from the specified directory
    loader = PyPDFDirectoryLoader(str(batch_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs = loader.load()
     
    split_docs = text_splitter.split_documents(docs)
    uuids = [str(uuid4()) for _ in range(len(split_docs))]
            
    # Create new vector store for this batch
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=split_docs, ids=uuids)
    
    # Save the vector store in the batch's faiss_index directory
    vector_store.save_local(str(persist_path))
    print(f"Vector store saved to {persist_path}")


# load_or_create_vector_store()
    # retriever = vector_store.as_retriever(search_kwargs={"k": 10})