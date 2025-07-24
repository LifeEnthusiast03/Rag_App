from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
import os

load_dotenv()

PERSIST_PATH = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_or_create_vector_store():
    if os.path.exists(PERSIST_PATH):
        vector_store = FAISS.load_local(PERSIST_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFDirectoryLoader('./example')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        uuids = [str(uuid4()) for _ in range(len(split_docs))]

        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(documents=split_docs, ids=uuids)
        vector_store.save_local(PERSIST_PATH)
    return vector_store

def get_retriever():
    vector_store = load_or_create_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return retriever
