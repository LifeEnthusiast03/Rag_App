from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)
loader1 = PyPDFDirectoryLoader('./example')
text_spiter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs1 = loader1.load()
split_docs  = text_spiter.split_documents(docs1)
uuids = [str(uuid4()) for _ in range(len(split_docs))]

vector_store.add_documents(documents=split_docs, ids=uuids)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

result = retriever.invoke('what is a array')

