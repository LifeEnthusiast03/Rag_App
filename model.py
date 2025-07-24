from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from rag import get_retriever

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

retriever = get_retriever()

prompt = PromptTemplate.from_template(
    "You are a very good answer giver. You are given a query: '{query}' and a context: '{context}'. "
    "You must answer the query using the context only."
)

chain1 = RunnableParallel({"query": RunnablePassthrough(), "context": retriever})

parser = StrOutputParser()

chain2 = chain1 | prompt | llm | parser

result = chain2.invoke("what are the causes of soil degradation?")

print(result)
