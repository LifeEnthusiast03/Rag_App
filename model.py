from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

res = llm.invoke('Who is the prime minister of india')
print(res.content)