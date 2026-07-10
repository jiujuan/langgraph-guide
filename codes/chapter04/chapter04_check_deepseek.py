from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
)

response = llm.invoke("用一句话解释 LangGraph 是什么。")

print(response.content)
