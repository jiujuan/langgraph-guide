from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

response = llm.invoke("用一句话解释 LangGraph 是什么。")

print(response.content)
