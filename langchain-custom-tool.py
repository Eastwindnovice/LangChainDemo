import os
from typing import Literal

from dotenv import load_dotenv
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_deepseek import ChatDeepSeek


# 读取 .env 文件
load_dotenv()

# 初始化 Tavily
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# 初始化 DeepSeek 模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
)


agent = create_deep_agent(
    model=model,
    tools=[internet_search],
)

result = agent.invoke(
    {
        "messages" : [
            {
                "role": "user",
                "content": "请搜索一下最近 AI Agent 领域有什么重要新闻，并总结三点。",
            }
        ]
    },
    config={"configurable": {"thread_id": "1"}},
)

print(result)
print(result["messages"][-1].content)
