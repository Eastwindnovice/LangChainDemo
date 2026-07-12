# LangChain Demo From Beginner to Advanced
## QuickStart
这是一个LangChain的从入门到高级的，适合国内用户学习的仓库，这边以DeepSeek的API Key为例进行案例训练。
首先我们新建一个工作区目录，取名什么都可以，然后使用uv命令进行处理
```
uv init
```
然后新建一个langchain-demo.py文件，以及一个.env文件（文件里面放DP的API Key）然后粘贴以下内容：
```
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent


load_dotenv()

def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="deepseek-chat",
    # tools里面可以传入自定义函数，LangChain 工具或来自任何 MCP 服务器的工具
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Xia Men?"}]
})

print(result)
print(result["messages"][-1].content)
```
这段代码的整体逻辑是：先加载环境变量，让程序能够读取 DeepSeek 的 API Key；然后定义一个天气查询工具，这个工具接收城市名，并返回一段天气描述。接着创建一个 Deep Agent，也就是一个带工具能力的 AI 助手，并告诉它使用 DeepSeek 模型，同时把刚才定义的天气工具交给它。

当我们向 Agent 提问“厦门天气怎么样”时，Agent 会先理解用户的问题，判断这个问题和天气有关，于是调用我们提供的天气工具。工具返回结果后，Agent 再把这个结果整理成最终回复。

最后，程序打印两部分内容：第一部分是完整的执行结果，里面包含模型消息、工具调用记录、返回信息等，主要用于调试；第二部分只打印最后一条 AI 回复，也就是用户真正关心的最终答案。

简单来说，这段代码不是单纯地让模型聊天，而是创建了一个“会使用工具的 AI 助手”。用户问天气，助手就调用天气工具，再把工具结果返回给用户。下图出自官方文档，表示Deep Agent的核心功能。
![](docs/images/image.png)
![](docs/images/2026-07-12-16-52-27.png)
这句话表明Deep Agents 是一个把大模型包装成可靠 agent 的“工作框架”或“运行外壳”。这就是代理harness的意思。Deep Agents 这个 harness 的作用，就是给大模型加上一套结构，让它不是“自由发挥”，而是按照一个更可靠的方式工作。