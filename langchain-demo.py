from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent


load_dotenv()

def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="deepseek-chat",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Xia Men?"}]
})

print(result)
print(result["messages"][-1].content)

