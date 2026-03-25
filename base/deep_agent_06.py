# 实验6： 兼容langgraph子智能体格式
import os
from dataclasses import dataclass

from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import add_messages, StateGraph, END

# 加载环境变量
load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)


# 定义一个user对象
@dataclass
class User(TypedDict):
    name: str
    age: int


# 定一个一个langgraph图节点！！ 只写一个节点！！
# 1. 定一个state状态对象
class SubState(TypedDict):
    # 注意： graph图节点，想要兼容deepagents,state中必须包含一个messages属性，并且是集合类型，累加结果！！
    messages: Annotated[list, add_messages]
    users1: Annotated[User, ]
    users2: list[User]
    name: str


# 2. 定义图节点和编译图结构
def processing_node(state: SubState):
    # 自己的langgraph的节点进行处理
    print(f"调用了graph的子节点，传入的参数为：{state}")
    print("子节点的业务逻辑.....")
    return {"messages": [AIMessage(content=f"经过子节点处理后的结果！！原数据内容：{state['messages'][-1].content}")]}


workflow = StateGraph(SubState)
workflow.add_node("worker", processing_node)
workflow.set_entry_point("worker")
workflow.add_edge("worker", END)
compile_graph = workflow.compile()

# 3. 包装成一个deepAgent认识的subAgent
sub_agent = CompiledSubAgent(
    name="graph_agent",
    description="处理所有的业务逻辑！！",
    runnable=compile_graph
)

# 4. 创建主智能体，设置子智能体
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[sub_agent],
    system_prompt="你是一个指挥官，所有的业务动作，都需要使用graph_agent进行处理！"
)

for chunk in main_agent.stream({"input": "处理一段复杂业务，并核对id=1用户的数据是什么？"}):
    print(chunk)
