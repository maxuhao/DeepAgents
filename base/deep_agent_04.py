#   案例4： 演示字典类型创建子智能体！！ 并且演示异步执行
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
import os
import asyncio
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv())

# 极简初始化（自动读取OPENAI环境变量）
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    temperature=0.1,  # 自定义温度（更严谨的回答）
    model_provider="openai"
)

"""
**示例**：创建一个主智能体，它拥有三个助手：
1.  **天气助手**：查询天气（固定返回“晴朗”）。
2.  **计算助手**：处理数学问题。
3.  **翻译助手**：负责中英互译。
"""

# 1. 创建一个天气助手
weather_agent = {
    "name":"weather_helper",  # 名称
    "description":"用于查询天气信息智能助手，当用户询问查询天气的时候，调用此助手完成任务！", # 描述，给main_agent看的
    "system_prompt":"你是一个天气查询助手，无论用户查询哪个城市，你统一回复：'今天天气晴朗，温度25度！'", # 系统提示词，给lm模型看的
    "tools":[]
    # "model":
}

# 2. 计算助手
math_agent = {
    "name":"math_helper",  # 名称
    "description":"用于处理数据计算问题！", # 描述，给main_agent看的
    "system_prompt":"你是一个严谨的数学助手，帮助用户回答计算算数等问题！", # 系统提示词，给lm模型看的
    "tools":[]
    # "model":
}

translate_agent = {
    "name":"translate_helper",  # 名称
    "description":"用于中英相互翻译的助手！", # 描述，给main_agent看的
    "system_prompt":"你是一个中英翻译助手，如果是中文翻译成英文，如果是英文就翻译成对应的中文！", # 系统提示词，给lm模型看的
    "tools":[]
    # "model":
}


# 4. 创建主智能体
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[weather_agent, math_agent, translate_agent],
    system_prompt="你是一个全能管家，你会根据用户的需求，调用对应子智能体和助手实现对应的功能！注意：不能自己执行，必须调用子智能体！"
)


async def test_steam(query):
    """
    使用mainagent执行传入的问题
    :param query:
    :return:
    """
    stream = main_agent.astream({
        "messages":[
            {"role":"user", "content":query}
        ]
    })

    async  for chunk in stream:
        # chunk -> {"model / tools " : {"messages":[{},{},{}]}}
        # model   |  {messages : []}
        for node_name , state in chunk.items():
            # 如果state是None,或者state没有messages我们就跳过！！
            if state is None or "messages" not in state: continue
            # 获取messages数据
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                # 决定如何处理  node_name = model 1. 大模型决定调用工具 2. 大模型决定调用子agent 3.大模型返回结果了
                # || node_name = tools  调用自己的工具，并获取返回结果
                if node_name == "model":
                    # model = 》 返回的结果 =》 决定调用哪些
                    if last_msg.tool_calls:
                        # 决定调用子工具或者subAgent
                        for tool_call in last_msg.tool_calls:
                            if tool_call['name'] == 'task':
                                # 决定调用某个subAgent
                                print(f"【model】决定调用子智能体{tool_call['args']['subagent_type']}")
                            else:
                                # 决定调用某个工具
                                print(f"【model】决定调用子工具{tool_call['name']},传入的参数为：{tool_call['args']}")
                    elif last_msg.content:
                        # 模型返回最终结果
                        print(f"【model】返回最终结果：{last_msg.content}")
                elif node_name == "tools":
                    # agent = > 调用自己的工具了，并获取了结果
                    name = last_msg.name
                    content = last_msg.content
                    print(f"【agent】调用了具体的工具{name},返回结果为：{content[:100]+'...'}")


# test_steam("北京今天的天气怎么样？")
# # test_steam("998+889 运算后等于多少？")
# #test_steam("请将'我要上楼打他'翻译成英文！并且查询今天北京的天气信息！")
# test_steam("请将'我要上楼打他'翻译成英文！")


if __name__ == "__main__":
    # asyncio.run(test_steam("北京今天的天气怎么样？"))
    async def batch_run():
        # 要执行的并发协程对象获取到
        tast1 = test_steam("北京今天的天气怎么样？")
        tast2 = test_steam("请将'我要上楼打他'翻译成英文！")
        print(type(tast1))
        print(type(tast2))
        await asyncio.gather(tast1,tast2)

    asyncio.run(batch_run())

"""
    创建deepagent -> 配置工具和子智能体以及同步执行（invoke / stream）
    如何异步执行！！ astream -》 async for
"""