"""
  案例1： 体验deepagents的创建和配置对应的工具！ 同时学习deepagent执行和结果解析！！
  步骤1： 准备网络搜索工具
  步骤2： 初始化千文模型对象
  步骤3： 创建深度智能体并且配置网络搜索工具
  步骤4： 非流式执行深度智能体，进行结果解析（分析下，深度智能体的执行原理）
"""
from typing import Literal

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv, find_dotenv
import os

# 确保 .env配置加到参数中，才可以进行读取
load_dotenv(find_dotenv())

llm_name = os.getenv("LLM_QWEN_MAX")
tavily_key = os.getenv("tavily_api_key")

# 步骤1 准备网络搜索工具
# 初始化tavily的网络工具客户端，可以进行网络搜索 每天1000次免费机会
tavily_client = TavilyClient(api_key=tavily_key)


@tool
def internet_search(
        query:str,  # 搜索关键字和内容
        max_results:int = 5,  # 最大结果数量
        topic: Literal["news", "finance", "general"] = "general",
        include_raw_content:bool = False,  # 是否精简搜索 False 精简搜索 true 搜索原文详细结果
):
    """
    互联网搜索工具！ 用于网络信息搜索
    :param query: 搜索关键字
    :param max_results: 返回的条数
    :param topic: 查询新闻类型
    :param include_raw_content: 是否精简 false 精简 true 详细
    :return: 查询结果
    """
    print(f"开始网络搜索工具调用！核心参数为：{query},{max_results},{topic},{include_raw_content}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )


# 步骤2： 初始化千文模型对象
llm = init_chat_model(
    model=llm_name,
    model_provider="openai"
)

# 我们的目标：创建深度智能体！ 需要模型 工具 （子代理）
# from deepagents import create_deep_agent
#   步骤3： 创建深度智能体并且配置网络搜索工具
deep_agent = create_deep_agent(
    model=llm, #智能体都是需要模型
    tools=[internet_search],#配置工具tool
    subagents=[], #配置子代理
    system_prompt="""
    你是一个专家级的研究员！你有权使用工具：internet_search网络信息！
    最终，需要你根据收集工具，生成一份精美的报告！
    """
)

# 4. 执行deepagent
# 非流式执行，不体现中间的过程！直接等结果！！
result = deep_agent.invoke(
    {
        "messages":[
            {"role":"user","content":"查询人工智能和机器人的热门新闻信息！"}
        ]
    }
)

# 输出过程
print(result)

# 获取结果
print(result['messages'][-1].content)
