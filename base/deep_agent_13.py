# -*- coding: utf-8 -*-
"""
DeepAgents 中断审批机制示例
核心功能：测试模型调用限制的中间件
"""
import os

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver  # 内存检查点，用于保存中断状态
from langgraph.types import Command  # 恢复执行的指令类型
from dotenv import load_dotenv, find_dotenv

# 加载环境变量（DASHSCOPE_API_KEY等），优先查找当前目录的.env文件
load_dotenv(find_dotenv())

# 初始化大模型（通义千问）
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),          # 模型名称（从环境变量读取）
    model_provider="openai"                   # 兼容OpenAI格式的接口
)

# 删除表工具
@tool
def delete_database(table_name: str):
    """
    高危动作工具，删除传入的表！
    :param table_name: 要删除的表名
    :return: 操作的返回结果
    """
    print(f"调用了删除了delete_database工具。删除了{table_name}表！！")
    return f"删除了表{table_name}！"

# 删除文件工具
@tool
def delete_file(file_name: str):
    """
    高危动作工具，删除传入的文件！
    :param file_name: 要删除的文件名
    :return: 操作的返回结果
    """
    print(f"调用了删除了delete_file工具。删除了{file_name}文件！！")
    return f"删除了文件{file_name}！"

# 查询表数据工具
@tool
def select_database(table_name: str):
    """
    查询动作工具，查询传入的表数据！
    :param table_name: 要查询的表名
    :return: 查询结果
    """
    print(f"调用了select_database工具。查询了{table_name}表数据！！")
    return f"查询了表{table_name}的数据！"


# 创建deepagent，同时给高危工具设置人机交互！ 拦截动作！
# 1. 必备内容 短期记忆 + 线程id
checkpointer = InMemorySaver()
thread_config = {
    "configurable" :{"thread_id":"erdaye"}
}

# 2.常见deepagent，设置高危工具需要拦截处理
# 使用langchain的中间件
# deepagent和子智能体都可以配置中间件！
# deepagent -> create_deep_agent( -> middleware=[]
# subagent -> dict -> middleware:[]
main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,  # 必须的记录过程
    system_prompt="回答使用中文，调用对应的工具实现对应的功能！",
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1, # 同一个线程id的总调用次数限制
            run_limit=1, # 一次执行会话内的总调用次数限制
            exit_behavior="error", # 达到条件的行为限制  end -> 结束   | error -> 抛出异常 [error - try]
        )
    ],
    interrupt_on={
        "delete_database": False, # 通过 编辑 或者拒绝 ,
        "delete_file": False, # 通过 编辑 或者拒绝 ,  {"allowed_decisions": ["approve", "reject"]}
        "select_database": False
    }
)

# 3. 预执行，本次不会真正的执行！ 所有都不会执行，判定本次执行链中是否存在 人机交互节点！ 存在需要设置后 二次执行！
result_1 = main_agent.invoke({
    "messages":[
        {"role":"user","content":"先查询product表的数据！再删除user表，最后，删除zhaoweifeng.txt文件"}
    ]
},config=thread_config) # 记录人 继续工作

print(f"最终结果{result_1['messages'][-1].content}")



