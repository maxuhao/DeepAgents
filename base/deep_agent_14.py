# 使用 装饰器 wrap_tool_call 监控工具调用时的中间件！
# 对比提供好的中间件，我们可以自定义处理代码！！
# -*- coding: utf-8 -*-
"""
DeepAgents Middleware 极简案例
核心：实现工具调用的日志监控中间件
"""
import os
import time

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())


# ======================== 1. 定义测试工具 ========================
@tool
def add_numbers(a: int, b: int):
    """计算两个数字的和"""
    time.sleep(0.5)  # 模拟耗时操作
    result = a + b
    print(f"[工具执行] {a} + {b} = {result}")
    return result


# 自定义中间件，监控工具的调用和结束
# 1. 定义一个函数即可 -》 必须有两个参数  request , handler
@wrap_tool_call
def log_tool_call(request,handler):
    """
    中间件，监控工具的调用，进行前后的日志输出！！ 工具中间件！ 自定义代码
    :param request: 调用目标工具的参数
    :param handler: 执行目标工具的执行器
    :return: 最终的返回结果
    """
    print("--------进入了工具中间件----------")
    print(f"request : {request}")
    print(f"handler : {handler}")

    # 这块代码 -》 前置增强 -》 执行目标工具之前 -》 修改参数的！ -》 args : {table_name : user -> erdaye}

    # 必须干一个事 【摇铃 执行目标工具】
    result = handler(request)

    # 这块代码 -》 后置增强 -> 修改目标工具的返回结果 -》 result
    # 英雄联盟 -》 盲僧 -》 300盘 -》 不咋地 -》 会骂队友！ -》 队友骂我 -》 我也骂队友！！
    # 我 xxx 你  -》 游戏的检查机制 -》  返回结果 我 xx 你  *******
    print("--------退出工具中间件----------")
    print(f"result:{result}")

    return result


# 2. 函数上添加装饰器 @wrap_tool_call -> 你是一个工具调用的中间件，调用所有工具都会经过你处理
#     agent  -》  中间件 《-  tool
# 3. 自定义中间件的处理逻辑 ： 日志 ， 权限 ， 访问次数设置...
# 4. 只要是中间件，不管是langchain提供好的还是使用装饰器自定义的，都需要配置到deepagent或者subagent的中间件列表中







# ======================== 3. 配置Agent并绑定Middleware ========================
# 初始化LLM
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 创建Agent，绑定中间件
deep_agent = create_deep_agent(
    model=llm,
    tools=[add_numbers],
    checkpointer=InMemorySaver(),
    middleware=[log_tool_call], # 配置好了  调用了工具 就会生效了！！
    # 绑定中间件：传入 Middleware 实例列表
    system_prompt="你是一个计算器助手，使用add_numbers工具完成加法计算，回答仅返回计算结果。"
)

# ======================== 4. 执行测试 ========================
if __name__ == "__main__":
    # 会话配置
    thread_config = {"configurable": {"thread_id": "middleware_test_1"}}

    # 调用Agent
    result = deep_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "帮我计算 100 + 200 的结果"}
            ]
        },
        config=thread_config
    )

    # 输出最终结果
    print("\n=== 最终回复 ===")
    print(result["messages"][-1].content)