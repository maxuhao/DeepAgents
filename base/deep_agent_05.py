# 实验5： deepagents 现在版本不支持子智能体创建子智能体
import os
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# ---------------------------------------------------------
# 强制层级委派的关键配置
# ---------------------------------------------------------

# 1. 底层 Coder 配置
# 职责明确：只有他能写代码
coder_config = {
    "name": "Coder",
    "description": "高级Python工程师，他是唯一有权限编写具体代码的人。",
    "system_prompt": "你是一个高级Python工程师。你的职责是接收具体的编码任务并实现它。",
    "tools": [],  # Coder 拥有默认的文件操作工具
}

# 2. 中间层 CTO 配置
# 职责明确：承上启下，必须指挥 Coder
cto_config = {
    "name": "CTO",
    "description": "技术总监，负责将战略需求转化为技术任务并分配给工程师。指挥Coder写代码的！",
    # 关键修改：明确告诉 CTO 不要自己写代码，必须找 Coder
    "system_prompt": """你是技术总监。
    注意：你没有编写代码的权限！
    你的职责是：
    1. 分析 CEO 的需求。
    2. 设计技术方案。
    3. 调用 'Coder' 子代理来完成具体的代码编写工作。
    """,
    "tools": [],
    "subagents": [coder_config]  #没有这个属性，硬写的！ dict 底层不识别
}

# 3. 顶层 CEO 配置
# 职责明确：只负责战略，禁止干具体的活
ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    # 关键修改：明确告诉 CEO 不要自己动手，必须找 CTO
    system_prompt="""你是CEO，负责公司战略决策。
    注意：你严禁直接编写代码或操作文件！
    你必须将所有技术相关的开发任务委派给 'CTO' 处理。
    你的工作是验收 CTO 提交的结果。
    """,
    subagents=[cto_config]
)

# 运行 CEO 代理
print(">>> 开始执行任务链...")
stream = ceo_agent.stream({
    "messages": [
        {"role": "user", "content": "帮我开发一个贪吃蛇游戏，要求用Python实现,直接提供代码的字符串即可！！"}
    ]
})

# 打印最终结果
print("\n>>> 最终结果：")
for chunk in stream:
    print(chunk)