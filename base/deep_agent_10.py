from pathlib import Path  # 导入Path类
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend,StateBackend,StoreBackend,CompositeBackend
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# 1. 准备本地工作目录（用Path改写）
workspace_dir = Path("./agent_workspace").resolve()  # resolve() 等价于 os.path.abspath()，获取绝对路径
if not workspace_dir.exists():  # 等价于 os.path.exists()
    workspace_dir.mkdir(parents=True, exist_ok=True)  # 等价于 os.makedirs()

# 演示使用fileSystemBeckEnd实现长期记忆 （跨会话 跨线程）实现数据共享 -》 多轮会话 有之前的环境！！

# 2. 创建FileBackEnd
file_backend = FilesystemBackend(root_dir=workspace_dir,virtual_mode=True)

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 3. 创建deepAgent指定长期记忆类型
main_agen = create_deep_agent(
    model=llm,
    tools=[],
    backend=file_backend, # 指定长期记忆类型
    system_prompt="你是一个智能助手，可以使用文件工具进行文件操作和读写！但是只有在用户明确要求的情况下，你才可以创建文件！！"
)

# 4. 运行并且验证
print("1：：： 不明确，看看会不会创建")
result_1 = main_agen.invoke(
    {
        "messages": [
            {"role": "user", "content": "帮我查询下python语言的介绍！！！"}
        ]
    }
)
print(f"最终结果{result_1['messages'][-1].content}")
print("2：：： 明确，看看会不会创建")

result_2 = main_agen.invoke(
    {
        "messages": [
            {"role": "user", "content": "帮我查询下java语言的介绍,并且帮我写到 java.txt文件中！！"}
        ]
    }
)
print(f"最终结果{result_2['messages'][-1].content}")
