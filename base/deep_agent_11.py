from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, StateBackend,FilesystemBackend,CompositeBackend
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv(find_dotenv())

# 生产环境建议使用 RedisStore: from langgraph.store.redis import RedisStore



# 2. 配置 Store 后端
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 1. 准备 Store (模拟数据库)
# InMemoryStore 是轻量级内存存储，重启后数据丢失。
# 最终存储的位置 内存行的k = v 使用langgraph自带的！ 也可以切换成其他的！（redis...）
store = InMemoryStore()

# 3. 创建deepagents执行StoreBackEnd 具体的存储store = InMemoryStore()
main_agent = create_deep_agent(
    model=llm,
    store=store, # 具体的存储位置  key (user_profile.txt) = value (重要信息)
    backend=StoreBackend, # 开启了k=v 库存储
    system_prompt="""
    你要把用户的重要信息保存到user_profile.txt文件中！
    获取用户信息可以读取user_profile.txt文件！
    """
)

# 4. 演示跨会话 跨线程进行长期记忆
config_a  = {
    "configurable":{
        "thread_id":"a"
    }
}
config_b  = {
    "configurable":{
        "thread_id":"b"
    }
}

#第一遍执行 -》 存储一些重要信息
result_a = main_agent.invoke(
    {
        "messages":[
            {"role":"user","content":"我是大风哥，我今年19岁！"}
        ]
    },
    config=config_a
)

print(f"第一次回复结果：{result_a['messages'][-1].content}")

print(f"读取store保存的用户信息！！")
# store -> namespace命令空间（库） filesystem  -》 key  user_profile.txt | value  重要信息
items = store.search(('filesystem',)) # ('filesystem',) -> 元组   ('filesystem' 不写, 变成字符串)
for item in items:
    print(f"key = {item.key}")
    print(f"value = {item.value}")

#第二遍执行 -》 读取一些重要信息

result_b = main_agent.invoke(
    {
        "messages":[
            {"role":"user","content":"我叫啥和我的年龄！"}
        ]
    },
    config=config_b
)

print(f"第一次回复结果：{result_b['messages'][-1].content}")