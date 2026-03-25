"""
  案例1： 体验deepagents的创建和配置对应的工具！ 同时学习deepagent执行和结果解析！！
  步骤1： 准备网络搜索工具
  步骤2： 初始化千文模型对象
  步骤3： 创建深度智能体并且配置网络搜索工具
  步骤4： 非流式执行深度智能体，进行结果解析（分析下，深度智能体的执行原理）
"""
from langchain.tools import tool # 封装工具装饰器
from typing import Literal # tavily工具在使用的时候，指定查询网络信息的类型 （新闻，金融。。。）
from tavily import TavilyClient # 网络搜索工具
from dotenv import load_dotenv,find_dotenv # 加载配置文件
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
import os # 读取配置文件

# 确保 .env配置加到参数中，才可以进行读取
load_dotenv(find_dotenv())

llm_name = os.getenv("LLM_QWEN_MAX")
tavily_key = os.getenv("tavily_api_key")

# 步骤1： 准备网络搜索工具
# 初始化tavily的网络工具客户端 可以进行网络搜索搜索  每天 1000次免费机会！！
tavily_client = TavilyClient(api_key=tavily_key)

@tool
def internet_search(
        query:str,  # 搜索关键字和内容
        max_results:int = 5,  # 搜索结果数量
        topic: Literal["news", "finance", "general"] = "general", # 新闻类型
        include_raw_content:bool = False,  # 是否精简搜索
):
    """
    互联网搜索工具！ 用于网络信息搜索
    :param query: 搜索关键字
    :param max_results: 搜索结果数量
    :param topic: 新闻类型
    :param include_raw_content: 是否精简 false 精简 true 详细
    :return: 查询结果
    """
    print(f"开始网络搜索工具调用！核心参数为：{query},{max_results},{topic},{include_raw_content}")
    return tavily_client.search(
        query = query,
        max_results= max_results,
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
    model=llm,
    tools=[internet_search],
    subagents=[],
    system_prompt="""
    你是一个专家级的研究员！你有权使用工具：internet_search网络信息！
    最终，需要你根据收集工具，生成一份精美的报告！
    """
)

# 4. 执行deepagent [非流式]
# 非流式执行，不体现中间的过程！直接等结果！！
# result = deep_agent.invoke(
#     {
#         "messages":[
#             {"role":"user","content":"查询人工智能和机器人的热门新闻信息！"}
#         ]
#     }
# )

# 获取结果
# print(result['messages'][-1].content)

# 5. 流式执行
stream = deep_agent.stream({"messages": [{"role": "user", "content": "查询人工智能和机器人的热门新闻信息！"}]})

"""
  chunk 四种状态  1.是不是模型决定调用工具 2. 是不是模型决定调用subagent  3. 是不是调用工具  4. 是不是模型返回结果 
     chunk -> 字典   key 【model -> 最终返回结果 4 | 决定调用哪个工具 1 | 决定调用哪个子agent 2 | tools -> 调用了的具体的工具 3】: value (返回的值和具体的内容和方向)
{
    "model / tools " : { messages : [ .... ,
    "model / tools " : { messages : [ .... ,
    "model / tools " : { messages : [ .... ,
    "model / tools " : { messages : [ .... ,
    "model / tools " : { messages : [ .... 
}
"""
# 循环获取块
for chunk in stream:
    # 字典key - value分离！ key判定动作  value具体取值
    # items -> "model" -> node_name    { messages : [ .... , -> state
    for node_name,state in chunk.items():
        # 避开 {'TodoListMiddleware.after_model': None} 数据的干扰
        # 如果 state 为None 或者 没有messages属性，我就不处理！
        if not state or "messages" not in state: continue
        # state = { messages : [ .... ,
        # 获取本次的消息信息集合
        messages = state["messages"]
        # 获取值之前做一下判断，不为null，且是集合 过滤 {'PatchToolCallsMiddleware.before_agent': {'messages': Overwrit
        if messages and isinstance(messages,list):
            # [AIMessage(content='', additional_kwargs={'refusal': None}, respo
            # 获取messages集合的最后一条数据！当前chunk的真实结果
            last_message = messages[-1]
            if node_name == "model":
                # 三种可能，调用了大模型，大模型决定 调用工具 | 调用subagent | 直接最终的返回结果
                if last_message.tool_calls:
                    # 调用了哪个大模型，大模型绝对调用哪个工具 or 调用哪个子代理（subagent）
                    for tool_call in last_message.tool_calls:
                        if tool_call['name'] == 'task':
                            # 决定调用了子智能体
                            print(f"【大模型】决定调用子智能体：{tool_call['args']['subagent_type']}")
                        else:
                            # 决定调用哪个工具
                            print(f"【大模型】决定调用工具：{tool_call['name']} 传入的参数：{tool_call['args']}")
                elif last_message.content:
                    # 最终大模型润色后的结果 -》 agent 最后一条结果
                    print(f"【大模型】最终执行的结果：{last_message.content}")
            elif node_name == "tools":
                # 一种课程，agent 要调用具体的工具了！！工具返回了具体的结果
                # 获取tool执行返回的结果
                tool_return_result = last_message.content[:100] + "..."
                tool_name = last_message.name
                # sse | websocket -> 数据 -》 推送 -》 前端 -》 展示 调用工具和返回的结果
                print(f"【agent】调用了{tool_name}工具，返回的结果为：{tool_return_result}")


"""
{'messages': 
  [  1.  HumanMessage(content='查询人工智能和机器人的热门新闻信息！', additional_kwargs={}, response_metadata={}, id='3f3ba066-2fdc-4b3c-8b4d-9b719d658ee0'), 
     2.  调用模型规定后续的走向！ AIMessage(content='',  tool_calls=[{'name': 'internet_search', 'args': {'query': '人工智能 机器人', 'topic': 'news', 'max_results': 5, 'include_raw_content': False}, 
     3.  调用工具了 ToolMessage(content='{"query": "人工智能 机器人", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.marinelink.com/news/kimm-fast-tracks-ai-system-robotics-536890", "title": "KIMM Fast Tracks AI System, Robotics Development - Maritime News, Maritime Magazine", "score": 0.6052631, "published_date": "Thu, 12 Mar 2026 07:51:22 GMT", "content": "**Korea Institute of Machinery and Materials (KIMM)** has developed a new robot task artificial intelligence system designed to learn and perfort Developed sfosform Japan’s Robots With AI. March 9, 2026, 4:15\u202fAM UTC. A Silicon Valley-born AI startup is turning to Japan to prove AI can reshape one of the world’s largest industrial robot supply chains. Integral AI Inc., a five-year-old company founded by former Google researchers Jad Tarifi and Nima Asgharbeygi, develops AI models geared for automated systems such as robots and self-driving cars. The company has worked with auto parts maker Denso Corp.since 2021to help teach industrial robots new skills by observing demonstrations. The 15-person startup is holding initial discussions with Toyota Motor Corp., Sony Group Corp., Honda Motor Co., Nissan Motor Co. and ... ## Learn more about Bloomberg Law or Log In to keep reading:. ### See Breaking News in Context. Bloomberg Law provides trusted coverage of current events enhanced with legal analysis. ### Already a subscriber? Log in to keep reading or access research tools and resources. © 2026 Bloomberg Industry Group, Inc.", "raw_content": null}, {"url": "https://techcrunch.com/2026/03/09/qualcomms-partnership-with-neura-robotics-is-just-the-beginning/", "title": "Qualcomm’s partnership with Neura Robotics is just the beginning - TechCrunch", "score": 0.41151342, "published_date": "Mon, 09 Mar 2026 15:58:11 GMT", "content": "German robotics startup Neura Robotics has inked a partnership with semiconductor giant Qualcomm to build the next generation of robots and physical AI. The deal is the latest coupling in the emerging physical AI industry between robotics startups and larger tech hardware and software companies. “This collaboration marks a major step toward making physical AI real: open, scalable, and trusted,” David Reger, CEO and founder of Neura Robotics, said in a press release. For instance, Boston Dynamics announced a strategic partnership with Google DeepMind in January to speed up the development of the robotic company’s Atlas humanoid robot by using Google’s AI foundational models. ### Save up to $300 or 30% to TechCrunch Founder Summit. In Neura’s case, the company gets to build and test robots designed for the chips they are running on while Qualcomm gets an intimate look at how robotic companies can use its processors. AI, humanoids, Neura Robotics, nvidia, Qualcomm, Robotics, robotics.", "raw_content": null}, {"url": "https://www.techradar.com/pro/the-rise-of-the-silver-collar-workforce", "title": "The rise of the silver collar workforce - TechRadar", "score": 0.2764492, "published_date": "Wed, 11 Mar 2026 15:15:51 GMT", "content": "Robots managing production or AI systems optimizing infrastructure have become active participants in work, yet human judgment remains central. Attention is moving away from what machines can achieve alone and toward how intelligence is applied responsibly in real-world environments, where every decision carries consequences. As machines moves from theory to action, the human role evolves from operator to steward, ensuring that systems perform safely, efficiently, and ethically. Systems trained in controlled environments often struggle when faced with real-world complexity, including unexpected human behavior or environmental variation. Humans can observe how intelligent systems behave under varying circumstances, intervene if decisions risk negative outcomes, and validate results before changes are applied to the real world. The silver collar era is already underway, with humans working alongside increasingly autonomous machines in factories, infrastructure, and industrial environments. In the silver collar era, progress is measured not by what machines do on their own, but by the quality of collaboration between humans and machines.", "raw_content": null}], "response_time": 0.41, "request_id": "2b4b73bd-0873-47e4-949e-5e1ff3787cb7"}', name='internet_search', id='486a6105-7468-48dc-bd5e-be798b8cc76c', tool_call_id='call_b64da90394a94bad830449'), 
     4.  调用了模型，进行tool返回结果的润色 AIMessage(content='以下是关于人工智能和机器人的最新热门新闻：\n\n1. **KIMM Fast Tracks AI System, Robotics Development**\n   - 韩国机械材料研究所（KIMM）开发了一种新的机器人任务人工智能系统，该系统通过观察人类演示来学习并执行日常重复性任务。这项技术旨在扩展服务机器人在零售、仓库物流和一般工作场所支持等领域的角色。\n   - [阅读更多](https://www.marinelink.com/news/kimm-fast-tracks-ai-system-robotics-536890)\n\n2. **Qualcomm teams up with Neura Robotics for physical AI push**\n   - Qualcomm与Neura Robotics合作，结合了Qualcomm在AI计算、连接性和机器人平台方面的专长以及Neura在深度机器人系统专业知识和具身AI软件上的优势，目标是加速现实世界中的机器人智能。\n   - [阅读更多](https://www.telecoms.com/ai/qualcomm-teams-up-with-neura-robotics-for-physical-ai-push)\n\n3. **Ex-Google Researcher Seeks to Transform Japan’s Robots With AI**\n   - 一家由前Google研究人员创立的硅谷AI初创公司Integral AI正致力于利用AI重塑日本的工业机器人供应链。\n   - [阅读更多](https://news.bloomberglaw.com/daily-labor-report/ex-google-researcher-seeks-to-transform-japans-robots-with-ai)\n\n4. **Qualcomm’s partnership with Neura Robotics is just the beginning**\n   - 德国机器人初创公司Neura Robotics与半导体巨头Qualcomm达成合作，共同构建下一代机器人和物理AI。这种合作关系标志着朝着开放、可扩展且值得信赖的物理AI迈出的重要一步。\n   - [阅读更多](https://techcrunch.com/2026/03/09/qualcomms-partnership-with-neura-robotics-is-just-the-beginning/)\n\n5. **The rise of the silver collar workforce**\n   - 在银领时代，人类与日益自主的机器在工厂、基础设施和工业环境中并肩工作。随着机器从理论走向实践，人类的角色从操作员演变为监督者，确保系统安全、高效且合乎道德地运行。\n   - [阅读更多](https://www.techradar.com/pro/the-rise-of-the-silver-collar-workforce)', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 492, 'prompt_tokens': 7463, 'total_tokens': 7955, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'qwen-max', 'system_fingerprint': None, 'id': 'chatcmpl-f6915f0b-7274-9662-ab3e-a30c2b13c2cc', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ce238-a059-77d3-9f98-c7344289038c-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 7463, 'output_tokens': 492, 'total_tokens': 7955, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}
"""

