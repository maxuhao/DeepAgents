import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient

# 1. 环境准备
load_dotenv(find_dotenv())
llm = init_chat_model(model=os.getenv("LLM_QWEN_MAX"), model_provider="openai")

# 初始化 Tavily 客户端 (关键：用于获取真实数据)
# 确保 .env 中配置了 TAVILY_API_KEY
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ==========================================
# 第一步：定义真实联网工具 (Real Tools)
# ==========================================

@tool
def search_attractions(city: str):
    """
    【真实联网】搜索指定城市的著名景点、特色及评分。
    """
    print(f"\n[景点官] 正在联网搜索 {city} 的热门景点...")
    # 构建搜索查询，获取Top 5结果
    query = f"{city} 必去旅游景点 排行榜 介绍"
    response = tavily_client.search(query=query, max_results=5)

    # 整理搜索结果
    results = []
    for item in response.get('results', []):
        results.append(f"- {item['title']}: {item['content'][:150]}...")

    return "\n".join(results) if results else f"未找到 {city} 的相关景点信息。"


@tool
def search_transport(origin: str, dest: str):
    """
    【真实联网】查询两地之间的交通方式（高铁/飞机）及参考价格。
    """
    print(f"\n✈️ [出行师] 正在联网查询 {origin} -> {dest} 的交通票价...")
    # 同时查询高铁和机票
    query = f"{origin} 到 {dest} 高铁票价 飞机票价 时间"
    response = tavily_client.search(query=query, max_results=3)

    results = []
    for item in response.get('results', []):
        results.append(f"- {item['title']}: {item['content'][:150]}...")

    return "\n".join(results)


@tool
def calculate_budget(expense_list: str):
    """
    【智能计算】计算总预算。
    输入示例：'机票:约500元, 酒店:200*3=600元, 门票:300元'
    这个工具本身不需要联网，它负责解析文本并求和。
    """
    print(f"\n💰 [财务官] 正在核算总成本: {expense_list[:50]}...")
    # 使用 LLM 来做这个计算（因为输入是非结构化文本，LLM 比正则更准）
    # 这里为了简单演示，我们用一个简易的提取逻辑，或者直接让 Agent 自己算
    # 其实 BudgetPlanner Agent 本身就能算，但为了符合 Tool 的定义，我们写一个辅助函数
    import re
    # 提取所有数字
    numbers = re.findall(r'(\d+)', expense_list)
    total = sum(int(n) for n in numbers)
    return f"根据提取到的数字，预估总费用约为: {total} 元 (仅供参考，请以实际为准)"


# ==========================================
# 第二步：创建子智能体 (Sub-Agents)
# ==========================================

# 1. 景点推荐官 (Attraction Specialist)
attraction_agent = create_deep_agent(
    model=llm,
    name="AttractionAgent",
    tools=[search_attractions],
    system_prompt="""你是一个【景点推荐官】。
    你的任务是调用 `search_attractions` 工具去网上搜索最真实、最热门的景点信息。
    请挑选 3-4 个最值得去的地方，简要介绍它们的特色。不要编造！"""
)
sub_attraction = CompiledSubAgent(
    name="AttractionAgent",
    description="负责联网搜索目的地最热门的景点和游玩建议。",
    runnable=attraction_agent
)

# 2. 出行规划师 (Transport Planner)
transport_agent = create_deep_agent(
    model=llm,
    name="TransportAgent",
    tools=[search_transport],
    system_prompt="""你是一个【出行规划师】。
    你的任务是调用 `search_transport` 工具去网上查询真实的交通方案。
    请对比高铁和飞机的价格/时间，给出性价比最高的建议。"""
)
sub_transport = CompiledSubAgent(
    name="TransportAgent",
    description="负责联网查询交通方式（高铁/飞机）及实时参考价格。",
    runnable=transport_agent
)

# 3. 财务管家 (Budget Keeper) - 智能动态版
budget_agent = create_deep_agent(
    model=llm,
    name="BudgetAgent",
    tools=[calculate_budget],
    system_prompt="""你是一个【财务管家】。
    
    【任务目标】
    你需要汇总交通、门票费用，并根据用户的【旅行偏好】估算食宿成本。
    
    【动态估算逻辑】
    请仔细分析用户的需求描述（例如："穷游"、"带孩子"、"预算充足"、"学生党"），动态调整估算标准：
    
    - **穷游/学生**：住宿按 150元/晚 (青旅)，餐饮按 80元/天 计算。
    - **普通/舒适 (默认)**：住宿按 400元/晚 (舒适酒店)，餐饮按 200元/天 计算。
    - **豪华/享受**：住宿按 1200元/晚 (五星级)，餐饮按 600元/天 计算。
    
    如果用户没有明确说明，默认按【普通/舒适】标准计算。
    最后输出总预算表，并注明你是按什么标准估算的。"""
)

sub_budget = CompiledSubAgent(
    name="BudgetAgent",
    description="负责汇总所有开销，并估算食宿成本，给出总预算。",
    runnable=budget_agent
)

# ==========================================
# 第三步：创建主智能体 (Main Agent)
# ==========================================

ceo_agent = create_deep_agent(
    model=llm,
    name="TravelCEO",
    subagents=[sub_attraction, sub_transport, sub_budget],
    system_prompt="""你是【私人旅行管家 (CEO)】。
    用户会告诉你他的旅行计划。你需要指挥你的专家团队完成这份计划书。

    【执行SOP】
    1. **调用 AttractionAgent**：先确定去哪玩（获取景点信息）。
    2. **调用 TransportAgent**：再确定怎么去（获取交通成本）。
    3. **调用 BudgetAgent**：最后算算要花多少钱（汇总总成本）。
    4. **最终输出**：给用户一份包含【行程亮点】、【交通方案】、【预算预估】的完整报告。语气要专业且热情。
    """
)

# ==========================================
# 第四步：运行测试
# ==========================================
if __name__ == "__main__":
    # 这里换一个具体的例子，测试联网能力
    request = "我想从北京去杭州玩3天，帮我规划一下，我想去西湖和灵隐寺。"
    print(f"用户需求: {request}\n")
    print("Agent 团队正在联网工作中...\n")

    stream = ceo_agent.stream({
        "messages": [{"role": "user", "content": request}]
    })

    for chunk in stream:
        if "model" in chunk and "messages" in chunk["model"]:
            msg = chunk["model"]["messages"][0]
            # 打印主智能体的思考过程（通常是他在分配任务）
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['name'] == 'task':
                        print(f"👉 [CEO] 正在委派任务给: {tc['args'].get('subagent_type')}")

            # 打印最终回复
            if msg.content and not msg.tool_calls:
                print(f"\n[最终方案]:\n{msg.content}")