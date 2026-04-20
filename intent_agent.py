"""
IntentAgent — 基于 LLM 语义理解的多意图识别
支持单条输入识别出多个意图（如"头痛+能吃布洛芬吗" → symptom_analysis + drug_query）
"""
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from solutions.llm import llm
from solutions.langgraph_medical.registry import registry
from solutions.langgraph_medical.state import MedicalState


_SYSTEM = """你是一个医疗问答系统的意图识别模块。
根据用户输入，从以下意图类别中选出所有匹配的意图（可多选）：

{intent_descriptions}

规则：
1. 输出 JSON 数组，只包含意图标识符，例如：["symptom_analysis", "drug_query"]
2. 最多选 3 个意图，按相关度从高到低排列
3. 如果没有明确医疗意图，返回 ["general_chat"]
4. 只输出 JSON 数组，不要有任何其他内容
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("user", "{input}")
])
_chain = _prompt | llm | StrOutputParser()


def intent_node(state: MedicalState) -> MedicalState:
    t0 = time.perf_counter()

    descriptions = registry.get_all_descriptions()
    raw = _chain.invoke({
        "intent_descriptions": descriptions,
        "input": state["user_input"]
    }).strip()

    # 解析 JSON，容错处理
    try:
        # 去掉可能的 markdown 代码块
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        intents = json.loads(raw)
        if not isinstance(intents, list):
            intents = ["general_chat"]
    except Exception:
        intents = ["general_chat"]

    # 过滤掉 registry 中不存在的意图
    valid = registry.list_intents()
    intents = [i for i in intents if i in valid] or ["general_chat"]

    elapsed = time.perf_counter() - t0
    timing = dict(state.get("timing", {}))
    timing["intent_agent"] = round(elapsed, 3)

    return {**state, "intents": intents, "timing": timing}
