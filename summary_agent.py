"""
SummaryAgent — 整合多个 Agent 结果，生成最终回答
"""
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from solutions.llm import llm
from solutions.langgraph_medical.state import MedicalState

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业医疗助手，负责整合多个专科分析结果，给患者一个完整、清晰的回答。

要求：
1. 综合所有分析结果，去除重复内容
2. 按"症状分析→用药建议→科室推荐→注意事项"顺序组织
3. 涉及心脑血管等重大疾病必须加预警
4. 如患者有过敏史或禁忌药，回答中必须明确提示
5. 结尾固定加：【本回答仅供参考，如症状加重请及时就医】
6. 回答精炼，不超过400字

患者信息：性别={gender}，年龄={age}，既往病史={history}
过敏史：{allergy}，禁忌药：{forbidden}

历史健康摘要（供参考，了解患者背景）：
{health_summary}"""),
    ("user", "用户问题：{user_input}\n\n各专科分析结果：\n{agent_results}")
])
_chain = _prompt | llm | StrOutputParser()


def summary_node(state: MedicalState) -> MedicalState:
    t0 = time.perf_counter()

    agent_results = state.get("agent_results", {})
    patient = state.get("patient_info", {})

    # 格式化各 agent 结果
    results_text = "\n\n".join(
        f"【{intent}】\n{result}"
        for intent, result in agent_results.items()
        if result
    ) or "暂无分析结果"

    final = _chain.invoke({
        "user_input": state["user_input"],
        "agent_results": results_text,
        "gender": patient.get("性别", "未知"),
        "age": patient.get("年龄", "未知"),
        "history": patient.get("既往病史", "无"),
        "allergy": patient.get("过敏史", "无"),
        "forbidden": patient.get("禁忌药", "无"),
        "health_summary": state.get("health_summary", "暂无历史记录"),
    })

    elapsed = time.perf_counter() - t0
    timing = dict(state.get("timing", {}))
    timing["summary_agent"] = round(elapsed, 3)

    return {**state, "final_answer": final, "timing": timing}
