from solutions.tools.graphcypher import cypher_qa
from solutions.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 对 KG 结果做二次增强：结合患者信息和健康摘要给出个性化解读
_enhance_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业医疗助手，根据知识图谱查询结果，结合患者个人情况给出个性化解读。

要求：
1. 直接基于知识图谱结果回答，不要编造
2. 结合患者病史、过敏史给出针对性提示
3. 如涉及心脑血管等重大疾病，必须加预警
4. 回答精炼，重点突出
5. 如知识图谱无结果，明确说明"知识图谱中暂无该疾病的结构化数据"

患者信息：性别={gender}，年龄={age}，既往病史={history}
过敏史：{allergy}，禁忌药：{forbidden}
历史健康背景：{health_summary}"""),
    ("user", "用户问题：{question}\n\n知识图谱查询结果：\n{kg_result}")
])

_enhance_chain = _enhance_prompt | llm | StrOutputParser()


def create_agent():
    def run(state: dict) -> str:
        user_input = state.get("user_input", "")
        patient = state.get("patient_info", {})

        # Step1: 调用原始 GraphCypherQAChain（含完整 Cypher 生成模板）
        kg_result = ""
        try:
            result = cypher_qa.invoke({"query": user_input})
            kg_result = result.get("result", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            kg_result = f"查询异常：{str(e)}"

        # Step2: 结合患者信息做个性化增强
        return _enhance_chain.invoke({
            "question": user_input,
            "kg_result": kg_result or "知识图谱未返回结果",
            "gender": patient.get("性别", "未知"),
            "age": patient.get("年龄", "未知"),
            "history": patient.get("既往病史", "无"),
            "allergy": patient.get("过敏史", "无"),
            "forbidden": patient.get("禁忌药", "无"),
            "health_summary": state.get("health_summary", "暂无历史记录"),
        })

    return run
