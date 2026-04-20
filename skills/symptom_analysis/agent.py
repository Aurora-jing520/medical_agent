from solutions.llm import llm
from solutions.tools.medical_search import medical_hybrid_search
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业的医疗问诊助手，擅长症状分析。
根据患者描述的症状，结合知识图谱检索结果，给出：
1. 症状分析（可能的疾病）
2. 严重程度评估
3. 建议的处理方式
4. 是否需要立即就医

患者信息：性别={gender}，年龄={age}，既往病史={history}，过敏史={allergy}
历史健康背景：{health_summary}
回答精炼专业，如涉及心脑血管等重大疾病务必预警。"""),
    ("user", "症状描述：{input}\n\n参考信息：{context}")
])

_chain = _prompt | llm | StrOutputParser()


def create_agent():
    def run(state: dict) -> str:
        user_input = state.get("user_input", "")
        patient = state.get("patient_info", {})
        result = medical_hybrid_search.search(user_input)
        context = result.get("answer", "暂无检索结果")
        return _chain.invoke({
            "input": user_input,
            "context": context,
            "gender": patient.get("性别", "未知"),
            "age": patient.get("年龄", "未知"),
            "history": patient.get("既往病史", "无"),
            "allergy": patient.get("过敏史", "无"),
            "health_summary": state.get("health_summary", "暂无历史记录"),
        })
    return run
