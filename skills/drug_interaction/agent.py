from solutions.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业AI药师，专门分析药物相互作用。
对用户提到的多种药物，给出：
1. 各药物间相互作用风险等级（无/轻度/中度/严重）
2. 可能的不良反应
3. 具体用药建议（时间间隔、剂量调整等）
4. 需要监测的指标

患者信息：性别={gender}，年龄={age}，既往病史={history}
过敏史：{allergy}，禁忌药：{forbidden}
历史健康背景：{health_summary}
严重相互作用必须加粗警示，如患者有过敏史必须特别提醒。"""),
    ("user", "{input}")
])
_chain = _prompt | llm | StrOutputParser()


def create_agent():
    def run(state: dict) -> str:
        user_input = state.get("user_input", "")
        patient = state.get("patient_info", {})
        return _chain.invoke({
            "input": user_input,
            "gender": patient.get("性别", "未知"),
            "age": patient.get("年龄", "未知"),
            "history": patient.get("既往病史", "无"),
            "allergy": patient.get("过敏史", "无"),
            "forbidden": patient.get("禁忌药", "无"),
            "health_summary": state.get("health_summary", "暂无历史记录"),
        })
    return run
