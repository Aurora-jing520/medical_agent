from solutions.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import urllib.parse
import urllib3
import json

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业AI药师，根据用户问题和药物信息给出专业解答。
包含：药物用途、用法用量、主要副作用、禁忌症、特殊人群注意事项。
回答精炼，重点突出安全风险。
患者信息：性别={gender}，年龄={age}，既往病史={history}
过敏史：{allergy}，禁忌药：{forbidden}
历史健康背景：{health_summary}
如患者有过敏史或禁忌药，必须在回答中明确提示风险。"""),
    ("user", "问题：{input}\n\n药物参考信息：{drug_info}")
])
_chain = _prompt | llm | StrOutputParser()

APPCODE = "dbcff0aeda04460b998e0aa184605209"


def _fetch_drug_info(drug_name: str) -> str:
    try:
        querys = f"classifyId=599ad2a0600b2149d689b75a&searchType=1&searchKey={urllib.parse.quote(drug_name)}&page=1&maxResult=5"
        http = urllib3.PoolManager()
        resp = http.request("GET", f"http://drug.market.alicloudapi.com/drugDetail?{querys}",
                            headers={"Authorization": f"APPCODE {APPCODE}"})
        data = json.loads(resp.data.decode("utf-8"))
        return json.dumps(data, ensure_ascii=False)[:1000]
    except Exception:
        return ""


def create_agent():
    def run(state: dict) -> str:
        user_input = state.get("user_input", "")
        patient = state.get("patient_info", {})
        # 简单提取药名：取输入中2-6字的词
        words = [w for w in user_input.replace("，", " ").replace("。", " ").split() if 2 <= len(w) <= 6]
        drug_info = ""
        for w in words:
            info = _fetch_drug_info(w)
            if info:
                drug_info = info
                break
        if not drug_info:
            drug_info = "未能从API获取药物信息，请基于医学知识回答"
        return _chain.invoke({
            "input": user_input,
            "drug_info": drug_info,
            "gender": patient.get("性别", "未知"),
            "age": patient.get("年龄", "未知"),
            "history": patient.get("既往病史", "无"),
            "allergy": patient.get("过敏史", "无"),
            "forbidden": patient.get("禁忌药", "无"),
            "health_summary": state.get("health_summary", "暂无历史记录"),
        })
    return run
