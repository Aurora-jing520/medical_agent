"""
记忆节点：LangGraph 图中的读写记忆节点
- memory_load_node: 图入口，读取长期记忆注入 state
- memory_save_node: 图出口，保存问诊结果 + 异步刷新摘要
"""
import time
from solutions.langgraph_medical.state import MedicalState
from solutions.langgraph_medical.memory_manager import memory_manager


def memory_load_node(state: MedicalState) -> MedicalState:
    """
    图入口节点：从长期记忆加载健康档案和摘要，注入 state
    优先从短期缓存取摘要（缓存命中则跳过文件读取）
    """
    t0 = time.perf_counter()
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")

    # 1. 读取健康档案（长期记忆）
    profile = memory_manager.get_health_profile(user_id)

    # 2. 读取 LLM 摘要（优先短期缓存）
    summary, is_hit = memory_manager.get_summary_cached(user_id)

    # 3. 读取会话状态（短期记忆）
    session = memory_manager.get_session(session_id)
    session_round = session.get("round", 0) + 1
    memory_manager.update_session(session_id, {"round": session_round})

    # 4. 合并 patient_info：state 传入的优先，缺失字段从长期记忆补全
    patient_info = dict(state.get("patient_info") or {})
    if not patient_info.get("性别") and profile.get("性别"):
        patient_info["性别"] = profile["性别"]
    if not patient_info.get("年龄") and profile.get("年龄"):
        patient_info["年龄"] = profile["年龄"]
    if not patient_info.get("既往病史") and profile.get("慢性病"):
        patient_info["既往病史"] = "、".join(profile["慢性病"])
    if profile.get("过敏史"):
        patient_info["过敏史"] = "、".join(profile["过敏史"])
    if profile.get("禁忌药"):
        patient_info["禁忌药"] = "、".join(profile["禁忌药"])

    elapsed = round(time.perf_counter() - t0, 3)
    timing = dict(state.get("timing") or {})
    timing["memory_load"] = elapsed
    timing["memory_cache_hit"] = is_hit

    return {
        **state,
        "patient_info": patient_info,
        "health_summary": summary,
        "timing": timing,
    }


def memory_save_node(state: MedicalState) -> MedicalState:
    """
    图出口节点：将本次问诊结果写入长期记忆，异步刷新 LLM 摘要
    """
    t0 = time.perf_counter()
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")

    # 1. 从 agent_results 提取关键信息存入问诊历史
    intents = state.get("intents", [])
    agent_results = state.get("agent_results", {})
    patient_info = state.get("patient_info", {})

    consultation = {
        "user_input": state.get("user_input", ""),
        "intents": intents,
        "symptoms": _extract_symptoms(state.get("user_input", "")),
        "final_answer": state.get("final_answer", "")[:300],  # 截断防止过长
    }
    memory_manager.add_consultation(user_id, consultation)

    # 2. 更新健康档案（从 patient_info 同步）
    profile_updates = {}
    if patient_info.get("性别"):
        profile_updates["性别"] = patient_info["性别"]
    if patient_info.get("年龄"):
        profile_updates["年龄"] = patient_info["年龄"]
    if patient_info.get("既往病史"):
        profile_updates["慢性病"] = [patient_info["既往病史"]]
    if profile_updates:
        memory_manager.update_health_profile(user_id, profile_updates)

    # 3. 更新会话状态
    memory_manager.update_session(session_id, {
        "last_intents": intents,
        "last_input": state.get("user_input", ""),
    })

    # 4. 异步刷新 LLM 摘要（不阻塞返回）
    memory_manager.async_refresh_summary(user_id)

    elapsed = round(time.perf_counter() - t0, 3)
    timing = dict(state.get("timing") or {})
    timing["memory_save"] = elapsed

    return {**state, "timing": timing}


def _extract_symptoms(text: str) -> list[str]:
    """简单关键词提取症状，用于统计"""
    keywords = ["头痛", "发烧", "咳嗽", "胸闷", "腹痛", "恶心", "头晕",
                "乏力", "失眠", "腰痛", "关节痛", "皮疹", "腹泻", "便秘"]
    return [k for k in keywords if k in text]
