"""
LangGraph 主图
流程：memory_load → intent → dispatch（并行）→ summary → memory_save
"""
import asyncio
import time
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from solutions.langgraph_medical.state import MedicalState
from solutions.langgraph_medical.registry import registry
from solutions.langgraph_medical.intent_agent import intent_node
from solutions.langgraph_medical.summary_agent import summary_node
from solutions.langgraph_medical.memory_node import memory_load_node, memory_save_node


def _init_registry():
    if not registry.list_intents():
        registry.scan()


async def _run_single_agent(intent: str, state: MedicalState) -> tuple[str, str]:
    agent = registry.get(intent)
    if agent is None:
        return intent, f"[{intent}] Agent 未找到"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent, state)
        return intent, result
    except Exception as e:
        return intent, f"[{intent}] 执行失败：{str(e)}"


async def _dispatch_parallel(state: MedicalState) -> MedicalState:
    intents = state.get("intents", ["general_chat"])
    t0 = time.perf_counter()

    tasks = [_run_single_agent(intent, state) for intent in intents]
    results = await asyncio.gather(*tasks)

    agent_results = dict(state.get("agent_results", {}))
    for intent, result in results:
        agent_results[intent] = result

    timing = dict(state.get("timing", {}))
    timing["dispatch_parallel"] = round(time.perf_counter() - t0, 3)
    return {**state, "agent_results": agent_results, "timing": timing}


def dispatch_node(state: MedicalState) -> MedicalState:
    # asyncio.run() 在已有事件循环时会报错，用 nest_asyncio 或新线程跑
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _dispatch_parallel(state))
        return future.result()


def build_graph() -> StateGraph:
    _init_registry()

    builder = StateGraph(MedicalState)

    builder.add_node("memory_load", memory_load_node)
    builder.add_node("intent", intent_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("summary", summary_node)
    builder.add_node("memory_save", memory_save_node)

    builder.set_entry_point("memory_load")
    builder.add_edge("memory_load", "intent")
    builder.add_edge("intent", "dispatch")
    builder.add_edge("dispatch", "summary")
    builder.add_edge("summary", "memory_save")
    builder.add_edge("memory_save", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def chat(
    user_input: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    patient_info: dict = None,
) -> dict:
    """
    对外接口

    Args:
        user_input: 用户输入
        user_id: 用户ID，用于长期记忆
        session_id: 会话ID，用于短期记忆和多轮上下文
        patient_info: 可选，首次传入基本信息

    Returns:
        final_answer, intents, agent_results, timing, health_summary
    """
    state = graph.invoke(
        {
            "user_input": user_input,
            "user_id": user_id,
            "session_id": session_id,
            "intents": [],
            "agent_results": {},
            "patient_info": patient_info or {},
            "health_summary": "",
            "final_answer": "",
            "timing": {},
        },
        config={"configurable": {"thread_id": session_id}},
    )
    return {
        "final_answer": state["final_answer"],
        "intents": state["intents"],
        "agent_results": state["agent_results"],
        "timing": state["timing"],
        "health_summary": state.get("health_summary", ""),
    }
