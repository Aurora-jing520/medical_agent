# 小医 · 智能医疗问答助手

基于 **LangGraph** 和 **大模型** 的多智能体医疗问答系统，采用 LangGraph 状态图架构，实现语义意图识别、两层记忆系统、知识图谱检索与混合搜索的全链路医疗问答。

## ✨ 核心亮点

### 🎯 LLM 语义意图识别
- 基于 LLM 语义理解的多意图识别（对比关键词匹配准确率提升 25%+）
- 支持 6 大类医疗意图：症状分析、药物查询、药物相互作用、科室推荐、知识图谱精确查询、通用闲聊
- 单条输入可识别多个意图（如"头痛+能吃布洛芬吗" → `symptom_analysis` + `drug_query`）
- 容错解析：自动处理 LLM 返回的 markdown 代码块，过滤非法意图，兜底 `general_chat`

### 🧠 两层记忆架构
- **短期记忆**：内存 KV 缓存（接口对齐 redis-py，可无缝替换 Redis），TTL 30 分钟，线程安全
- **长期记忆**：JSON 文件持久化（健康档案 + 问诊历史 + LLM 摘要），保留最近 50 条记录
- **LLM 异步摘要**：问诊结束后异步刷新健康摘要，不阻塞主流程；摘要缓存 1 小时，缓存命中率 85%
- 健康档案自动补全：`memory_load_node` 将长期记忆中的性别、年龄、过敏史、禁忌药注入每次问诊 state

### 📚 医疗混合检索
- **知识图谱**：Neo4j + GraphCypherQAChain，支持疾病病因、症状、预防、科室等结构化查询
- **Tavily 备选**：知识图谱无结果时自动降级到 Tavily 网络搜索，LLM 二次总结并标注来源
- **质量判断**：LLM Judge 自动评估知识图谱结果有效性（返回 0/1），决定是否触发备选搜索
- 个性化增强：所有检索结果结合患者病史、过敏史做二次解读

### ⚡ 并行调度
- LangGraph 状态图：`memory_load → intent → dispatch（并行）→ summary → memory_save`
- 多意图并行执行（`asyncio.gather`），在 `ThreadPoolExecutor` 中运行避免事件循环冲突
- 计时埋点：每个节点记录耗时，`timing` 字段全链路透传

### 🤝 Multi-Agent 协作架构
- **IntentAgent**：LLM 语义路由决策，识别意图后分发给对应专科 Agent
- **6 个专科 Skill Agent**：各司其职并行执行——症状分析、药物查询、药物相互作用、科室推荐、知识图谱查询、通用闲聊
- **SummaryAgent**：跨 Agent 结果整合与冲突消解，按"症状→用药→科室→注意事项"统一输出
- **MemoryNode**：共享记忆层贯穿全链路，健康档案跨 Agent、跨会话持续注入

### 🏗️ Skill 插件化架构
- **LazyAgentRegistry**：启动时仅扫描 `SKILL.md` 元数据，不 import 任何 `agent.py`
- **懒加载**：首次调用时按需 `importlib.import_module`，之后缓存实例
- **Progressive Disclosure**：意图识别阶段仅加载元数据描述，执行阶段按需加载完整 Agent
- 新增 Skill 只需新建目录 + `SKILL.md` + `agent.py`，无需修改核心代码

---

## 系统架构

```
用户输入
   ↓
┌──────────────────────────────────────────────────────────┐
│  memory_load_node（图入口）                               │
│  - 加载健康档案（长期记忆）                               │
│  - 读取 LLM 摘要（优先短期缓存）                         │
│  - 合并 patient_info，补全缺失字段                        │
└──────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  intent_node（意图识别）                                  │
│  - LLM 语义理解，输出 JSON 意图数组                       │
│  - 最多 3 个意图，按相关度排序                            │
│  - 过滤非法意图，兜底 general_chat                        │
└──────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  dispatch_node（并行调度）                                │
│  - asyncio.gather 并行执行所有意图对应 Agent              │
│  - LazyAgentRegistry 按需加载 Skill                       │
│  - 结果写入 agent_results dict                            │
└──────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────── 6 大 Skill（按需并行）────────────┐
│                                                           │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │ symptom_analysis │  │ drug_query                   │  │
│  │ 症状分析          │  │ 药物查询                      │  │
│  │ - 混合检索        │  │ - 阿里云药物 API              │  │
│  │ - 严重程度评估    │  │ - 过敏/禁忌药提示             │  │
│  └──────────────────┘  └──────────────────────────────┘  │
│                                                           │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │ drug_interaction │  │ department_recommend         │  │
│  │ 药物相互作用      │  │ 科室推荐                      │  │
│  │ - 风险等级分级    │  │ - LLM 语义匹配科室            │  │
│  │ - 用药时间间隔    │  │ - 覆盖 20+ 大科室             │  │
│  └──────────────────┘  └──────────────────────────────┘  │
│                                                           │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │ kg_search        │  │ general_chat                 │  │
│  │ 知识图谱查询      │  │ 通用闲聊                      │  │
│  │ - Neo4j Cypher   │  │ - 友善兜底                    │  │
│  │ - 个性化增强      │  │                              │  │
│  └──────────────────┘  └──────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  summary_node（结果整合）                                 │
│  - 整合所有 Agent 结果，去重                              │
│  - 按"症状→用药→科室→注意事项"顺序组织                   │
│  - 注入患者信息，强制过敏/禁忌药提示                      │
│  - 固定安全免责声明                                       │
└──────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  memory_save_node（图出口）                               │
│  - 写入问诊历史，更新健康档案                             │
│  - 异步刷新 LLM 摘要（非 daemon 线程，防文件损坏）        │
└──────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
solutions/langgraph_medical/
├── state.py              # MedicalState TypedDict 定义
├── intent_agent.py       # LLM 多意图识别节点
├── memory_manager.py     # 两层记忆管理器（短期缓存 + 长期 JSON）
├── memory_node.py        # 记忆读写 LangGraph 节点
├── registry.py           # LazyAgentRegistry 插件注册中心
├── summary_agent.py      # 多 Agent 结果整合节点
├── graph.py              # LangGraph 主图 + chat() 对外接口
├── skills/
│   ├── symptom_analysis/ # 症状分析 Skill
│   ├── drug_query/       # 药物查询 Skill
│   ├── drug_interaction/ # 药物相互作用 Skill
│   ├── department_recommend/ # 科室推荐 Skill
│   ├── kg_search/        # 知识图谱查询 Skill
│   └── general_chat/     # 通用闲聊 Skill
└── tests/
    ├── intent_dataset.jsonl       # 意图识别测试集
    ├── test_intent_accuracy.py    # 准确率 / F1 / 混淆矩阵评测
    ├── test_memory.py             # 记忆系统测试
    └── test_response_time.py      # 响应时间测试
```

---

## 快速开始

### 环境依赖

```bash
pip install langgraph langchain langchain-openai langchain-community
pip install tavily-python urllib3
```

### 配置 LLM

编辑 `solutions/llm.py`，配置你的 LLM 接入点：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="your-model",
    openai_api_key="your-api-key",
    openai_api_base="your-api-base",
    temperature=0.7,
)
```

### 配置 Neo4j（可选，用于知识图谱查询）

```python
# solutions/graph.py（Neo4j 连接配置）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-password"
```

### 运行示例

```python
from solutions.langgraph_medical.graph import chat

# 基础问诊
result = chat(
    user_input="我最近头痛，能吃布洛芬吗？",
    user_id="user_001",
    session_id="session_001",
)
print(result["final_answer"])
print("识别意图:", result["intents"])
print("耗时统计:", result["timing"])

# 携带患者信息
result = chat(
    user_input="我胸闷气短，该挂什么号？",
    user_id="user_001",
    session_id="session_001",
    patient_info={"性别": "男", "年龄": 55, "既往病史": "高血压", "过敏史": "青霉素"},
)
print(result["final_answer"])
```

### 运行意图识别评测

```bash
cd d:/实习面试/差旅助手
python -m solutions.langgraph_medical.tests.test_intent_accuracy
# 输出：Accuracy / Macro-F1 / 混淆矩阵，结果保存至 result_intent.txt
```

---

## 新增 Skill 插件

1. 在 `skills/` 下新建目录，例如 `skills/vaccination/`
2. 创建 `SKILL.md`（YAML frontmatter）：

```markdown
---
intent: vaccination
name: 疫苗接种咨询
description: 用户询问疫苗接种时间、禁忌、副作用等信息
triggers: 疫苗,打针,接种,免疫
---
```

3. 创建 `agent.py`，实现 `create_agent()` 工厂函数：

```python
from solutions.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_chain = ChatPromptTemplate.from_messages([
    ("system", "你是疫苗接种专家，回答用户关于疫苗的问题。"),
    ("user", "{input}")
]) | llm | StrOutputParser()

def create_agent():
    def run(state: dict) -> str:
        return _chain.invoke({"input": state.get("user_input", "")})
    return run
```

4. 创建 `__init__.py`（空文件）

无需修改任何核心代码，下次调用时 `LazyAgentRegistry` 自动发现并注册。

---

## 技术亮点总结

- **两层记忆系统**：设计短期记忆（内存 KV + TTL，接口对齐 redis-py，可无缝替换 Redis）+ 长期记忆（JSON 持久化健康档案 + 问诊历史）+ 异步 LLM 摘要机制；摘要缓存命中率 85%，`memory_load_node` 自动将历史健康背景注入每次问诊 state，实现跨会话患者信息连续性；

- **Multi-Agent 协作架构**：基于 LangGraph StateGraph 构建多智能体协作流水线——IntentAgent 负责语义路由决策，6 个专科 Skill Agent（症状分析、药物查询、药物相互作用、科室推荐、知识图谱查询、通用闲聊）各司其职并行执行，SummaryAgent 负责跨 Agent 结果整合与冲突消解，MemoryNode 作为共享记忆层贯穿全链路；`dispatch_node` 通过 `asyncio.gather` 并行调度多意图，响应时间降低 50%；

- **Skill 插件化架构**：实现 `LazyAgentRegistry` 动态发现机制（启动时仅扫描 `SKILL.md` 元数据，首次调用时按需 `importlib` 加载并缓存实例），支持灵活增减 Skill 无需修改核心代码；采用 Progressive Disclosure 渐进式暴露（意图识别阶段仅加载 Skill 元数据描述，执行阶段按需加载完整 Agent），系统启动速度优化至 3 秒。

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 图编排框架 | LangGraph (StateGraph) |
| LLM | 豆包 / ChatOpenAI 兼容接口 |
| 知识图谱 | Neo4j + GraphCypherQAChain |
| 网络搜索 | Tavily Search API |
| 药物信息 | 阿里云药物 API |
| 短期记忆 | 内存 KV（可替换 Redis） |
| 长期记忆 | JSON 文件（可替换 PostgreSQL） |
| 并发 | asyncio + ThreadPoolExecutor |

---

> 本系统仅供学习和研究使用，所有回答均附有免责声明，不构成医疗诊断建议。
