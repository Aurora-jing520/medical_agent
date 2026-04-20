"""
两层记忆管理器
- 长期记忆：JSON 文件持久化（健康档案 + 问诊历史 + LLM摘要）
- 短期记忆：内存字典模拟 Redis（会话级症状链，TTL 30min）
  接口与 redis-py 完全一致，装了 Redis 后只需替换 _ShortTermCache 实现
"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from solutions.llm import llm


# ── 短期缓存（内存模拟 Redis，TTL 秒） ─────────────────────────
class _ShortTermCache:
    """线程安全的内存 KV 缓存，支持 TTL，接口对齐 redis-py"""

    def __init__(self, default_ttl: int = 1800):
        self._store: dict[str, tuple[str, float]] = {}  # key → (value, expire_at)
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def set(self, key: str, value: str, ex: int = None) -> None:
        ttl = ex or self.default_ttl
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expire_at = item
            if time.time() > expire_at:
                del self._store[key]
                return None
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


# 若已安装 Redis 且服务在运行，替换为：
# import redis
# _cache = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
_cache = _ShortTermCache(default_ttl=1800)


# ── LLM 摘要 Prompt ────────────────────────────────────────────
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是医疗记录整理助手。将患者的问诊历史压缩为一段简洁的健康摘要。
要求：
1. 保留关键信息：慢性病、过敏史、常用药、近期症状
2. 不超过150字
3. 用第三人称描述，例如"患者为35岁男性..."
4. 只输出摘要文本，不要有其他内容"""),
    ("user", "患者健康档案：\n{profile}\n\n近期问诊记录：\n{history}")
])
_summary_chain = _SUMMARY_PROMPT | llm | StrOutputParser()


# ── 长期记忆管理器 ─────────────────────────────────────────────
class MemoryManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data" / "memory"
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        return self._dir / f"{user_id}.json"

    def _default_record(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "health_profile": {
                "性别": None,
                "年龄": None,
                "慢性病": [],
                "过敏史": [],
                "常用药": [],
                "禁忌药": [],
            },
            "consultation_history": [],
            "llm_summary": "",
            "statistics": {
                "total_consultations": 0,
                "frequent_symptoms": {}
            }
        }

    # ── 读写长期记忆 ───────────────────────────────────────────

    def load(self, user_id: str) -> dict:
        path = self._path(user_id)
        if not path.exists():
            return self._default_record(user_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, user_id: str, record: dict) -> None:
        record["updated_at"] = datetime.now().isoformat()
        with open(self._path(user_id), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def get_health_profile(self, user_id: str) -> dict:
        return self.load(user_id)["health_profile"]

    def get_summary(self, user_id: str) -> str:
        return self.load(user_id).get("llm_summary", "")

    def update_health_profile(self, user_id: str, updates: dict) -> None:
        """更新健康档案，支持追加列表字段"""
        record = self.load(user_id)
        profile = record["health_profile"]
        for key, value in updates.items():
            if key in profile and isinstance(profile[key], list):
                # 列表字段追加去重
                if isinstance(value, list):
                    for v in value:
                        if v and v not in profile[key]:
                            profile[key].append(v)
                elif value and value not in profile[key]:
                    profile[key].append(value)
            else:
                profile[key] = value
        record["health_profile"] = profile
        self.save(user_id, record)

    def add_consultation(self, user_id: str, consultation: dict) -> None:
        """追加一条问诊记录"""
        record = self.load(user_id)
        consultation["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        record["consultation_history"].append(consultation)
        # 只保留最近50条
        record["consultation_history"] = record["consultation_history"][-50:]
        # 更新统计
        record["statistics"]["total_consultations"] += 1
        for symptom in consultation.get("symptoms", []):
            freq = record["statistics"]["frequent_symptoms"]
            freq[symptom] = freq.get(symptom, 0) + 1
        self.save(user_id, record)

    def refresh_summary(self, user_id: str) -> str:
        """同步刷新 LLM 摘要（问诊结束后调用）"""
        record = self.load(user_id)
        profile = record["health_profile"]
        history = record["consultation_history"][-10:]  # 最近10条
        if not any(profile.values()) and not history:
            return ""
        try:
            summary = _summary_chain.invoke({
                "profile": json.dumps(profile, ensure_ascii=False),
                "history": json.dumps(history, ensure_ascii=False, indent=2)
            })
            record["llm_summary"] = summary
            self.save(user_id, record)
            return summary
        except Exception:
            return record.get("llm_summary", "")

    def async_refresh_summary(self, user_id: str) -> None:
        """异步刷新摘要，不阻塞主流程，用线程池确保写入完整"""
        def _run():
            try:
                self.refresh_summary(user_id)
            except Exception:
                pass  # 摘要刷新失败不影响主流程
        # 非 daemon 线程：进程退出前会等待写入完成，防止文件损坏
        t = threading.Thread(target=_run, daemon=False)
        t.start()

    # ── 短期记忆（会话级） ─────────────────────────────────────

    def get_session(self, session_id: str) -> dict:
        """读取当前会话状态"""
        cache_key = f"session:{session_id}"
        raw = _cache.get(cache_key)
        if raw is None:
            return {"symptoms": [], "intents_chain": [], "round": 0}
        return json.loads(raw)

    def update_session(self, session_id: str, updates: dict) -> None:
        """更新会话状态，TTL 30分钟"""
        cache_key = f"session:{session_id}"
        current = self.get_session(session_id)
        current.update(updates)
        _cache.set(cache_key, json.dumps(current, ensure_ascii=False), ex=1800)

    def clear_session(self, session_id: str) -> None:
        _cache.delete(f"session:{session_id}")

    # ── 缓存命中率统计 ─────────────────────────────────────────

    _hit = 0
    _miss = 0

    def get_summary_cached(self, user_id: str) -> tuple[str, bool]:
        """
        带缓存的摘要读取，优先从短期缓存取
        返回 (summary, is_cache_hit)
        """
        cache_key = f"summary:{user_id}"
        cached = _cache.get(cache_key)
        if cached:
            MemoryManager._hit += 1
            return cached, True
        MemoryManager._miss += 1
        summary = self.get_summary(user_id)
        if summary:
            _cache.set(cache_key, summary, ex=3600)  # 摘要缓存1小时
        return summary, False

    @classmethod
    def cache_hit_rate(cls) -> float:
        total = cls._hit + cls._miss
        return round(cls._hit / total * 100, 1) if total > 0 else 0.0


# 全局单例
memory_manager = MemoryManager()
