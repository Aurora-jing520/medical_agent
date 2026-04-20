"""
LazyAgentRegistry — Skill 插件化核心
启动时只扫描 SKILL.md 元数据，首次调用时按需加载 agent.py 并缓存
"""
import importlib
import time
from pathlib import Path


class LazyAgentRegistry:
    def __init__(self):
        # 元数据索引：{intent: {name, description, module_path, skill_dir}}
        self._metadata: dict[str, dict] = {}
        # Agent 实例缓存：{intent: agent_instance}
        self._cache: dict[str, any] = {}
        self._scan_time: float = 0.0

    def scan(self, skills_dir: str | Path = None) -> None:
        """
        启动时调用一次，只扫描 SKILL.md，不 import 任何 agent.py
        """
        t0 = time.perf_counter()
        if skills_dir is None:
            skills_dir = Path(__file__).parent / "skills"
        skills_dir = Path(skills_dir)

        for skill_dir in sorted(skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not skill_md.exists():
                continue
            meta = self._parse_skill_md(skill_md)
            if meta:
                intent = meta["intent"]
                meta["skill_dir"] = str(skill_dir)
                meta["module_path"] = f"solutions.langgraph_medical.skills.{skill_dir.name}.agent"
                self._metadata[intent] = meta

        self._scan_time = time.perf_counter() - t0

    def _parse_skill_md(self, path: Path) -> dict | None:
        """解析 SKILL.md 中的 YAML frontmatter"""
        text = path.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        # 必须以 --- 开头才是合法 frontmatter
        if not lines or lines[0].strip() != "---":
            return None
        meta = {}
        in_front = False
        for line in lines:
            if line.strip() == "---":
                in_front = not in_front
                continue
            if in_front and ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
        return meta if "intent" in meta else None

    def get(self, intent: str):
        """
        按需加载并缓存 Agent 实例
        第一次调用时 import agent.py，之后直接返回缓存
        """
        if intent in self._cache:
            return self._cache[intent]

        if intent not in self._metadata:
            return None

        module_path = self._metadata[intent]["module_path"]
        module = importlib.import_module(module_path)
        agent = module.create_agent()
        self._cache[intent] = agent
        return agent

    def list_intents(self) -> list[str]:
        return list(self._metadata.keys())

    def get_metadata(self, intent: str) -> dict:
        return self._metadata.get(intent, {})

    def get_all_descriptions(self) -> str:
        """给 IntentAgent 用：返回所有意图的描述，用于 prompt"""
        lines = []
        for intent, meta in self._metadata.items():
            lines.append(f"- {intent}: {meta.get('description', '')}")
        return "\n".join(lines)

    @property
    def scan_time_ms(self) -> float:
        return round(self._scan_time * 1000, 2)


# 全局单例，import 后直接用
registry = LazyAgentRegistry()
