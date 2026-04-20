from solutions.llm import llm
from solutions.DetectLLM import SmartDetect
from langchain_core.output_parsers import StrOutputParser

_parser = StrOutputParser()


def create_agent():
    def run(state: dict) -> str:
        user_input = state.get("user_input", "")
        return _parser.invoke(SmartDetect.invoke({"input": user_input}))
    return run
