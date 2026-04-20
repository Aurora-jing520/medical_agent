from solutions.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友善的医疗助手小医，回答用户的问候或非医疗问题，保持温暖简洁。"),
    ("user", "{input}")
])
_chain = _prompt | llm | StrOutputParser()


def create_agent():
    def run(state: dict) -> str:
        return _chain.invoke({"input": state.get("user_input", "")})
    return run
