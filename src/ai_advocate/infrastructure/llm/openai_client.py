from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class LangChainLLMAdapter:
    """Thin adapter implementing ``LLMPort`` via LangChain's ChatOpenAI."""

    def __init__(self, *, api_key: str, model: str) -> None:
        base = ChatOpenAI(api_key=api_key, model=model, temperature=0.1)
        self._model = base
        self._json_model = base.bind(response_format={"type": "json_object"})

    def invoke_json(self, system: str, user: str) -> str:
        resp = self._json_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
        )
        return str(resp.content)

    def invoke_messages(self, system: str, user: str) -> str:
        resp = self._model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(resp.content)
