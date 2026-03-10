from typing import Any, Dict, List

from helm.clients.openai_client import OpenAIClient
from helm.common.request import Request

# System prompt to ensure the model outputs only the answer letter,
# preventing the regex from matching letters inside repeated Arabic words.
_MCQ_SYSTEM_PROMPT = (
    "أجب عن أسئلة الاختيار من متعدد بحرف الإجابة فقط (أ، ب، ج، د، هـ) دون أي شرح أو تكرار للسؤال."
)


class FireworksNoThinkingClient(OpenAIClient):
    """OpenAI-compatible client for Fireworks AI that always disables thinking."""

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"

        # Prepend a system message so the model responds with just the answer letter.
        messages: List[Dict[str, Any]] = raw_request.get("messages", [])
        if messages and not any(m.get("role") == "system" for m in messages):
            raw_request["messages"] = [{"role": "system", "content": _MCQ_SYSTEM_PROMPT}] + messages

        return raw_request
