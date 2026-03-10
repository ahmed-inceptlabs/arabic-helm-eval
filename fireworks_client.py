from typing import Any, Dict, List

from helm.clients.openai_client import OpenAIClient
from helm.common.request import Request

# Default system prompt for MCQ benchmarks — ensures the model outputs only the
# answer letter, preventing HELM's regex from matching letters inside repeated text.
_MCQ_SYSTEM_PROMPT = (
    "أجب عن أسئلة الاختيار من متعدد بحرف الإجابة فقط (أ، ب، ج، د، هـ) دون أي شرح أو تكرار للسؤال."
)


class FireworksNoThinkingClient(OpenAIClient):
    """OpenAI-compatible client for Fireworks AI that always disables thinking.

    Args:
        system_prompt: Optional system prompt to prepend. Defaults to the Arabic
            MCQ prompt. Pass an empty string to suppress for generation tasks.
    """

    def __init__(self, *args, system_prompt: str = _MCQ_SYSTEM_PROMPT, **kwargs):
        super().__init__(*args, **kwargs)
        self._system_prompt = system_prompt

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"

        if self._system_prompt:
            messages: List[Dict[str, Any]] = raw_request.get("messages", [])
            if messages and not any(m.get("role") == "system" for m in messages):
                raw_request["messages"] = [
                    {"role": "system", "content": self._system_prompt}
                ] + messages

        return raw_request
