from typing import List
from .prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from ..config import OPENAI_API_KEY, MODEL_NAME, PROVIDER, OLLAMA_BASE_URL

class ProposalAgent:
    def __init__(self):
        self.provider = PROVIDER.lower()
        self.model = MODEL_NAME
        if self.provider == "openai":
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "ollama":
            # Lazy import to avoid hard dependency if not used
            from langchain_community.chat_models import ChatOllama
            self.client = ChatOllama(model=self.model, base_url=OLLAMA_BASE_URL, temperature=0.2)
        else:
            raise RuntimeError(f"Unsupported PROVIDER: {self.provider}")

    def generate(self, client_name: str, rfq_title: str, context_chunks: List[str]) -> str:
        context_text = "\n\n---\n\n".join(context_chunks[:8]) if context_chunks else "(no prior context)"
        user_prompt = USER_PROMPT_TEMPLATE.format(client=client_name, rfq_title=rfq_title, context=context_text)

        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        else:
            # Ollama via LangChain interface
            msg = self.client.invoke([
                ("system", SYSTEM_PROMPT),
                ("user", user_prompt),
            ])
            return getattr(msg, "content", str(msg))
