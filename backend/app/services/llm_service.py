"""Environment-driven LLM provider abstraction."""

try:
    from backend.app.core.config import settings
except ImportError:
    from app.core.config import settings


class LLMService:
    def __init__(self):
        self._llm = None

    @property
    def configured(self) -> bool:
        provider = getattr(settings, "LLM_PROVIDER", "gemini").lower()

        if provider == "gemini":
            return bool(getattr(settings, "GOOGLE_API_KEY", ""))

        if provider == "openai":
            return bool(getattr(settings, "OPENAI_API_KEY", ""))

        return False

    def get_llm(self):
        provider = getattr(settings, "LLM_PROVIDER", "gemini").lower()

        if not self.configured:
            raise RuntimeError(
                f"LLM is not configured for provider '{provider}'. Check your .env file."
            )

        if self._llm is None:
            if provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm = ChatGoogleGenerativeAI(
                    google_api_key=settings.GOOGLE_API_KEY,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                )

            elif provider == "openai":
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                )

            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER: {provider}"
                )

        return self._llm

    def invoke_text(self, prompt: str) -> str:
        response = self.get_llm().invoke(prompt)
        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)

            return "\n".join(text_parts).strip()

        return str(content).strip()


llm_service = LLMService()