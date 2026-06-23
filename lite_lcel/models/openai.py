import os
from typing import Any, Dict, List, Union
from lite_lcel.models.base import BaseLanguageModel
from lite_lcel.prompts import BaseMessage, AIMessage

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAILLM(BaseLanguageModel):
    """
    Gerçek OpenAI Chat Completion API'sini çağıran sınıf.
    Kullanım için 'openai' kütüphanesinin kurulu olması ve 'OPENAI_API_KEY'
    çevre değişkeninin tanımlı olması gerekir.
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        temperature: float = 0.7,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.extra_kwargs = kwargs
        
        self._client = None
        self._async_client = None

    def _get_client(self) -> "OpenAI":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI modülü bulunamadı. Lütfen 'pip install openai' ile kurun "
                "veya projeyi 'pip install -e .[openai]' ile kurun."
            )
        if not self.api_key:
            raise ValueError(
                "OpenAI API anahtarı bulunamadı. Lütfen 'OPENAI_API_KEY' çevre değişkenini tanımlayın "
                "veya sınıfı başlatırken 'api_key' parametresini sağlayın."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _get_async_client(self) -> "AsyncOpenAI":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI modülü bulunamadı. Lütfen 'pip install openai' ile kurun."
            )
        if not self.api_key:
            raise ValueError(
                "OpenAI API anahtarı bulunamadı. Lütfen 'OPENAI_API_KEY' çevre değişkenini tanımlayın."
            )
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
        """Girdiyi OpenAI API formatına (role/content) dönüştürür."""
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
            
        messages = []
        for msg in input_data:
            if isinstance(msg, BaseMessage):
                role = msg.type
                # OpenAI API rol isimleri eşleştirmesi
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append({"role": "user", "content": str(msg)})
        return messages

    def invoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        client = self._get_client()
        api_messages = self._prepare_messages(input_data)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            **self.extra_kwargs
        )
        
        content = response.choices[0].message.content
        return AIMessage(content=content)

    async def ainvoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        client = self._get_async_client()
        api_messages = self._prepare_messages(input_data)
        
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            **self.extra_kwargs
        )
        
        content = response.choices[0].message.content
        return AIMessage(content=content)
