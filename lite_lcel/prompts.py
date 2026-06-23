from typing import Any, Dict, List, Tuple, Union
from lite_lcel.base import Runnable

class BaseMessage:
    """
    Dil modellerine gönderilen mesajlar için temel sınıf.
    """
    def __init__(self, content: str):
        self.content = content

    @property
    def type(self) -> str:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={repr(self.content)})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseMessage):
            return False
        return self.type == other.type and self.content == other.content


class SystemMessage(BaseMessage):
    """Sistem yönlendirme mesajı."""
    @property
    def type(self) -> str:
        return "system"


class HumanMessage(BaseMessage):
    """Kullanıcı girdi mesajı."""
    @property
    def type(self) -> str:
        return "human"


class AIMessage(BaseMessage):
    """
    Dil modeli yanıt mesajı.
    Model fonksiyon çağrıları (tool_calls) talep ederse bunları da barındırır.
    """
    def __init__(self, content: str, tool_calls: List[Dict[str, Any]] = None):
        super().__init__(content)
        self.tool_calls = tool_calls or []

    @property
    def type(self) -> str:
        return "ai"

    def __repr__(self) -> str:
        if self.tool_calls:
            return f"AIMessage(content={repr(self.content)}, tool_calls={self.tool_calls})"
        return super().__repr__()


class PromptTemplate(Runnable):
    """
    Standart metin tabanlı şablon.
    """
    def __init__(self, template: str):
        self.template = template

    def _invoke(self, input_data: dict, config: Dict[str, Any] = None) -> str:
        if not isinstance(input_data, dict):
            raise ValueError("PromptTemplate girdisi bir sözlük (dict) olmalıdır.")
        return self.template.format(**input_data)

    async def _ainvoke(self, input_data: dict, config: Dict[str, Any] = None) -> str:
        return self._invoke(input_data, config)


class ChatPromptTemplate(Runnable):
    """
    Modern sohbet modelleri için mesaj dizisi şablonu.
    """
    def __init__(self, messages: List[Union[BaseMessage, Tuple[str, str]]]):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages: List[Union[BaseMessage, Tuple[str, str]]]) -> "ChatPromptTemplate":
        return cls(messages)

    def _invoke(self, input_data: dict, config: Dict[str, Any] = None) -> List[BaseMessage]:
        if not isinstance(input_data, dict):
            raise ValueError("ChatPromptTemplate girdisi bir sözlük (dict) olmalıdır.")
            
        formatted_messages = []
        for msg in self.messages:
            if isinstance(msg, BaseMessage):
                msg_class = msg.__class__
                formatted_messages.append(msg_class(content=msg.content.format(**input_data)))
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content_tmpl = msg
                formatted_content = content_tmpl.format(**input_data)
                
                if role == "system":
                    formatted_messages.append(SystemMessage(content=formatted_content))
                elif role in ("human", "user"):
                    formatted_messages.append(HumanMessage(content=formatted_content))
                elif role in ("ai", "assistant"):
                    formatted_messages.append(AIMessage(content=formatted_content))
                else:
                    raise ValueError(f"Geçersiz rol tanımlandı: {role}. Sadece system, human, user, ai, assistant desteklenir.")
            else:
                raise TypeError(f"Bilinmeyen mesaj formatı: {type(msg)}")
                
        return formatted_messages

    async def _ainvoke(self, input_data: dict, config: Dict[str, Any] = None) -> List[BaseMessage]:
        return self._invoke(input_data, config)
