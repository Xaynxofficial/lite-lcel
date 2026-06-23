from typing import Any, Callable, Dict, List, Generator, AsyncGenerator
from lite_lcel.base import Runnable
from lite_lcel.prompts import BaseMessage, HumanMessage, AIMessage

class ChatMessageHistory:
    """
    Sohbet geçmişindeki mesajları saklayan ve yöneten bellek sınıfı.
    """
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    def clear(self) -> None:
        self.messages.clear()

    def __repr__(self) -> str:
        return f"ChatMessageHistory(messages={self.messages})"


class RunnableWithMessageHistory(Runnable):
    """
    Zincirleri durumsal (stateful) hale getiren ve sohbet geçmişini
    otomatik yükleyip kaydeden Runnable sınıfı.
    
    Örnek:
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session,
            input_messages_key="soru",
            history_messages_key="gecmis"
        )
    """
    def __init__(
        self,
        run_chain: Runnable,
        get_session_history: Callable[[str], ChatMessageHistory],
        input_messages_key: str = "soru",
        history_messages_key: str = "gecmis"
    ):
        self.run_chain = run_chain
        self.get_session_history = get_session_history
        self.input_messages_key = input_messages_key
        self.history_messages_key = history_messages_key

    def _get_history(self, config: Dict[str, Any]) -> ChatMessageHistory:
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        if not session_id:
            raise ValueError(
                "RunnableWithMessageHistory çağrılırken config içinde 'configurable.session_id' belirtilmelidir.\n"
                "Örn: chain.invoke({'soru': '...'}, config={'configurable': {'session_id': 'user-1'}})"
            )
        return self.get_session_history(session_id)

    def _format_history_str(self, history: ChatMessageHistory) -> str:
        lines = []
        for msg in history.messages:
            role = "İnsan" if msg.type == "human" else "Asistan"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _invoke(self, input_data: dict, config: Dict[str, Any]) -> Any:
        history = self._get_history(config)
        
        # Giriş verisinin kopyasını al ve geçmişi ekle
        chain_input = input_data.copy()
        chain_input[self.history_messages_key] = self._format_history_str(history)
        
        # Zinciri çalıştır
        output = self.run_chain.invoke(chain_input, config)
        
        # Geçmişi güncelle (Girdi sorusunu ve modelin cevabını ekle)
        input_question = input_data[self.input_messages_key]
        output_text = output.content if hasattr(output, "content") else str(output)
        
        history.add_user_message(input_question)
        history.add_ai_message(output_text)
        
        return output

    async def _ainvoke(self, input_data: dict, config: Dict[str, Any]) -> Any:
        history = self._get_history(config)
        
        chain_input = input_data.copy()
        chain_input[self.history_messages_key] = self._format_history_str(history)
        
        output = await self.run_chain.ainvoke(chain_input, config)
        
        input_question = input_data[self.input_messages_key]
        output_text = output.content if hasattr(output, "content") else str(output)
        
        history.add_user_message(input_question)
        history.add_ai_message(output_text)
        
        return output

    def _stream(self, input_data: dict, config: Dict[str, Any]) -> Generator[Any, None, None]:
        history = self._get_history(config)
        
        chain_input = input_data.copy()
        chain_input[self.history_messages_key] = self._format_history_str(history)
        
        accumulated_text = ""
        for chunk in self.run_chain.stream(chain_input, config):
            chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            accumulated_text += chunk_text
            yield chunk
            
        # Akış bittiğinde geçmişe kaydet
        input_question = input_data[self.input_messages_key]
        history.add_user_message(input_question)
        history.add_ai_message(accumulated_text)

    async def _astream(self, input_data: dict, config: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        history = self._get_history(config)
        
        chain_input = input_data.copy()
        chain_input[self.history_messages_key] = self._format_history_str(history)
        
        accumulated_text = ""
        async for chunk in self.run_chain.astream(chain_input, config):
            chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            accumulated_text += chunk_text
            yield chunk
            
        input_question = input_data[self.input_messages_key]
        history.add_user_message(input_question)
        history.add_ai_message(accumulated_text)
