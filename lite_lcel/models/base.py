from typing import Any, Dict, List, Union
import copy
from lite_lcel.base import Runnable
from lite_lcel.prompts import BaseMessage, AIMessage

class BaseLanguageModel(Runnable):
    """
    Dil Modelleri için soyut temel sınıf.
    Hem metin (str) girdilerini hem de sohbet mesaj listelerini (List[BaseMessage]) destekler.
    Çıktı olarak her zaman bir AIMessage nesnesi döndürür.
    """
    bound_tools: List[Any] = []

    def bind_tools(self, tools: List[Any]) -> "BaseLanguageModel":
        """
        Modele fonksiyon/araç bağlar (Tool Binding).
        Geriye araçları belleğinde barındıran kopyalanmış yeni bir model döner.
        """
        from lite_lcel.tools import StructuredTool, tool
        
        bound = []
        for t in tools:
            if isinstance(t, StructuredTool):
                bound.append(t)
            elif callable(t):
                bound.append(tool(t))
            else:
                raise TypeError(f"Desteklenmeyen araç tipi: {type(t)}")
        
        new_model = copy.copy(self)
        new_model.bound_tools = bound
        return new_model

    def _invoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        raise NotImplementedError("_invoke metodu alt sınıfta tanımlanmalıdır.")

    async def _ainvoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        return await super()._ainvoke(input_data, config)

    def _convert_input_to_string(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """
        Gelen girdiyi düz bir metin (string) formatına çeviren yardımcı metot.
        """
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list):
            lines = []
            for msg in input_data:
                if isinstance(msg, BaseMessage):
                    lines.append(f"{msg.type.capitalize()}: {msg.content}")
                else:
                    lines.append(str(msg))
            return "\n".join(lines)
        return str(input_data)
