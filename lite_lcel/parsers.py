import json
import re
import inspect
from typing import Any, Dict, Union, Generator, AsyncGenerator
from lite_lcel.base import Runnable
from lite_lcel.prompts import BaseMessage

class StringOutputParser(Runnable):
    """
    LLM çıktısını alan ve temizlenmiş string üreten sınıf.
    Streaming girdi desteği (accepts_streaming_input = True) sayesinde
    zincirleme akışta (stream/astream) gelen her parçayı gerçek zamanlı iletir.
    """
    accepts_streaming_input = True

    def _invoke(self, input_data: Union[str, BaseMessage], config: Dict[str, Any] = None) -> str:
        if isinstance(input_data, BaseMessage):
            text = input_data.content
        elif isinstance(input_data, str):
            text = input_data
        else:
            text = str(input_data)
            
        return text.strip()

    def _parse_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, BaseMessage):
            return chunk.content
        return str(chunk)

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[str, None, None]:
        if inspect.isgenerator(input_data):
            for chunk in input_data:
                yield self._parse_chunk(chunk)
        else:
            yield self._invoke(input_data, config)

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        if inspect.isasyncgen(input_data):
            async for chunk in input_data:
                yield self._parse_chunk(chunk)
        else:
            yield await self.ainvoke(input_data, config)


class JsonOutputParser(Runnable):
    """
    LLM çıktısından JSON ifadesini çıkaran ve nesneye dönüştüren sınıf.
    Akış sırasında ham çıktıyı biriktirip akış bittiğinde tek bir JSON nesnesi döner.
    """
    accepts_streaming_input = True

    def _invoke(self, input_data: Union[str, BaseMessage], config: Dict[str, Any] = None) -> Union[Dict[str, Any], list]:
        if isinstance(input_data, BaseMessage):
            text = input_data.content
        elif isinstance(input_data, str):
            text = input_data
        else:
            text = str(input_data)
            
        text = text.strip()
        
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
            
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM çıktısı geçerli bir JSON formatına dönüştürülemedi.\n"
                f"Temizlenmeye çalışılan metin:\n{json_str}\n"
                f"Hata detayı: {str(e)}"
            )

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[Union[Dict[str, Any], list], None, None]:
        if inspect.isgenerator(input_data):
            accumulated = ""
            for chunk in input_data:
                if isinstance(chunk, BaseMessage):
                    accumulated += chunk.content
                else:
                    accumulated += str(chunk)
            yield self._invoke(accumulated, config)
        else:
            yield self._invoke(input_data, config)

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[Union[Dict[str, Any], list], None]:
        if inspect.isasyncgen(input_data):
            accumulated = ""
            async for chunk in input_data:
                if isinstance(chunk, BaseMessage):
                    accumulated += chunk.content
                else:
                    accumulated += str(chunk)
            yield self._invoke(accumulated, config)
        else:
            yield await self.ainvoke(input_data, config)
