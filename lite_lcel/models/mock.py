import asyncio
import json
import time
from typing import Any, Dict, List, Union, Generator, AsyncGenerator
from lite_lcel.models.base import BaseLanguageModel
from lite_lcel.prompts import BaseMessage, AIMessage

class MockLLM(BaseLanguageModel):
    """
    Kural tabanlı yanıtlar üreten ve gecikme simülasyonu yapan Mock Dil Modeli.
    Streaming (akış), Callbacks (olay izleme) ve Tool Calling (fonksiyon çağırma) altyapılarını destekler.
    """
    def __init__(self, model_name: str = "mock-lcel-gpt", delay: float = 0.5):
        self.model_name = model_name
        self.delay = delay

    def _invoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        config = config or {}
        prompt_str = self._convert_input_to_string(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_llm_start(self.model_name, prompt_str, config)
            
        time.sleep(self.delay)
        
        # Eğer modele bağlı araçlar varsa ve prompt içinde araç ismi geçiyorsa araç çağrısı simüle et
        if self.bound_tools:
            prompt_lower = prompt_str.lower()
            for t in self.bound_tools:
                # Eğer bu araç daha önce çalıştırılıp sonucu sisteme verildiyse, tekrar çağırma
                already_run = f"tool '{t.name.lower()}' returned:" in prompt_lower
                if not already_run:
                    normalized_tool_name = t.name.lower().replace("_", " ")
                    if t.name.lower() in prompt_lower or normalized_tool_name in prompt_lower:
                        # Simüle argümanlar
                        mock_args = {}
                        if "topla" in t.name:
                            mock_args = {"a": 5, "b": 10}
                        elif "hava" in t.name or "weather" in t.name:
                            mock_args = {"sehir": "İstanbul"}
                        else:
                            # Şemaya göre otomatik doldur
                            for k, v in t.args_schema.items():
                                mock_args[k] = 42 if v.get("type") in ("integer", "number") else "test_val"
                                
                        tool_call = {
                            "id": "call_mock_12345",
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "arguments": json.dumps(mock_args)
                            }
                        }
                        
                        for cb in callbacks:
                            cb.on_llm_end(self.model_name, f"[Tool Call Request: {t.name}]", config)
                            
                        return AIMessage(content="", tool_calls=[tool_call])
        
        response_text = self._generate_response(prompt_str)
        
        for cb in callbacks:
            cb.on_llm_end(self.model_name, response_text, config)
            
        return AIMessage(content=response_text)

    async def _ainvoke(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AIMessage:
        config = config or {}
        prompt_str = self._convert_input_to_string(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_llm_start(self.model_name, prompt_str, config)
            
        await asyncio.sleep(self.delay)
        
        if self.bound_tools:
            prompt_lower = prompt_str.lower()
            for t in self.bound_tools:
                already_run = f"tool '{t.name.lower()}' returned:" in prompt_lower
                if not already_run:
                    normalized_tool_name = t.name.lower().replace("_", " ")
                    if t.name.lower() in prompt_lower or normalized_tool_name in prompt_lower:
                        mock_args = {}
                        if "topla" in t.name:
                            mock_args = {"a": 5, "b": 10}
                        elif "hava" in t.name or "weather" in t.name:
                            mock_args = {"sehir": "İstanbul"}
                        else:
                            for k, v in t.args_schema.items():
                                mock_args[k] = 42 if v.get("type") in ("integer", "number") else "test_val"
                                
                        tool_call = {
                            "id": "call_mock_12345",
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "arguments": json.dumps(mock_args)
                            }
                        }
                        
                        for cb in callbacks:
                            cb.on_llm_end(self.model_name, f"[Tool Call Request: {t.name}]", config)
                            
                        return AIMessage(content="", tool_calls=[tool_call])
        
        response_text = self._generate_response(prompt_str)
        
        for cb in callbacks:
            cb.on_llm_end(self.model_name, response_text, config)
            
        return AIMessage(content=response_text)

    def _stream(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> Generator[AIMessage, None, None]:
        config = config or {}
        prompt_str = self._convert_input_to_string(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_llm_start(self.model_name, prompt_str, config)
            
        response_text = self._generate_response(prompt_str)
        words = response_text.split(" ")
        
        chunk_delay = self.delay / max(len(words), 1)
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            time.sleep(chunk_delay)
            yield AIMessage(content=chunk_content)
            
        for cb in callbacks:
            cb.on_llm_end(self.model_name, response_text, config)

    async def _astream(self, input_data: Union[str, List[BaseMessage]], config: Dict[str, Any] = None) -> AsyncGenerator[AIMessage, None]:
        config = config or {}
        prompt_str = self._convert_input_to_string(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_llm_start(self.model_name, prompt_str, config)
            
        response_text = self._generate_response(prompt_str)
        words = response_text.split(" ")
        
        chunk_delay = self.delay / max(len(words), 1)
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(chunk_delay)
            yield AIMessage(content=chunk_content)
            
        for cb in callbacks:
            cb.on_llm_end(self.model_name, response_text, config)

    def _generate_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        # Eğer geçmişte araç sonuçları varsa, bunları birleştirip nihai cevabı dön
        if "returned:" in prompt_lower:
            if "15" in prompt_lower and "güneşli" in prompt_lower:
                return "İstanbul'da hava şu anda güneşli ve sıcaklık 28°C'dir. Ayrıca 5 ile 10 sayılarının toplamı 15'tir."
            elif "15" in prompt_lower:
                return "Toplama işleminin sonucu 15'tir."
            elif "güneşli" in prompt_lower:
                return "İstanbul'da hava şu anda güneşli ve sıcaklık 28°C'dir."
        
        # Sohbet geçmişinden etkilenmemek için yalnızca son satırdaki (güncel) soruyu analiz et
        lines = [line.strip() for line in prompt_lower.split("\n") if line.strip()]
        current_question = lines[-1] if lines else prompt_lower
        
        if "json" in current_question or "formatında" in current_question or "sözlük" in current_question:
            return """
```json
{
  "status": "success",
  "data": {
    "dil": "Python",
    "kullanim": "Yapay Zeka ve Web",
    "zorluk": "Kolay"
  },
  "mesaj": "Simüle edilmiş JSON cevabıdır."
}
```
"""
        if "gökyüzü" in current_question:
            return "Gökyüzü mavi görünür çünkü Güneş ışınları Rayleigh saçılmasına uğrar."
        elif "python" in current_question:
            return "Python okunabilirliği yüksek, esnek ve popüler bir dildir."
        elif "lcel" in current_question:
            return "LCEL (LangChain Expression Language), zincirleri kolayca tanımlamayı sağlayan deklaratif bir dildir."
            
        return f"[MockLLM - {self.model_name}] Girdi metni: '{prompt}' için yanıt."
