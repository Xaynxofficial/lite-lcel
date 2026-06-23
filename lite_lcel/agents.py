import json
import asyncio
from typing import Any, Dict, List
from lite_lcel.base import Runnable
from lite_lcel.prompts import HumanMessage
from lite_lcel.tools import StructuredTool

class AgentExecutor(Runnable):
    """
    Dil modeli ile araçları otonom bir karar döngüsünde yürüten Ajan sınıfı.
    Model nihai bir yanıt üretinceye kadar araçları çalıştırır ve geçmişi besler.
    """
    def __init__(self, model: Any, tools: List[StructuredTool], max_iterations: int = 5):
        self.model = model.bind_tools(tools)  # Araçları otomatik modele bağla
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    def _invoke(self, input_data: dict, config: Dict[str, Any] = None) -> str:
        config = config or {}
        
        # Girdiyi soru anahtarından alalım
        question = input_data.get("soru") or input_data.get("input")
        if not question:
            raise ValueError("Ajan girdisi 'soru' veya 'input' anahtarı içermelidir.")
            
        messages = [HumanMessage(content=question)]
        
        for iteration in range(self.max_iterations):
            # Modeli çağır
            response = self.model.invoke(messages, config)
            messages.append(response)
            
            # Eğer araç çağrısı yoksa, otonom döngü tamamlanmıştır, nihai cevabı dön
            if not response.tool_calls:
                return response.content
                
            # Araç çağrısı varsa yürüt
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # İlgili aracı bul
                if tool_name not in self.tools:
                    tool_result = f"Hata: '{tool_name}' isimli bir araç kütüphanede bulunamadı."
                else:
                    try:
                        tool_result = str(self.tools[tool_name].invoke(tool_args))
                    except Exception as e:
                        tool_result = f"Araç çalışırken hata fırlattı: {type(e).__name__}: {e}"
                
                # Araç sonucunu geçmişe ekle (modelin okuyabilmesi için)
                tool_msg = HumanMessage(
                    content=f"System: Tool '{tool_name}' returned: {tool_result}"
                )
                messages.append(tool_msg)
                
        return "[Ajan Hatası] Maksimum yineleme sınırına ulaşıldı ve nihai cevap üretilemedi."

    async def _ainvoke(self, input_data: dict, config: Dict[str, Any] = None) -> str:
        config = config or {}
        question = input_data.get("soru") or input_data.get("input")
        if not question:
            raise ValueError("Ajan girdisi 'soru' veya 'input' anahtarı içermelidir.")
            
        messages = [HumanMessage(content=question)]
        
        for iteration in range(self.max_iterations):
            response = await self.model.ainvoke(messages, config)
            messages.append(response)
            
            if not response.tool_calls:
                return response.content
                
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                if tool_name not in self.tools:
                    tool_result = f"Hata: '{tool_name}' isimli bir araç kütüphanede bulunamadı."
                else:
                    try:
                        # Asenkron araç çalıştırma simülasyonu
                        loop = asyncio.get_running_loop()
                        tool_result = str(await loop.run_in_executor(None, self.tools[tool_name].invoke, tool_args))
                    except Exception as e:
                        tool_result = f"Araç çalışırken hata fırlattı: {type(e).__name__}: {e}"
                
                tool_msg = HumanMessage(
                    content=f"System: Tool '{tool_name}' returned: {tool_result}"
                )
                messages.append(tool_msg)
                
        return "[Ajan Hatası] Maksimum yineleme sınırına ulaşıldı ve nihai cevap üretilemedi."
