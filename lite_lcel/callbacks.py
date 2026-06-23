from typing import Any, Dict

class BaseCallbackHandler:
    """
    Kütüphanedeki Runnable ve Model olaylarını dinleyen temel Callback sınıfı.
    """
    def on_chain_start(self, name: str, input_data: Any, config: Dict[str, Any] = None) -> None:
        pass

    def on_chain_end(self, name: str, output: Any, config: Dict[str, Any] = None) -> None:
        pass

    def on_chain_error(self, name: str, error: Exception, config: Dict[str, Any] = None) -> None:
        pass

    def on_llm_start(self, name: str, prompt: str, config: Dict[str, Any] = None) -> None:
        pass

    def on_llm_end(self, name: str, response: str, config: Dict[str, Any] = None) -> None:
        pass

    def on_llm_error(self, name: str, error: Exception, config: Dict[str, Any] = None) -> None:
        pass


class ConsoleCallbackHandler(BaseCallbackHandler):
    """
    Olayları konsola şık bir formatta yazdıran izleme yöneticisi.
    """
    def on_chain_start(self, name: str, input_data: Any, config: Dict[str, Any] = None) -> None:
        print(f"\n⚡ [Chain Başladı] => Sınıf: {name}")
        print(f"   ↳ Girdi: {input_data}")

    def on_chain_end(self, name: str, output: Any, config: Dict[str, Any] = None) -> None:
        print(f"✅ [Chain Bitti] => Sınıf: {name}")
        print(f"   ↳ Çıktı: {output}")

    def on_chain_error(self, name: str, error: Exception, config: Dict[str, Any] = None) -> None:
        print(f"❌ [Chain Hatası] => Sınıf: {name}")
        print(f"   ↳ Hata: {type(error).__name__}: {error}")

    def on_llm_start(self, name: str, prompt: str, config: Dict[str, Any] = None) -> None:
        print(f"\n🤖 [LLM Başladı] => Model: {name}")
        print(f"   ↳ Prompt: {repr(prompt.strip()[:100])}...")

    def on_llm_end(self, name: str, response: str, config: Dict[str, Any] = None) -> None:
        print(f"✨ [LLM Bitti] => Model: {name}")
        print(f"   ↳ Yanıt: {repr(response.strip()[:100])}...")

    def on_llm_error(self, name: str, error: Exception, config: Dict[str, Any] = None) -> None:
        print(f"💥 [LLM Hatası] => Model: {name}")
        print(f"   ↳ Hata: {error}")
