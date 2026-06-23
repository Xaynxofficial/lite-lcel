import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import tool, MockLLM, AgentExecutor, ConsoleCallbackHandler

@tool
def topla(a: int, b: int) -> int:
    """İki tam sayıyı matematiksel olarak toplar."""
    return a + b


@tool
def hava_durumu_sorgula(sehir: str) -> str:
    """Belirtilen şehir için güncel hava durumunu getirir."""
    hava_verisi = {
        "istanbul": "güneşli, 28°C",
        "ankara": "bulutlu, 22°C",
        "izmir": "rüzgarlı, 26°C"
    }
    sehir_clean = sehir.lower().replace("ı", "i").replace("i̇", "i")
    return hava_verisi.get(sehir_clean, "bilinmiyor (veri bulunamadı)")


def run_autonomous_agent():
    print("=== Otonom Karar Döngüsüne Sahip Ajan Örneği (ReAct Ajanı) ===")
    
    # 1. Model tanımla
    model = MockLLM(model_name="Autonomous-GPT-4", delay=0.3)
    
    # 2. Araç listesini tanımla
    tools = [topla, hava_durumu_sorgula]
    
    # 3. AgentExecutor'ı başlat (en fazla 5 döngü yapabilsin)
    agent = AgentExecutor(model=model, tools=tools, max_iterations=5)
    
    # Detaylı adım izleme için Callbacks ekleyelim
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    # Ajanın otonom olarak iki aracı sırayla çağırıp çözmesi gereken karmaşık soru:
    soru = "Lütfen İstanbul için hava durumu sorgula ve ayrıca 5 ile 10 sayılarının toplamı kaçtır?"
    print(f"\nKullanıcı Sorusu: {soru}")
    print("="*60)
    
    # Ajanı çalıştır
    nihai_cevap = agent.invoke({"soru": soru}, config=config)
    
    print("="*60)
    print(f"\n[Ajanın Nihai Yanıtı]:\n{nihai_cevap}")


if __name__ == "__main__":
    run_autonomous_agent()
