import json
import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import tool, MockLLM

# 1. Python fonksiyonumuzu `@tool` dekoratörüyle sarmalayalım
@tool
def topla(a: int, b: int) -> int:
    """İki tam sayıyı matematiksel olarak toplar."""
    return a + b


@tool
def hava_durumu_sorgula(sehir: str) -> str:
    """Belirtilen şehir için güncel hava durumunu getirir."""
    # Simüle edilmiş bir hava durumu veri tabanı
    hava_verisi = {
        "istanbul": "güneşli, 28°C",
        "ankara": "bulutlu, 22°C",
        "izmir": "rüzgarlı, 26°C"
    }
    # Türkçe büyük İ harfinin küçük i'ye doğru eşleşmesi için temizlik yapalım
    sehir_clean = sehir.lower().replace("ı", "i").replace("i̇", "i")
    return hava_verisi.get(sehir_clean, "bilinmiyor (veri bulunamadı)")


def run_tool_calling_example():
    print("=== @tool Dekoratörü ve Fonksiyon Şeması Çıkarımı ===")
    
    # Otomatik üretilen şemayı inceleyelim (OpenAI'ın beklediği format)
    print("\n'topla' Fonksiyonunun JSON Şeması:")
    print(json.dumps(topla.to_openai_tool(), indent=2, ensure_ascii=False))
    
    print("\n'hava_durumu_sorgula' Fonksiyonunun JSON Şeması:")
    print(json.dumps(hava_durumu_sorgula.to_openai_tool(), indent=2, ensure_ascii=False))
    
    print("\n" + "="*50 + "\n")
    print("=== Model Tool Binding (Araç Bağlama) & Çalıştırma ===")
    
    # Modelimizi kurgulayalım
    model = MockLLM(model_name="Tool-GPT-4", delay=0.2)
    
    # Araçları modele bağlayalım
    model_with_tools = model.bind_tools([topla, hava_durumu_sorgula])
    
    # Modelden araç çağrısı talep edecek bir prompt gönderelim
    soru = "Lütfen İstanbul şehri için hava durumu sorgula."
    print(f"Kullanıcı: {soru}")
    
    # Çalıştır
    yanit = model_with_tools.invoke(soru)
    
    print(f"\nModel Yanıtı: {yanit}")
    
    # Model araç çağırmak istedi mi?
    if yanit.tool_calls:
        print("\n🤖 Dil modeli araç çalıştırma (tool call) talebinde bulundu!")
        for i, call in enumerate(yanit.tool_calls):
            tool_name = call["function"]["name"]
            tool_args = json.loads(call["function"]["arguments"])
            
            print(f"  [{i+1}] Talep Edilen Araç: {tool_name}")
            print(f"      ↳ Argümanlar: {tool_args}")
            
            # Aracı bulup çalıştıralım (Tool Execution)
            if tool_name == "topla":
                sonuc = topla.invoke(tool_args)
            elif tool_name == "hava_durumu_sorgula":
                sonuc = hava_durumu_sorgula.invoke(tool_args)
            else:
                sonuc = "Bilinmeyen araç"
                
            print(f"      ✅ Araç Çalıştırma Sonucu: {sonuc}")


if __name__ == "__main__":
    run_tool_calling_example()
