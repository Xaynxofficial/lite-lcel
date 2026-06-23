import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    RunnableBranch,
    RunnableLambda,
    ConsoleCallbackHandler
)

def run_branch_routing_example():
    print("=== 1. RunnableBranch ile Dinamik Koşullu Yönlendirme ===")
    
    model = MockLLM(delay=0.1)
    parser = StringOutputParser()
    
    # 3 farklı alt zincir kurgulayalım
    science_chain = PromptTemplate("Fen Bilgisi Sorusu: {soru}") | model | parser
    coding_chain = PromptTemplate("Yazılım/Programlama Sorusu: {soru}") | model | parser
    general_chain = PromptTemplate("Genel Soru: {soru}") | model | parser
    
    # Branş yönlendiricisini tanımla
    # Tuple yapısı: (koşul_fonksiyonu, çalıştırılacak_zincir)
    routing_branch = RunnableBranch(
        (lambda x: "gökyüzü" in x["soru"].lower(), science_chain),
        (lambda x: "python" in x["soru"].lower() or "kod" in x["soru"].lower(), coding_chain),
        general_chain  # default zincir
    )
    
    # Test 1: Fen Bilgisi Yönlendirmesi
    girdi_1 = {"soru": "Gökyüzü neden mavidir?"}
    print(f"\nGirdi: {girdi_1}")
    sonuc_1 = routing_branch.invoke(girdi_1)
    print(f"Çıktı: {sonuc_1}")
    
    # Test 2: Yazılım Yönlendirmesi
    girdi_2 = {"soru": "Python nedir?"}
    print(f"\nGirdi: {girdi_2}")
    sonuc_2 = routing_branch.invoke(girdi_2)
    print(f"Çıktı: {sonuc_2}")
    
    # Test 3: Genel Yönlendirme
    girdi_3 = {"soru": "Bugün hava nasıl?"}
    print(f"\nGirdi: {girdi_3}")
    sonuc_3 = routing_branch.invoke(girdi_3)
    print(f"Çıktı: {sonuc_3}")
    print("\n" + "="*50 + "\n")


def run_fallback_example():
    print("=== 2. RunnableWithFallbacks ile Hata Kurtarma (Fallback) ===")
    
    # Bir hata fırlatan kararsız/bozuk zincir simülasyonu
    def unstable_api_call(x):
        print("⚠️  [Birincil Model] Sunucu hatası oluştu! API çağrısı başarısız.")
        raise ConnectionError("OpenAI API sunucusuna erişilemiyor.")
        
    primary_chain = RunnableLambda(unstable_api_call)
    
    # Yedek (backup) çalışan zincir
    backup_model = MockLLM(model_name="Yedek-Mock-Model", delay=0.1)
    backup_chain = PromptTemplate("Yedek Model Sorusu: {soru}") | backup_model | StringOutputParser()
    
    # Fallback zinciri oluşturuluyor
    # Eğer primary_chain ConnectionError verirse otomatik olarak backup_chain'e geçecek
    robust_chain = primary_chain.with_fallbacks([backup_chain])
    
    # Callback'ler ile izleyelim
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    girdi = {"soru": "Python nedir?"}
    print(f"Sorgu gönderiliyor: {girdi}\n")
    
    # Çalıştır (Hata vermeden yedek zincire geçmesi gerekir)
    sonuc = robust_chain.invoke(girdi, config=config)
    
    print(f"\n[Nihai Sonuç (Hata Kurtarma Sonrası)]:\n{sonuc}")


if __name__ == "__main__":
    run_branch_routing_example()
    run_fallback_example()
