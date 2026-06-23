import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

def combine_results(results: dict) -> str:
    """
    Paralel kollardan gelen sonuçları birleştiren yardımcı fonksiyon.
    """
    tanim = results["tanim"]
    avantajlar = results["avantajlar"]
    
    combined = (
        "=== BİRLEŞTİRİLMİŞ RAPOR ===\n"
        f"1. Tanım ve Açıklama:\n{tanim}\n\n"
        f"2. Avantajlar ve Detaylar:\n{avantajlar}\n"
        "==========================="
    )
    return combined


def run_parallel_chain():
    print("--- Paralel LCEL Dallanma Örneği ---")
    
    # Ortak Model
    model = MockLLM(model_name="Parallel-GPT", delay=0.4)
    parser = StringOutputParser()
    
    # 1. Kol: Tanım zinciri
    prompt_tanim = PromptTemplate(template="Şunun ne olduğunu kısaca tanımla: {konu}")
    tanim_chain = prompt_tanim | model | parser
    
    # 2. Kol: Avantaj zinciri
    prompt_avantaj = PromptTemplate(template="Şunun avantajlarını listele: {konu}")
    avantaj_chain = prompt_avantaj | model | parser
    
    # Paralel Zincir Kurulumu
    # Girdi: {"konu": "Python"}
    # Adım 1: RunnablePassthrough ile gelen girdiyi doğrudan okuruz.
    # Adım 2: Sözlük yapısı | (pipe) işlemine tabi tutulduğunda otomatik olarak
    #         RunnableParallel'e dönüştürülür. İki kol paralel threadlerde çalıştırılır.
    # Adım 3: İki koldan gelen sözlük bir birleştirme fonksiyonuna (RunnableLambda) aktarılır.
    zincir = (
        {
            "tanim": tanim_chain,
            "avantajlar": avantaj_chain
        }
        | RunnableLambda(combine_results)
    )
    
    # Çalıştır ve süreyi ölç
    import time
    start_time = time.time()
    
    rapor = zincir.invoke({"konu": "Python dili"})
    
    end_time = time.time()
    
    print(rapor)
    # İki model de 0.4 saniye gecikmeye sahip. 
    # Eğer ardışık çalışsalardı en az 0.8 saniye sürerdi.
    # Paralel çalıştıkları için toplam süre 0.4 saniyeye yakın olmalıdır.
    print(f"\nToplam işlem süresi: {end_time - start_time:.4f} saniye.")


if __name__ == "__main__":
    print("=== Lite-LCEL Paralel Zincir Uygulaması ===\n")
    run_parallel_chain()
