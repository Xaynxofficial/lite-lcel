import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import PromptTemplate, MockLLM, StringOutputParser, JsonOutputParser

def run_basic_text_chain():
    print("--- 1. Metin Tabanlı Basit Zincir Örneği ---")
    
    # Adım 1: Prompt Şablonunu Tanımla
    prompt = PromptTemplate(template="Lütfen bana şu konu hakkında bilgi ver: {soru}")
    
    # Adım 2: Simüle LLM Tanımla
    model = MockLLM(model_name="Basic-GPT-3", delay=0.1)
    
    # Adım 3: Çıktı Ayrıştırıcıyı Tanımla
    parser = StringOutputParser()
    
    # Adım 4: LCEL Zincirini Oluştur
    chain = prompt | model | parser
    
    # Adım 5: Çalıştır (Invoke)
    sonuc = chain.invoke({"soru": "Gökyüzü neden mavidir?"})
    
    print(f"Çıktı:\n{sonuc}\n")


def run_basic_json_chain():
    print("--- 2. JSON Çıktı Ayrıştırma Örneği ---")
    
    # Şablon
    prompt = PromptTemplate(template="Bana {konu} konusu ile ilgili bilgileri JSON formatında sağla.")
    
    # Model (JSON formatı taleplerini yakalayıp JSON metni üretecek şekilde Mock'landı)
    model = MockLLM(model_name="JSON-GPT-4", delay=0.1)
    
    # JSON Çıktı Ayrıştırıcısı (Ham markdown kod bloklarını ayıklar ve dict'e çevirir)
    parser = JsonOutputParser()
    
    # Zincir
    chain = prompt | model | parser
    
    # Çalıştır
    sonuc = chain.invoke({"konu": "Python programlama dili"})
    
    print(f"Ayrıştırılmış Çıktı Tipi: {type(sonuc)}")
    print(f"Ayrıştırılmış Değerler:")
    for k, v in sonuc.items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    print("=== Lite-LCEL Basit Zincir Uygulamaları ===\n")
    run_basic_text_chain()
    run_basic_json_chain()
