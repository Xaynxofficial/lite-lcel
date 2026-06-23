import asyncio
import os
import sys
import time

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import PromptTemplate, MockLLM, StringOutputParser

async def run_async_invoke():
    print("--- 1. Asenkron Tekli Çağrı (ainvoke) ---")
    prompt = PromptTemplate(template="Sor: {soru}")
    # 0.5 saniye yapay gecikmesi olan mock model
    model = MockLLM(model_name="Async-GPT", delay=0.5)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    start = time.time()
    # ainvoke metodu event loop'u bloklamaz
    sonuc = await chain.ainvoke({"soru": "Asenkron Programlama"})
    end = time.time()
    
    print(f"Yanıt: {sonuc}")
    print(f"Süre: {end - start:.4f} saniye.\n")


async def run_async_batch():
    print("--- 2. Asenkron Toplu Çağrı (abatch) ---")
    prompt = PromptTemplate(template="Bilgi ver: {soru}")
    model = MockLLM(model_name="Async-GPT", delay=0.5)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    girdiler = [
        {"soru": "Gökyüzü"},
        {"soru": "Python"},
        {"soru": "LCEL Zincirleme"}
    ]
    
    print(f"{len(girdiler)} adet soru paralel ve asenkron olarak gönderiliyor...")
    
    start = time.time()
    # abatch tüm girdileri asyncio.gather ile eşzamanlı olarak arka arkaya tetikler
    sonuclar = await chain.abatch(girdiler)
    end = time.time()
    
    print("\nSonuçlar:")
    for girdi, sonuc in zip(girdiler, sonuclar):
        print(f"  Girdi: {girdi['soru']} -> Yanıt: {sonuc}")
        
    # Her sorgu 0.5 sn gecikmeli. Sıralı olsaydı 1.5 sn sürerdi. 
    # Asenkron paralel çalıştığı için ~0.5 sn civarında bitmelidir.
    print(f"\nToplam asenkron toplu işlem süresi: {end - start:.4f} saniye.")


async def main():
    print("=== Lite-LCEL Asenkron Zincir Uygulamaları ===\n")
    await run_async_invoke()
    await run_async_batch()


if __name__ == "__main__":
    asyncio.run(main())
