import os
import sys
import time

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import RunnableLambda, ConsoleCallbackHandler

# Çağrı sayısını izlemek için sayaç
attempt_counter = 0

def unstable_api_endpoint(x: int) -> int:
    """
    İlk iki çağrıda hata fırlatan, 3. çağrıda başarıyla çalışan istikrarsız fonksiyon.
    """
    global attempt_counter
    attempt_counter += 1
    
    print(f"🔄 [Deneme {attempt_counter}] API çağrısı yapılıyor...")
    
    if attempt_counter < 3:
        print("   ❌ Bağlantı hatası oluştu! Sunucu yanıt vermiyor.")
        raise ConnectionError("Geçici API Bağlantı Hatası (HTTP 503)")
        
    print("   ✅ Başarılı! Sunucu yanıt verdi.")
    return x * 10


def run_retry_example():
    global attempt_counter
    attempt_counter = 0
    
    print("=== Üssel Bekleme (Exponential Backoff) Yeniden Deneme Örneği ===")
    
    # 1. İstikrarsız fonksiyondan bir Runnable oluşturalım
    unstable_runnable = RunnableLambda(unstable_api_endpoint)
    
    # 2. Yeniden deneme zinciri kurgulayalım
    # attempts=3: En fazla 3 kere dener.
    # backoff_factor=0.5: İlk bekleme 0.5sn, ikinci bekleme 0.5 * 2^1 = 1.0sn olacak şekilde artar.
    robust_chain = unstable_runnable.with_retry(
        attempts=3,
        backoff_factor=0.5,
        exceptions=(ConnectionError,)
    )
    
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    start_time = time.time()
    print("\nZincir başlatılıyor...")
    
    # Çalıştır (Hata vermemesi ve 3. denemede başarıyla 50 dönmesi gerekir)
    result = robust_chain.invoke(5, config=config)
    
    end_time = time.time()
    
    print(f"\n[Nihai Sonuç]: {result}")
    print(f"Toplam süre: {end_time - start_time:.4f} saniye.")
    # (İlk hata -> 0.5sn bekleme) + (İkinci hata -> 1.0sn bekleme) = ~1.5 saniye yapay bekleme süresi olmalıdır.


if __name__ == "__main__":
    run_retry_example()
