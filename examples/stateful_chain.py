import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    ChatMessageHistory,
    RunnableWithMessageHistory
)

# Bellek depomuz (Bellekler in-memory saklanıyor)
session_db = {}

def get_chat_history(session_id: str) -> ChatMessageHistory:
    """Verilen session_id'ye ait sohbet geçmişini getirir/yaratır."""
    if session_id not in session_db:
        session_db[session_id] = ChatMessageHistory()
    return session_db[session_id]


def run_stateful_chat_example():
    print("=== Bellek Destekli Durumsal Sohbet Robotu (Chatbot) ===")
    
    # Adım 1: Prompt Şablonunu Oluştur.
    # Şablon, sohbet geçmişini {gecmis} değişkeninden, güncel soruyu ise {soru} değişkeninden alır.
    prompt = PromptTemplate(
        template="Aşağıdaki konuşma geçmişine göre soruyu yanıtla.\n"
                 "Konuşma Geçmişi:\n{gecmis}\n"
                 "Soru: {soru}\n"
                 "Yanıt:"
    )
    
    # Adım 2: Model ve parser tanımla
    model = MockLLM(model_name="Chat-GPT", delay=0.2)
    parser = StringOutputParser()
    
    # Temel zincirimiz
    base_chain = prompt | model | parser
    
    # Adım 3: Zinciri Bellek Çevreleyicisi ile sarmala
    # Bu sayede geçmiş otomatik olarak yüklenip kaydedilecektir.
    chat_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history=get_chat_history,
        input_messages_key="soru",
        history_messages_key="gecmis"
    )
    
    # Session ID yapılandırması
    config_user1 = {"configurable": {"session_id": "kullanici_ahmet"}}
    config_user2 = {"configurable": {"session_id": "kullanici_ayse"}}
    
    print("\n--- AHMET İÇİN 1. MESAJ ---")
    res_ahmet_1 = chat_chain.invoke({"soru": "Gökyüzü neden mavi?"}, config=config_user1)
    print(f"Ahmet: Gökyüzü neden mavi?\nBot: {res_ahmet_1}")
    
    print("\n--- AHMET İÇİN 2. MESAJ (GEÇMİŞ KONTROLÜ) ---")
    # Ahmet devam sorusu soruyor, bot geçmişe hakim olmalı
    res_ahmet_2 = chat_chain.invoke({"soru": "Peki Python programlama dili nedir?"}, config=config_user1)
    print(f"Ahmet: Peki Python programlama dili nedir?\nBot: {res_ahmet_2}")
    
    print("\n--- AYŞE İÇİN 1. MESAJ (AHMET'TEN BAĞIMSIZ YENİ SEANS) ---")
    # Ayşe tamamen farklı bir session_id kullanıyor, Ahmet'in geçmişini görmemeli
    res_ayse_1 = chat_chain.invoke({"soru": "Gökyüzü neden mavi?"}, config=config_user2)
    print(f"Ayşe: Gökyüzü neden mavi?\nBot: {res_ayse_1}")
    
    # Sistemdeki kayıtlı geçmişleri yazdıralım
    print("\n--- VERİTABANI HAFIZA DETAYLARI ---")
    for session_id, history in session_db.items():
        print(f"Session: {session_id}")
        for i, msg in enumerate(history.messages):
            print(f"  [{i}] {msg.type.upper()}: {msg.content}")


if __name__ == "__main__":
    run_stateful_chat_example()
