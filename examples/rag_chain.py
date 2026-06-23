import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    InMemoryVectorStore,
    RunnablePassthrough,
    RunnableLambda,
    ConsoleCallbackHandler
)

def format_docs(docs: list) -> str:
    """Retriever'dan dönen döküman nesnelerini metin haline getiren yardımcı fonksiyon."""
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)


def run_rag_chain_example():
    print("=== LCEL ile Uçtan Uca RAG (Retrieval-Augmented Generation) Zinciri ===")
    
    # 1. Vektör veritabanını oluştur ve bilgi tabanını yükle (Knowledge Base)
    vectorstore = InMemoryVectorStore()
    
    print("\nBilgi tabanına dökümanlar ekleniyor...")
    vectorstore.add_texts(
        texts=[
            "Python okunabilirliği yüksek, genel amaçlı, Guido van Rossum tarafından tasarlanan bir dildir.",
            "Gökyüzünün mavi olmasının sebebi Güneş ışınlarının atmosferdeki Rayleigh saçılmasıdır.",
            "LCEL (LangChain Expression Language), LangChain içindeki bileşenleri | operatörüyle bağlamayı sağlayan bir dildir."
        ],
        metadatas=[{"kaynak": "wiki"}, {"kaynak": "bilim_dergisi"}, {"kaynak": "lcel_rehberi"}]
    )
    
    # 2. Vektör veritabanını bir Retriever'a çevir (En benzer 1 dokümanı dönecek şekilde k=1 yapıyoruz)
    retriever = vectorstore.as_retriever(k=1)
    
    # 3. Prompt şablonunu oluştur (Bağlam ve soru yer tutucularıyla)
    prompt = PromptTemplate(
        template="Aşağıdaki döküman bağlamını temel alarak soruyu cevapla.\n"
                 "Bağlam:\n{context}\n"
                 "Soru: {soru}\n"
                 "Cevap:"
    )
    
    # Model ve parser
    model = MockLLM(model_name="RAG-GPT-4", delay=0.2)
    parser = StringOutputParser()
    
    # 4. RAG Zincirini LCEL ile kuralım
    # Girdi: "sorgu_metni" (örneğin invoke'a gönderilen dict veya doğrudan passthrough)
    # Adım 1: context alanını retriever'dan çekip format_docs ile formatlıyoruz.
    # Adım 2: soru alanını gelen girdiden aynen geçiriyoruz.
    # Adım 3: Prompt -> Model -> Parser zincirine bağlıyoruz.
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "soru": RunnablePassthrough()
        }
        | prompt
        | model
        | parser
    )
    
    # Callback loglarını ekleyerek detayları izleyelim
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    # RAG araması yapılması gereken soru
    soru = "Gökyüzünün mavi olmasının sebebi nedir?"
    print(f"\nKullanıcı: {soru}")
    print("="*60)
    
    # Zinciri tetikle (Sorguyu doğrudan gönderiyoruz)
    cevap = rag_chain.invoke(soru, config=config)
    
    print("="*60)
    print(f"\n[RAG Zinciri Yanıtı]:\n{cevap}")


if __name__ == "__main__":
    run_rag_chain_example()
