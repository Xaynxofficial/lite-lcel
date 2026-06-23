import asyncio
import os
import sys

# Proje dizinini Python yoluna ekleyelim (local import için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    ConsoleCallbackHandler
)

def run_sync_streaming_with_callbacks():
    print("=== 1. Senkron Akış ve Callbacks Yöneticisi ===")
    
    prompt = PromptTemplate(template="Lütfen bana şu konu hakkında bilgi ver: {soru}")
    model = MockLLM(model_name="Stream-GPT", delay=0.8)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    # ConsoleCallbackHandler ekleyerek zincir içindeki tüm olayları detaylıca izleyeceğiz
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    print("\nZincir tetikleniyor. Kelime kelime akış başlayacak...\n")
    
    # stream metodu çağrılıyor
    for chunk in chain.stream({"soru": "LCEL"}, config=config):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        
    print("\n\n--- Senkron Akış Tamamlandı ---\n")


async def run_async_streaming():
    print("=== 2. Asenkron Akış (astream) ===")
    
    prompt = PromptTemplate(template="Sor: {soru}")
    model = MockLLM(model_name="Async-Stream-GPT", delay=0.6)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    print("\nAsenkron akış başlıyor...\n")
    
    async for chunk in chain.astream({"soru": "Python"}):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        
    print("\n\n--- Asenkron Akış Tamamlandı ---\n")


if __name__ == "__main__":
    run_sync_streaming_with_callbacks()
    asyncio.run(run_async_streaming())
