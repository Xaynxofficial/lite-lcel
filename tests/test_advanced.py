import pytest
import asyncio
from lite_lcel import (
    PromptTemplate,
    MockLLM,
    StringOutputParser,
    RunnableBranch,
    RunnableLambda,
    BaseCallbackHandler
)

# Custom callback to inspect execution
class MockCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.starts = []
        self.ends = []
        self.errors = []
        self.llm_starts = []
        self.llm_ends = []

    def on_chain_start(self, name, input_data, config=None):
        self.starts.append((name, input_data))

    def on_chain_end(self, name, output, config=None):
        self.ends.append((name, output))

    def on_chain_error(self, name, error, config=None):
        self.errors.append((name, error))

    def on_llm_start(self, name, prompt, config=None):
        self.llm_starts.append((name, prompt))

    def on_llm_end(self, name, response, config=None):
        self.llm_ends.append((name, response))


def test_streaming():
    prompt = PromptTemplate("Soru: {soru}")
    # 0.1 sn gecikmeli model
    model = MockLLM(delay=0.1)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    # Kelime kelime akış olmalı
    chunks = list(chain.stream({"soru": "Gökyüzü"}))
    
    assert len(chunks) > 1
    full_text = "".join(chunks)
    assert "Rayleigh" in full_text
    # StringOutputParser whitespace temizlemeli
    assert full_text == full_text.strip()


@pytest.mark.asyncio
async def test_async_streaming():
    prompt = PromptTemplate("Soru: {soru}")
    model = MockLLM(delay=0.1)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    chunks = []
    async for chunk in chain.astream({"soru": "Python"}):
        chunks.append(chunk)
        
    assert len(chunks) > 1
    full_text = "".join(chunks)
    assert "okunabilirliği" in full_text


def test_runnable_branch():
    # Koşul 1: Sayı > 10 ise x 2 yap
    # Koşul 2: Sayı <= 10 ise x 10 yap
    branch = RunnableBranch(
        (lambda x: x > 10, RunnableLambda(lambda x: x * 2)),
        RunnableLambda(lambda x: x * 10)
    )
    
    assert branch.invoke(15) == 30
    assert branch.invoke(5) == 50


def test_runnable_with_fallbacks():
    # Hata veren bir lambda
    def failing_fn(x):
        raise ValueError("Simüle edilmiş hata")
        
    failing_runnable = RunnableLambda(failing_fn)
    backup_runnable = RunnableLambda(lambda x: x + 10)
    
    # fallback tanımla
    chain = failing_runnable.with_fallbacks([backup_runnable])
    
    # Hata durumunda backup çalışmalı
    result = chain.invoke(5)
    assert result == 15


def test_callbacks():
    handler = MockCallbackHandler()
    config = {"callbacks": [handler]}
    
    prompt = PromptTemplate("Soru: {soru}")
    model = MockLLM(delay=0.01)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    result = chain.invoke({"soru": "Python"}, config=config)
    
    # Callback'lerin çağrıldığını doğrula
    assert len(handler.starts) > 0
    assert len(handler.ends) > 0
    assert len(handler.llm_starts) == 1
    assert len(handler.llm_ends) == 1
    
    # Başlangıç girdi tip ve isimlerini kontrol et
    chain_names = [name for name, _ in handler.starts]
    assert "RunnableSequence" in chain_names
    assert "PromptTemplate" in chain_names
    assert "MockLLM" in chain_names
    assert "StringOutputParser" in chain_names
