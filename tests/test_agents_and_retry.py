import pytest
import asyncio
from lite_lcel import (
    RunnableLambda,
    RunnableRetry,
    AgentExecutor,
    MockLLM,
    tool
)

def test_runnable_retry_success():
    call_count = 0
    
    def failing_then_success(x):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Geçici sunucu hatası")
        return x * 2

    runnable = RunnableLambda(failing_then_success)
    # Üssel beklemeyi testlerde hızlandırmak için backoff_factor'ü küçük tutalım
    retry_chain = runnable.with_retry(attempts=3, backoff_factor=0.01)
    
    result = retry_chain.invoke(5)
    assert result == 10
    assert call_count == 3  # 2 hata, 1 başarı


def test_runnable_retry_failure():
    call_count = 0
    
    def always_failing(x):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Bağlantı koptu")
        
    runnable = RunnableLambda(always_failing)
    retry_chain = runnable.with_retry(attempts=3, backoff_factor=0.01)
    
    with pytest.raises(ConnectionError):
        retry_chain.invoke(5)
        
    assert call_count == 3


@pytest.mark.asyncio
async def test_runnable_retry_async():
    call_count = 0
    
    async def async_failing(x):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Async hata")
        return x + 10
        
    runnable = RunnableLambda(async_failing)
    retry_chain = runnable.with_retry(attempts=2, backoff_factor=0.01)
    
    result = await retry_chain.ainvoke(5)
    assert result == 15
    assert call_count == 2


def test_agent_executor_loop():
    @tool
    def topla(a: int, b: int) -> int:
        """İki sayıyı toplar."""
        return a + b
        
    model = MockLLM(delay=0.01)
    executor = AgentExecutor(model=model, tools=[topla], max_iterations=3)
    
    # MockLLM 'topla' fonksiyonunun ismini görünce tool_call isteği dönecek,
    # System: Tool returned 15 alınca da toplama sonucunu döndürecektir.
    sonuc = executor.invoke({"soru": "Lütfen topla fonksiyonunu çalıştır."})
    
    assert "15" in sonuc


@pytest.mark.asyncio
async def test_agent_executor_loop_async():
    @tool
    def topla(a: int, b: int) -> int:
        """İki sayıyı toplar."""
        return a + b
        
    model = MockLLM(delay=0.01)
    executor = AgentExecutor(model=model, tools=[topla], max_iterations=3)
    
    sonuc = await executor.ainvoke({"soru": "Lütfen topla fonksiyonunu çalıştır."})
    assert "15" in sonuc
