import pytest
import asyncio
from lite_lcel.base import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    coerce_to_runnable
)

def test_coerce_to_runnable():
    # Callable coerce test
    runnable_fn = coerce_to_runnable(lambda x: x + 1)
    assert isinstance(runnable_fn, RunnableLambda)
    assert runnable_fn.invoke(1) == 2

    # Dict coerce test
    runnable_dict = coerce_to_runnable({"val": lambda x: x + 1})
    assert isinstance(runnable_dict, RunnableParallel)
    assert runnable_dict.invoke(1) == {"val": 2}


def test_runnable_sequence():
    # Senkron zincir testi
    step1 = RunnableLambda(lambda x: x + 1)
    step2 = RunnableLambda(lambda x: x * 2)
    
    chain = step1 | step2
    assert chain.invoke(5) == 12


@pytest.mark.asyncio
async def test_runnable_sequence_async():
    # Asenkron zincir testi
    async def async_step1(x):
        await asyncio.sleep(0.01)
        return x + 1

    step1 = RunnableLambda(async_step1)
    step2 = RunnableLambda(lambda x: x * 2)
    
    chain = step1 | step2
    result = await chain.ainvoke(5)
    assert result == 12


def test_runnable_parallel():
    # Senkron paralel çalıştırma testi
    step1 = RunnableLambda(lambda x: x + 1)
    step2 = RunnableLambda(lambda x: x * 2)
    
    parallel = RunnableParallel({"add": step1, "multiply": step2})
    result = parallel.invoke(5)
    
    assert result == {"add": 6, "multiply": 10}


@pytest.mark.asyncio
async def test_runnable_parallel_async():
    # Asenkron paralel çalıştırma testi
    async def async_add(x):
        await asyncio.sleep(0.01)
        return x + 1

    step1 = RunnableLambda(async_add)
    step2 = RunnableLambda(lambda x: x * 2)
    
    parallel = RunnableParallel({"add": step1, "multiply": step2})
    result = await parallel.ainvoke(5)
    
    assert result == {"add": 6, "multiply": 10}


def test_runnable_passthrough():
    # Sadece passthrough
    passthrough = RunnablePassthrough()
    assert passthrough.invoke("test") == "test"

    # Passthrough.assign testi
    step = RunnablePassthrough.assign(y=lambda x: x["x"] * 2)
    result = step.invoke({"x": 5})
    assert result == {"x": 5, "y": 10}
