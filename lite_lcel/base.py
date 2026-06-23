import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Union, Generator, AsyncGenerator

def coerce_to_runnable(thing: Any) -> "Runnable":
    """
    Verilen nesneyi bir Runnable nesnesine dönüştürür.
    """
    if isinstance(thing, Runnable):
        return thing
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    elif callable(thing):
        return RunnableLambda(thing)
    else:
        raise TypeError(
            f"Nesne LCEL zincirine dönüştürülemedi. Runnable, callable veya dict olmalıdır. "
            f"Alınan tip: {type(thing)}"
        )


def _accumulate_generator(gen) -> Any:
    accumulated = None
    for chunk in gen:
        if accumulated is None:
            accumulated = chunk
        else:
            if isinstance(accumulated, dict) and isinstance(chunk, dict):
                accumulated.update(chunk)
            elif hasattr(accumulated, "content") and not isinstance(accumulated, str) and hasattr(chunk, "content") and not isinstance(chunk, str):
                # Mesaj içeriklerini birleştir (duck-typing)
                accumulated.content += chunk.content
            else:
                accumulated += chunk
    return accumulated


async def _accumulate_async_generator(agen) -> Any:
    accumulated = None
    async for chunk in agen:
        if accumulated is None:
            accumulated = chunk
        else:
            if isinstance(accumulated, dict) and isinstance(chunk, dict):
                accumulated.update(chunk)
            elif hasattr(accumulated, "content") and not isinstance(accumulated, str) and hasattr(chunk, "content") and not isinstance(chunk, str):
                accumulated.content += chunk.content
            else:
                accumulated += chunk
    return accumulated


class Runnable:
    """
    LCEL zincirleme mantığının temel sınıfı.
    Gelişmiş Callbacks (Olay İzleme), Streaming (Akış), Fallback ve Branch desteği içerir.
    """
    accepts_streaming_input = False  # Alt sınıflar girdi olarak jeneratör alıp alamayacağını belirtir.

    def _resolve_input(self, input_data: Any) -> Any:
        # Eğer girdi bir jeneratör ise ve bu Runnable akış girdisi kabul etmiyorsa, önce biriktirir
        if inspect.isgenerator(input_data):
            if not self.accepts_streaming_input:
                return _accumulate_generator(input_data)
        return input_data

    async def _resolve_async_input(self, input_data: Any) -> Any:
        if inspect.isasyncgen(input_data):
            if not self.accepts_streaming_input:
                return await _accumulate_async_generator(input_data)
        return input_data

    def invoke(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
        config = config or {}
        # Eğer girdi jeneratör ise invoke için önce biriktirilir
        input_data = self._resolve_input(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_chain_start(self.__class__.__name__, input_data, config)
        try:
            output = self._invoke(input_data, config)
        except Exception as e:
            for cb in callbacks:
                cb.on_chain_error(self.__class__.__name__, e, config)
            raise e
        for cb in callbacks:
            cb.on_chain_end(self.__class__.__name__, output, config)
        return output

    async def ainvoke(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
        config = config or {}
        # Async jeneratör veya normal jeneratör çözümlenir
        if inspect.isasyncgen(input_data):
            input_data = await self._resolve_async_input(input_data)
        else:
            input_data = self._resolve_input(input_data)
            
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_chain_start(self.__class__.__name__, input_data, config)
        try:
            output = await self._ainvoke(input_data, config)
        except Exception as e:
            for cb in callbacks:
                cb.on_chain_error(self.__class__.__name__, e, config)
            raise e
        for cb in callbacks:
            cb.on_chain_end(self.__class__.__name__, output, config)
        return output

    def stream(self, input_data: Any, config: Dict[str, Any] = None) -> Generator[Any, None, None]:
        config = config or {}
        # Eğer girdi jeneratör ise çözümlenir (sadece accepts_streaming_input False ise)
        input_data = self._resolve_input(input_data)
        
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_chain_start(self.__class__.__name__, input_data, config)
        try:
            for chunk in self._stream(input_data, config):
                yield chunk
        except Exception as e:
            for cb in callbacks:
                cb.on_chain_error(self.__class__.__name__, e, config)
            raise e
        for cb in callbacks:
            cb.on_chain_end(self.__class__.__name__, "[Stream Finished]", config)

    async def astream(self, input_data: Any, config: Dict[str, Any] = None) -> AsyncGenerator[Any, None]:
        config = config or {}
        if inspect.isasyncgen(input_data):
            input_data = await self._resolve_async_input(input_data)
        else:
            input_data = self._resolve_input(input_data)
            
        callbacks = config.get("callbacks") or []
        for cb in callbacks:
            cb.on_chain_start(self.__class__.__name__, input_data, config)
        try:
            async for chunk in self._astream(input_data, config):
                yield chunk
        except Exception as e:
            for cb in callbacks:
                cb.on_chain_error(self.__class__.__name__, e, config)
            raise e
        for cb in callbacks:
            cb.on_chain_end(self.__class__.__name__, "[Async Stream Finished]", config)

    # Alt sınıfların asıl iş mantıklarını implement edeceği korumalı metotlar
    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.invoke, input_data, config)

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[Any, None, None]:
        yield self.invoke(input_data, config)

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        yield await self.ainvoke(input_data, config)

    def batch(self, inputs: List[Any], config: Dict[str, Any] = None) -> List[Any]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda x: self.invoke(x, config), inputs))

    async def abatch(self, inputs: List[Any], config: Dict[str, Any] = None) -> List[Any]:
        tasks = [self.ainvoke(x, config) for x in inputs]
        return await asyncio.gather(*tasks)

    def __or__(self, other: Any) -> "RunnableSequence":
        return RunnableSequence(self, coerce_to_runnable(other))

    def __ror__(self, other: Any) -> "RunnableSequence":
        return RunnableSequence(coerce_to_runnable(other), self)

    def with_fallbacks(self, fallbacks: List["Runnable"]) -> "RunnableWithFallbacks":
        """Zincire yedek kurtarma yolları ekler."""
        return RunnableWithFallbacks(self, fallbacks)

    def with_retry(self, attempts: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)) -> "RunnableRetry":
        """Hata durumunda zinciri üssel bekleme ile otomatik yeniden dener."""
        return RunnableRetry(self, attempts, backoff_factor, exceptions)


class RunnableSequence(Runnable):
    """
    Sıralı veri boru hattını (pipeline) çalıştıran sınıf.
    """
    def __init__(self, first: Runnable, second: Runnable):
        self.first = first
        self.second = second

    @property
    def accepts_streaming_input(self) -> bool:
        # Sequence'ın kendisi ilk elemanına göre streaming input alabilir
        return self.first.accepts_streaming_input

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        first_output = self.first.invoke(input_data, config)
        return self.second.invoke(first_output, config)

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        first_output = await self.first.ainvoke(input_data, config)
        return await self.second.ainvoke(first_output, config)

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[Any, None, None]:
        # Akış zincirlenmesinde: ilk elemanın jeneratörünü doğrudan ikinci elemana paslarız!
        first_stream = self.first.stream(input_data, config)
        yield from self.second.stream(first_stream, config)

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        first_stream = self.first.astream(input_data, config)
        async for chunk in self.second.astream(first_stream, config):
            yield chunk


class RunnableParallel(Runnable):
    """
    İşlemleri paralel olarak yürüten sınıf.
    """
    def __init__(self, steps: Dict[str, Any]):
        self.steps = {k: coerce_to_runnable(v) for k, v in steps.items()}

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                k: executor.submit(step.invoke, input_data, config)
                for k, step in self.steps.items()
            }
            for k, future in futures.items():
                results[k] = future.result()
        return results

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        tasks = {
            k: asyncio.create_task(step.ainvoke(input_data, config))
            for k, step in self.steps.items()
        }
        for k, task in tasks.items():
            results[k] = await task
        return results


class RunnableLambda(Runnable):
    """
    Fonksiyonları zincire uyarlayan sınıf.
    """
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        sig = inspect.signature(self.func)
        if "config" in sig.parameters:
            return self.func(input_data, config)
        return self.func(input_data)

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        sig = inspect.signature(self.func)
        has_config = "config" in sig.parameters
        
        if inspect.iscoroutinefunction(self.func):
            if has_config:
                return await self.func(input_data, config)
            return await self.func(input_data)
        
        return self._invoke(input_data, config)


class RunnablePassthrough(Runnable):
    """
    Girdiyi aynen aktaran veya ek değişkenler bağlayan yardımcı sınıf.
    """
    def __init__(self, assign_func: Union[Callable[[Any], Any], None] = None, **kwargs: Any):
        self.assign_func = assign_func
        self.extra_steps = {k: coerce_to_runnable(v) for k, v in kwargs.items()}

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        if not self.extra_steps:
            return input_data
            
        if not isinstance(input_data, dict):
            raise ValueError("RunnablePassthrough.assign yalnızca girdi sözlük (dict) olduğunda çalışır.")
            
        output = input_data.copy()
        for k, step in self.extra_steps.items():
            output[k] = step.invoke(input_data, config)
        return output

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        if not self.extra_steps:
            return input_data
            
        if not isinstance(input_data, dict):
            raise ValueError("RunnablePassthrough.assign yalnızca girdi sözlük (dict) olduğunda çalışır.")
            
        output = input_data.copy()
        for k, step in self.extra_steps.items():
            output[k] = await step.ainvoke(input_data, config)
        return output

    @classmethod
    def assign(cls, **kwargs: Any) -> "RunnablePassthrough":
        return cls(**kwargs)


class RunnableBranch(Runnable):
    """
    Girdiye göre koşullu yönlendirme yapan sınıf (Routing).
    """
    def __init__(self, *branches: Union[tuple, Runnable]):
        self.branches = []
        self.default = None
        
        for branch in branches:
            if isinstance(branch, tuple) and len(branch) == 2:
                cond, runnable = branch
                self.branches.append((cond, coerce_to_runnable(runnable)))
            else:
                self.default = coerce_to_runnable(branch)
                
        if self.default is None:
            raise ValueError("RunnableBranch en az bir varsayılan (default) Runnable içermelidir.")

    def _select_runnable(self, input_data: Any, config: Dict[str, Any]) -> Runnable:
        for condition, runnable in self.branches:
            sig = inspect.signature(condition)
            if "config" in sig.parameters:
                is_met = condition(input_data, config)
            else:
                is_met = condition(input_data)
                
            if is_met:
                return runnable
        return self.default

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        runnable = self._select_runnable(input_data, config)
        return runnable.invoke(input_data, config)

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        runnable = self._select_runnable(input_data, config)
        return await runnable.ainvoke(input_data, config)

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[Any, None, None]:
        runnable = self._select_runnable(input_data, config)
        yield from runnable.stream(input_data, config)

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        runnable = self._select_runnable(input_data, config)
        async for chunk in runnable.astream(input_data, config):
            yield chunk


class RunnableWithFallbacks(Runnable):
    """
    Zincir hata verdiğinde alternatif zincirleri devreye sokan sınıf.
    """
    def __init__(self, runnable: Runnable, fallbacks: List[Runnable], exceptions: tuple = (Exception,)):
        self.runnable = runnable
        self.fallbacks = [coerce_to_runnable(fb) for fb in fallbacks]
        self.exceptions = exceptions

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        try:
            return self.runnable.invoke(input_data, config)
        except self.exceptions as e:
            last_error = e
            for fallback in self.fallbacks:
                try:
                    return fallback.invoke(input_data, config)
                except self.exceptions as fe:
                    last_error = fe
            raise last_error

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        try:
            return await self.runnable.ainvoke(input_data, config)
        except self.exceptions as e:
            last_error = e
            for fallback in self.fallbacks:
                try:
                    return await fallback.ainvoke(input_data, config)
                except self.exceptions as fe:
                    last_error = fe
            raise last_error

    def _stream(self, input_data: Any, config: Dict[str, Any]) -> Generator[Any, None, None]:
        try:
            yield from self.runnable.stream(input_data, config)
        except self.exceptions as e:
            last_error = e
            for fallback in self.fallbacks:
                try:
                    yield from fallback.stream(input_data, config)
                    return
                except self.exceptions as fe:
                    last_error = fe
            raise last_error

    async def _astream(self, input_data: Any, config: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        try:
            async for chunk in self.runnable.astream(input_data, config):
                yield chunk
        except self.exceptions as e:
            last_error = e
            for fallback in self.fallbacks:
                try:
                    async for chunk in fallback.astream(input_data, config):
                        yield chunk
                    return
                except self.exceptions as fe:
                    last_error = fe
            raise last_error


class RunnableRetry(Runnable):
    """
    Hata fırlatıldığında zinciri üssel bekleme (exponential backoff) ile yeniden deneyen sınıf.
    """
    def __init__(self, runnable: Runnable, attempts: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
        self.runnable = runnable
        self.attempts = attempts
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def _invoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        import time
        last_error = None
        for i in range(self.attempts):
            try:
                return self.runnable.invoke(input_data, config)
            except self.exceptions as e:
                last_error = e
                if i < self.attempts - 1:
                    sleep_time = self.backoff_factor * (2 ** i)
                    time.sleep(sleep_time)
        raise last_error

    async def _ainvoke(self, input_data: Any, config: Dict[str, Any]) -> Any:
        last_error = None
        for i in range(self.attempts):
            try:
                return await self.runnable.ainvoke(input_data, config)
            except self.exceptions as e:
                last_error = e
                if i < self.attempts - 1:
                    sleep_time = self.backoff_factor * (2 ** i)
                    await asyncio.sleep(sleep_time)
        raise last_error

