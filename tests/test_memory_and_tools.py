import pytest
from lite_lcel import (
    tool,
    StructuredTool,
    MockLLM,
    ChatMessageHistory,
    RunnableWithMessageHistory,
    PromptTemplate,
    StringOutputParser
)

def test_tool_decorator():
    @tool
    def topla(a: int, b: int) -> int:
        """İki sayıyı toplar."""
        return a + b
        
    assert isinstance(topla, StructuredTool)
    assert topla.name == "topla"
    assert topla.description == "İki sayıyı toplar."
    
    # Şema kontrolü
    schema = topla.to_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "topla"
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["function"]["parameters"]["properties"]["b"]["type"] == "integer"
    
    # Çalıştırma kontrolü
    assert topla.invoke({"a": 10, "b": 20}) == 30
    # Doğrudan fonksiyon çağrısı kontrolü
    assert topla(10, 20) == 30


def test_tool_binding():
    @tool
    def topla(a: int, b: int) -> int:
        """Topla"""
        return a + b
        
    model = MockLLM(delay=0.01)
    
    # Model kopyalanıp araçlar bağlanmalı
    bound_model = model.bind_tools([topla])
    assert bound_model != model
    assert len(bound_model.bound_tools) == 1
    assert bound_model.bound_tools[0].name == "topla"
    
    # Araç çağırma simülasyonu
    response = bound_model.invoke("Lütfen topla fonksiyonunu çalıştır.")
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "topla"
    assert "a" in response.tool_calls[0]["function"]["arguments"]


def test_message_history():
    history = ChatMessageHistory()
    assert len(history.messages) == 0
    
    history.add_user_message("Merhaba")
    history.add_ai_message("Selam!")
    
    assert len(history.messages) == 2
    assert history.messages[0].type == "human"
    assert history.messages[1].type == "ai"
    
    history.clear()
    assert len(history.messages) == 0


def test_runnable_with_message_history():
    # Bellek depolarını simüle edelim (in-memory)
    sessions = {}
    
    def get_session(session_id: str) -> ChatMessageHistory:
        if session_id not in sessions:
            sessions[session_id] = ChatMessageHistory()
        return sessions[session_id]
        
    prompt = PromptTemplate("Geçmiş:\n{gecmis}\nSoru: {soru}")
    model = MockLLM(delay=0.01)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    # Bellekli zincir kurulumu
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session,
        input_messages_key="soru",
        history_messages_key="gecmis"
    )
    
    config = {"configurable": {"session_id": "test-session-99"}}
    
    # 1. Soru
    res1 = chain_with_history.invoke({"soru": "Gökyüzü neden mavidir?"}, config=config)
    assert "Rayleigh" in res1
    
    # Tarihçede soru ve cevap birikmiş olmalı
    hist = get_session("test-session-99")
    assert len(hist.messages) == 2
    assert hist.messages[0].content == "Gökyüzü neden mavidir?"
    assert "Rayleigh" in hist.messages[1].content
    
    # 2. Soru (Geçmiş de prompta yansıtılmalı)
    res2 = chain_with_history.invoke({"soru": "Python nedir?"}, config=config)
    assert "okunabilirliği yüksek" in res2
    assert len(hist.messages) == 4
