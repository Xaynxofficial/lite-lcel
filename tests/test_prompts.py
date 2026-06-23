import pytest
from lite_lcel.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessage,
    HumanMessage,
    AIMessage
)

def test_prompt_template():
    prompt = PromptTemplate(template="Merhaba {isim}, nasılsın?")
    result = prompt.invoke({"isim": "Ahmet"})
    assert result == "Merhaba Ahmet, nasılsın?"


def test_chat_prompt_template():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen bir {rol} yardımcısısın."),
        ("human", "Soru: {soru}")
    ])
    
    messages = chat_prompt.invoke({"rol": "kodlama", "soru": "Python nedir?"})
    
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Sen bir kodlama yardımcısısın."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Soru: Python nedir?"


def test_message_equality():
    msg1 = HumanMessage(content="Test")
    msg2 = HumanMessage(content="Test")
    msg3 = AIMessage(content="Test")
    
    assert msg1 == msg2
    assert msg1 != msg3
