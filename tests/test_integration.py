import pytest
from lite_lcel.prompts import PromptTemplate
from lite_lcel.models import MockLLM
from lite_lcel.parsers import StringOutputParser, JsonOutputParser

def test_integration_chain_string():
    prompt = PromptTemplate(template="Sor: {soru}")
    model = MockLLM(delay=0.01)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    result = chain.invoke({"soru": "Gökyüzü neden mavi?"})
    
    assert "Rayleigh" in result
    # parser'ın baş/son boşlukları temizlediğini doğrula
    assert result == result.strip()


def test_integration_chain_json():
    prompt = PromptTemplate(template="Bana {soru} bilgisini JSON olarak ver.")
    model = MockLLM(delay=0.01)
    parser = JsonOutputParser()
    
    chain = prompt | model | parser
    result = chain.invoke({"soru": "python"})
    
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["data"]["dil"] == "Python"


def test_chain_batch():
    prompt = PromptTemplate(template="Sor: {soru}")
    model = MockLLM(delay=0.01)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    inputs = [{"soru": "Gökyüzü"}, {"soru": "Python"}]
    results = chain.batch(inputs)
    
    assert len(results) == 2
    assert "Rayleigh" in results[0]
    assert "okunabilirliği yüksek" in results[1]
