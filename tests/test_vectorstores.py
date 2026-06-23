import pytest
from lite_lcel.vectorstores import (
    Document,
    InMemoryVectorStore,
    VectorStoreRetriever,
    _cosine_similarity
)

def test_cosine_similarity():
    # Aynı metinler arası benzerlik 1.0 olmalıdır (hassasiyet toleransı ile)
    assert _cosine_similarity("elma armut", "elma armut") == pytest.approx(1.0)
    
    # Tamamen farklı metinler arası benzerlik 0.0 olmalıdır
    assert _cosine_similarity("kedi köpek", "elma armut") == 0.0
    
    # Kısmi benzerlik 0 ile 1 arasında olmalıdır
    score = _cosine_similarity("python programlama dili", "python yapay zeka")
    assert 0.0 < score < 1.0
    
    # Küçük/büyük harf ve noktalama toleransı
    assert _cosine_similarity("Python!", "python") == pytest.approx(1.0)


def test_vectorstore_search():
    vs = InMemoryVectorStore()
    vs.add_texts(
        texts=[
            "Gökyüzü neden mavi görünür? Rayleigh saçılması yüzünden.",
            "Python harika bir programlama dilidir.",
            "Bugün hava oldukça güneşli ve sıcak."
        ],
        metadatas=[
            {"konu": "fizik"},
            {"konu": "yazilim"},
            {"konu": "hava"}
        ]
    )
    
    # Yazılım konulu arama
    results = vs.similarity_search("python dili nedir", k=1)
    
    assert len(results) == 1
    assert "harika bir programlama" in results[0].page_content
    assert results[0].metadata["konu"] == "yazilim"


def test_retriever_runnable():
    vs = InMemoryVectorStore()
    vs.add_texts(["Gökyüzü mavidir.", "Yapay zeka gelecektir."])
    
    retriever = vs.as_retriever(k=1)
    assert isinstance(retriever, VectorStoreRetriever)
    
    # Invoke araması
    results = retriever.invoke("yapay zeka")
    assert len(results) == 1
    assert "gelecektir" in results[0].page_content


@pytest.mark.asyncio
async def test_retriever_runnable_async():
    vs = InMemoryVectorStore()
    vs.add_texts(["Gökyüzü mavidir.", "Yapay zeka gelecektir."])
    
    retriever = vs.as_retriever(k=1)
    
    # Asenkron arama
    results = await retriever.ainvoke("gökyüzü")
    assert len(results) == 1
    assert "mavidir" in results[0].page_content
