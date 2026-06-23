import math
from collections import Counter
from typing import Any, Dict, List, Tuple
from lite_lcel.base import Runnable

class Document:
    """
    RAG sistemi tarafından kullanılan döküman veri modeli.
    """
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(page_content={repr(self.page_content[:50])}..., metadata={self.metadata})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Document):
            return False
        return self.page_content == other.page_content and self.metadata == other.metadata


def _cosine_similarity(text1: str, text2: str) -> float:
    """
    Dış bağımlılık (numpy/scipy) olmadan, saf Python ile kelime frekansı
    tabanlı kosinüs benzerliği hesaplayan yardımcı fonksiyon.
    """
    # Basit bir kelime tokenizasyonu (küçük harfe çevirip kelimeleri ayırıyoruz)
    words1 = [w.strip(".,!?\"'()[]{}") for w in text1.lower().split() if w.strip()]
    words2 = [w.strip(".,!?\"'()[]{}") for w in text2.lower().split() if w.strip()]
    
    vec1 = Counter(words1)
    # Eğer boş metinler geldiyse sıfır dön
    if not vec1:
        return 0.0
        
    vec2 = Counter(words2)
    if not vec2:
        return 0.0
        
    # Kesişim kümesi
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    # Skaler Çarpım (Dot Product)
    dot_product = sum(vec1[x] * vec2[x] for x in intersection)
    
    # Vektör Boyutları (Magnitudes)
    sum1 = sum(vec1[x] ** 2 for x in vec1.keys())
    sum2 = sum(vec2[x] ** 2 for x in vec2.keys())
    
    magnitude = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not magnitude:
        return 0.0
    return float(dot_product) / magnitude


class InMemoryVectorStore:
    """
    Dokümanları hafızada saklayan ve kelime frekans kosinüs benzerliği
    kullanarak arama gerçekleştiren hafif vektör veritabanı.
    """
    def __init__(self):
        self.documents: List[Document] = []

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Veritabanına yeni metin dökümanları ekler."""
        metadatas = metadatas or [None] * len(texts)
        for text, meta in zip(texts, metadatas):
            self.documents.append(Document(page_content=text, metadata=meta))

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Sorguya en benzer k adet dökümanı kosinüs benzerliği ile bulur."""
        scored_docs: List[Tuple[Document, float]] = []
        
        for doc in self.documents:
            score = _cosine_similarity(query, doc.page_content)
            scored_docs.append((doc, score))
            
        # Skorlara göre büyükten küçüğe sırala
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # En iyi k sonucu dön
        return [doc for doc, score in scored_docs[:k]]

    def as_retriever(self, k: int = 2) -> "VectorStoreRetriever":
        """Vektör veritabanını zincirde (LCEL) kullanılabilir bir Retriever'a çevirir."""
        return VectorStoreRetriever(vectorstore=self, k=k)


class VectorStoreRetriever(Runnable):
    """
    LCEL zincirlerine döküman getirme adımı (Retriever) olarak entegre edilebilen Runnable sınıfı.
    """
    def __init__(self, vectorstore: InMemoryVectorStore, k: int = 2):
        self.vectorstore = vectorstore
        self.k = k

    def _invoke(self, query: str, config: Dict[str, Any] = None) -> List[Document]:
        if not isinstance(query, str):
            raise TypeError("Retriever girdisi sadece düz metin (str) olabilir.")
        return self.vectorstore.similarity_search(query, k=self.k)

    async def _ainvoke(self, query: str, config: Dict[str, Any] = None) -> List[Document]:
        # Arama işlemi CPU-bound olduğu için asenkron yürütmeyi thread'e delege ederiz
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._invoke, query, config)
