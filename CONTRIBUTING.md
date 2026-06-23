# Lite-LCEL Katkı Sağlama Kılavuzu

Lite-LCEL projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Bu kılavuz, projeyi yerel bilgisayarınızda nasıl kuracağınızı, testleri nasıl çalıştıracağınızı ve kod standartlarımızı nasıl takip edeceğinizi açıklar.

---

## 🛠️ Geliştirme Ortamı Kurulumu

Proje, Python paket yönetimi ve geliştirme araçlarını (`pytest`, `ruff`, `black`) içeren opsiyonel geliştirme bağımlılıklarını destekler.

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/kullanici_adi/lite-lcel.git
   cd lite-lcel
   ```

2. **Sanal ortam oluşturun ve aktif edin:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows için: .venv\Scripts\activate
   ```

3. **Bağımlılıkları geliştirici modunda kurun:**
   ```bash
   pip install -e .[dev,openai]
   ```

---

## 🧪 Testlerin Çalıştırılması

Tüm çekirdek mantık, RAG veritabanı, ajan döngüleri ve asenkron akışlar unit testlerle doğrulanmaktadır.

Testleri çalıştırmak için terminalden şu komutu verin:
```bash
pytest
```

Alternatif olarak, Makefile kullanıyorsanız:
```bash
make test
```

---

## 🎨 Kod Düzeni ve Standartlar

Kod kalitesini ve düzenini korumak için `ruff` aracı kullanılmaktadır. Lütfen kodunuzu göndermeden önce:

1. **Linter taraması yapın ve düzeltin:**
   ```bash
   ruff check . --fix
   ```

2. **Kod biçimlendirmesini kontrol edin:**
   ```bash
   ruff format .
   ```

---

## 📥 Çekme Talebi (Pull Request) Süreci

1. Yeni bir özellik veya hata düzeltmesi için ana daldan (`main`) yeni bir dal (branch) oluşturun: `git checkout -b ozellik/yeni-adim`
2. Değişikliklerinizi yapın ve ilgili unit testleri ekleyin/çalıştırın.
3. Değişikliklerinizi commitleyin: `git commit -m "Özellik: X adımı eklendi"`
4. Dalınızı GitHub'a gönderin ve `main` dalına yönelik bir Pull Request açın.
