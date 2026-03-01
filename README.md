# # Breast Cancer Wisconsin - İkili Sınıflandırma

Bu proje, **Breast Cancer Wisconsin** veri seti üzerinde **Random Forest** algoritması kullanarak ikili sınıflandırma (binary classification) yapmaktadır.

## Amaç

Meme kanseri teşhisinde tümörün **iyi huylu (Benign)** mı yoksa **kötü huylu (Malignant)** mı olduğunu 30 farklı özniteliğe dayanarak tahmin etmek.

## Kullanılan Teknolojiler

- Python
- Pandas
- Scikit-Learn (Random Forest Classifier)

## Veri Seti

- **Kaynak:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Örnek sayısı:** 569
- **Öznitelik sayısı:** 30
- **Sınıflar:** B (Benign - İyi Huylu), M (Malignant - Kötü Huylu)

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Accuracy | %96.49 |
| Precision | %100.00 |
| Recall | %90.70 |
| F1-Score | %95.12 |

## Çalıştırma

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn
python3 Main.py
```
