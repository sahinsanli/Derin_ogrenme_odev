# =============================================================
# Breast Cancer Wisconsin - İkili Sınıflandırma (Binary Classification)
# Kullanılan Kütüphaneler: Python, Pandas, Scikit-Learn
# =============================================================

# --- Adım 1: Gerekli kütüphanelerin içe aktarılması ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Adım 2: Veri setinin okunması ---
# 'data - data.csv' dosyasını Pandas ile okuyoruz
df = pd.read_csv("data - data.csv")

# Veri hakkinda bilgi yazdiriyoruz
print("Veri seti okundu.")
print(f"Toplam {df.shape[0]} satir ve {df.shape[1]} sutun var.")
print()

# --- Adım 3: Hedef değişken (y) ve özelliklerin (X) ayrılması ---
# Hedef değişkeni otomatik olarak buluyoruz.
# Önce bilinen hedef sütun isimlerini kontrol ediyoruz, bulunamazsa son sütunu kullanıyoruz.
hedef_adaylari = ["diagnosis", "target", "class", "label"]
hedef_sutun = None

for aday in hedef_adaylari:
    if aday in df.columns:
        hedef_sutun = aday
        break

# Eğer bilinen isimlerden hiçbiri yoksa, son sütunu hedef olarak kabul ediyoruz
if hedef_sutun is None:
    hedef_sutun = df.columns[-1]
    print(f"Hedef sutun bulunamadi, son sutun kullaniliyor: '{hedef_sutun}'")

print(f"Hedef degisken: '{hedef_sutun}'")
print(f"Sinif dagilimi:\n{df[hedef_sutun].value_counts()}")
print()

# 'id' sütunu varsa özelliklerden çıkarıyoruz (sınıflandırmaya katkısı yoktur)
ozellik_sutunlari = [col for col in df.columns if col != hedef_sutun and col.lower() != "id"]

X = df[ozellik_sutunlari]  # Özellik matrisi (bağımsız değişkenler)
y = df[hedef_sutun]         # Hedef değişken (bağımlı değişken)

# Eğer hedef değişken kategorikse (örn: 'M', 'B'), sayısal değerlere dönüştürüyoruz
# not: pandas bazen 'object' yerine 'category' veya başka tipler atayabilir
if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    print(f"Etiketler sayisala donusturuldu: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print()

print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print()

# --- Adım 4: Veri setini eğitim ve test olarak ayırma ---
# %80 eğitim, %20 test olarak bölüyoruz
# random_state sabit tutularak tekrarlanabilir sonuçlar elde ediyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Egitim seti: {X_train.shape[0]} ornek")
print(f"Test seti: {X_test.shape[0]} ornek")
print()

# --- Adım 5: Random Forest Classifier modeli oluşturma ve eğitme ---
# n_estimators: Ormandaki ağaç sayısı (100 varsayılan)
# random_state: Tekrarlanabilirlik için sabit değer
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model egitildi.")
print()

# --- Adım 6: Test seti üzerinde tahmin yapma ---
y_tahmin = model.predict(X_test)

# --- Adım 7: Model performansının değerlendirilmesi ---
# pos_label=1 → LabelEncoder sonrası: B=0, M=1 (Malignant pozitif sınıf)
accuracy  = accuracy_score(y_test, y_tahmin)
precision = precision_score(y_test, y_tahmin, pos_label=1)
recall    = recall_score(y_test, y_tahmin, pos_label=1)
f1        = f1_score(y_test, y_tahmin, pos_label=1)

print("--- Sonuclar ---")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print()

# Detaylı sınıflandırma raporu
print("Siniflandirma Raporu:")
print(classification_report(y_test, y_tahmin, target_names=["Benign (Iyi Huylu)", "Malignant (Kotu Huylu)"]))
