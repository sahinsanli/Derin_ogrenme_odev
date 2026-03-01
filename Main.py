


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("data - data.csv")


print("Veri seti okundu.")
print(f"Toplam {df.shape[0]} satir ve {df.shape[1]} sutun var.")
print()


hedef_adaylari = ["diagnosis", "target", "class", "label"]
hedef_sutun = None

for aday in hedef_adaylari:
    if aday in df.columns:
        hedef_sutun = aday
        break


if hedef_sutun is None:
    hedef_sutun = df.columns[-1]
    print(f"Hedef sutun bulunamadi, son sutun kullaniliyor: '{hedef_sutun}'")

print(f"Hedef degisken: '{hedef_sutun}'")
print(f"Sinif dagilimi:\n{df[hedef_sutun].value_counts()}")
print()


ozellik_sutunlari = [col for col in df.columns if col != hedef_sutun and col.lower() != "id"]

X = df[ozellik_sutunlari]  
y = df[hedef_sutun]         


if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    print(f"Etiketler sayisala donusturuldu: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print()

print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Egitim seti: {X_train.shape[0]} ornek")
print(f"Test seti: {X_test.shape[0]} ornek")
print()


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model egitildi.")
print()


y_tahmin = model.predict(X_test)


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


print("Siniflandirma Raporu:")
print(classification_report(y_test, y_tahmin, target_names=["Benign (Iyi Huylu)", "Malignant (Kotu Huylu)"]))
