# MLP_Flask

Çoklu Doğrusal Regresyon ve Flask GUI uygulaması
Adınız: Emmanuel
Soyadınız: HAKIRUWIZERA
Okul Numaranız: 2440631002
GitHub Repo Bağlantısı: https://github.com/emmanuelhakiruwizera/CNN_siniflandirma
Introduction
Amaç: Doğu Afrika bölgesinde başlıca mahsuller (Mısır, Pirinç ve Soya fasulyesi) için verimliliği tahmin etmek amacıyla Çoklu Doğrusal Regresyon modeli kullanmak.

Title: Smart Farming for Yield Prediction
URL: Kaggle Dataset: https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction
Explore Library

[ ]

# Gerekli kütüphanelerin içe aktarılması:
# sklearn: Makine öğrenmesi algoritmaları ve veri ön işleme araçları için.
# pandas: Veri analizi ve DataFrame yapıları için.
# numpy: Sayısal hesaplamalar ve dizi işlemleri için.
# train_test_split: Veriyi eğitim ve test setlerine ayırmak için.
# OneHotEncoder: Nominal kategorik değişkenleri One-Hot kodlamaya dönüştürmek için.
# StandardScaler: Sayısal değişkenleri standart ölçeklemeye tabi tutmak için.
# ColumnTransformer: Farklı veri tipleri için (kategorik ve say

import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


[ ]

# Veri kümesini Colab'a yükledim.
## Veri kümesini CSV formatında düzenledikten sonra,
# tarımsal girdi parametrelerini dikkate alarak Doğu Afrika'ya ait ilgilendiğim alanı çıkardım.

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/Colab_MLP/Smart_Farming/smart_farming.csv"


[ ]

# Hedef değişken (y): yield_kg_per_hectare — modelin tahmin edeceği çıktı.
# Giriş özellikleri (X):
#   - Kategorik (nominal): region, crop_type, irrigation_type, fertilizer_type
#   - Sayısal: soil_moisture_%, soil_pH, temperature_C, rainfall_mm,
#              humidity_%, sunlight_hours, pesticide_usage_ml, total_days

# Mantık kontrolü: DataFrame'de gerekli tüm sütunların mevcut olduğunu doğrular;
# eksik sütun varsa hata verir.
# Veri derleme:
#   - X: Belirtilen kategorik + sayısal özellik sütunlarından oluşturulur (kopyalanarak güvenlik sağlanır).


[ ]
print("DataFrame Info:")
df.info()

[ ]
print("\nDataFrame Description (numerical columns):")
display(df.describe())

[ ]
import pandas as pd

# Load data
df = pd.read_csv('/content/drive/MyDrive/Colab_MLP/Smart_Farming/smart farmimg.csv')

# Target and feature definitions for my project yield productivity based on
# smart farming dataset
TARGET_COL = "yield_kg_per_hectare"
CATEGORICAL_COLS = ["region", "crop_type", "irrigation_type", "fertilizer_type"]
NUMERIC_COLS = [
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm",
    "humidity_%", "sunlight_hours", "pesticide_usage_ml", "total_days"
]

# Sanity checks
print("Columns:", list(df.columns))
for col in [TARGET_COL] + CATEGORICAL_COLS + NUMERIC_COLS:
    assert col in df.columns, f"Missing column: {col}"

X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
y = df[TARGET_COL].copy()
print

[ ]

Start coding or generate with AI.
Train/Test Split & Preprocessing (version‑safe One‑Hot + Scaling)

[ ]

# Veri eğitim ve test setlerine %80-%20 oranında ayrıldı.
# scikit-learn sürümüne göre OneHotEncoder parametreleri belirlendi:
#   - drop='first': Multicollinearity önlemek için ilk kategori düşürülür.
#   - handle_unknown='ignore': Tahmin sırasında bilinmeyen kategoriler hata vermez.
#   - sparse_output veya sparse: Sürüm uyumluluğu için ayarlandı.
# Kategorik değişkenler için OneHotEncoder, sayısal değişkenler için StandardScaler kullanıldı.
# ColumnTransformer ile bu ön işleme adımları birleştirildi.
# Preprocessor yalnızca eğitim verisi üzerinde fit edildi (veri sızıntısını önlemek için).
# Eğitim ve


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Choose OneHotEncoder arg based on scikit-learn version
from sklearn import __version__ as skver
major, minor = map(int, skver.split(".")[:2])
use_sparse_output = (major, minor) >= (1, 2)

ohe_kwargs = {"drop": "first", "handle_unknown": "ignore"}
if use_sparse_output:
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False  # for older versions

cat_transformer = OneHotEncoder(**ohe_kwargs)
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, CATEGORICAL_COLS),
        ("num", num_transformer, NUMERIC_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=True
)

# Fit on *train* only to avoid leakage
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print("Encoded feature count:", len(feature_names))
print("First 15 encoded features:", feature_names[:15])

Backward Elimination (p‑values via OLS)

[ ]

# Geriye Doğru Eleme:
# OLS modeli kurulur, p-değerleri incelenir.
# p > alpha (0.05) olan en büyük p-değerli özellik her adımda çıkarılır.
# Tüm kalan özellikler anlamlı olduğunda durur.
# Çıktı: Seçilen özellik indeksleri ve adları


import statsmodels.api as sm

def backward_elimination(X_enc, y, feature_names, alpha=0.05):
    """
    Iteratively remove the feature with highest p-value above alpha (excluding intercept).
    Returns: selected_indices, selected_feature_names
    """
    X_be = sm.add_constant(X_enc, has_constant='add')
    current_indices = list(range(X_enc.shape[1]))
    current_feature_names = list(feature_names)

    while True:
        ols_model = sm.OLS(y, X_be).fit()
        pvalues = ols_model.pvalues
        p_no_const = pvalues.iloc[1:]  # drop intercept

        max_p = float(p_no_const.max())
        max_idx_name = p_no_const.idxmax()

        if max_p <= alpha:
            break

        # Identify positions
        drop_pos_be = list(pvalues.index).index(max_idx_name)  # const at 0
        drop_pos_X = drop_pos_be - 1

        print(f"Eliminating '{current_feature_names[drop_pos_X]}' (p={max_p:.4f})")

        # Drop column from X_be
        mask = np.ones(X_be.shape[1], dtype=bool)
        mask[drop_pos_be] = False
        X_be = X_be[:, mask]

        # Update trackers
        del current_feature_names[drop_pos_X]
        del current_indices[drop_pos_X]

        if len(current_indices) == 0:
            print("All features eliminated; stopping.")
            break

    final_model = sm.OLS(y, X_be).fit()
    print("\nFinal OLS summary after Backward Elimination:\n")
    print(final_model.summary())

    return current_indices, current_feature_names

selected_indices, selected_feature_names = backward_elimination(
    X_train_enc, y_train, feature_names, alpha=0.05
)

print("Selected features:", selected_feature_names)


Train Multiple Linear Regression & Evaluate

[ ]


# Çoklu Doğrusal Regresyon:
# Seçilen özelliklerle LinearRegression modeli eğitilir.
# Test setinde tahmin yapılır ve performans metrikleri hesaplanır:
#   - R²: Modelin açıklama gücü
#   - MAE: Ortalama mutlak hata
#   - MSE: Ortalama kare hata


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Use only selected features
X_train_sel = X_train_enc[:, selected_indices]
X_test_sel  = X_test_enc[:, selected_indices]

# Train
lr = LinearRegression()
lr.fit(X_train_sel, y_train)

# Predict & evaluate
y_pred = lr.predict(X_test_sel)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# Preview a few predictions
import pandas as pd
preview = pd.DataFrame({
    "pred": pd.Series(y_pred[:10]).round(2),
    "actual": pd.Series(y_test.values[:10]).round(2)
})
preview


Save the Model Bundle (.pkl) & Download

[ ]
}
joblib.dump(bundle, MODEL_BUNDLE_PATH)
print("Saved:", MODEL_BUNDLE_PATH)

# Download to your machine
from google.colab import files
files.download(MODEL_BUNDLE_PATH)

Create a Simple Flask App (app.py)

[ ]

%%writefile app.py
from flask import Flask, request, render_template_string
import pandas as pd
import joblib

BUNDLE_PATH = "model_bundle.pkl"
bundle = joblib.load(BUNDLE_PATH)

preprocessor = bundle["preprocessor"]
selected_indices = bundle["selected_indices"]
input_columns = bundle["input_columns"]
categorical_cols = set(bundle["categorical_cols"])
numeric_cols = set(bundle["numeric_cols"])
model = bundle["model"]
target = bundle.get("target", "prediction")

app = Flask(__name__)

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Yield Prediction (Multiple Linear Regression)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .row { margin-bottom: 10px; }
        label { display: inline-block; width: 220px; }
        input { width: 280px; padding: 6px; }
        .btn { padding: 8px 14px; }
        .card { margin-top: 20px; padding: 16px; border: 1px solid #ddd; border-radius: 6px; }
        .note { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h2>Yield Prediction — Multiple Linear Regression</h2>
    <p class="note">Enter farm features below. Categorical fields accept free text; numeric fields expect numbers.</p>
    /predict
        {% for col in input_columns %}
            <div class="row">
                <label for="{{col}}">{{col}}</label>
                {% if col in numeric_cols %}
                    <input type="text" id="{{col}}" name="{{col}}" placeholder="e.g., 23.5" required>
                {% else %}
                    <input type="text" id="{{col}}" name="{{col}}" placeholder="e.g., East Africa / Rice / Drip / Organic" required>
                {% endif %}
            </div>
        {% endfor %}
        <button class="btn" type="submit">Predict</button>
    </form>

    {% if prediction is defined %}
    <div class="card">
        <h3>Prediction Result</h3>
        <p><strong>{{ target }}:</strong> {{ prediction }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(FORM_HTML, input_columns=input_columns,
                                  numeric_cols=numeric_cols, target=target)

@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    for col in input_columns:
        val = request.form.get(col)
        if col in numeric_cols:
            try:
                data[col] = float(val)
            except:
                return f"Invalid numeric value for {col}: {val}", 400
        else:
            data[col] = str(val)

    X_input = pd.DataFrame([data], columns=input_columns)
    X_enc = preprocessor.transform(X_input)
    X_sel = X_enc[:, selected_indices]
    pred = float(model.predict(X_sel)[0])

    return render_template_string(FORM_HTML, input_columns=input_columns,
                                  numeric_cols=numeric_cols, target=target,
                                  prediction=round(pred, 3))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


References
Uguz, S., & Ipek, O. (2022). Prediction of the parameters affecting the performance of compact heat exchangers with an innovative design using machine learning techniques. Journal of Intelligent Manufacturing, 33(5), 1393–1417. https://doi.org/10.1007/s10845-020-01729-0

scikit‑learn developers. (2024). scikit‑learn user guide — OneHotEncoder, ColumnTransformer, StandardScaler, LinearRegression, RidgeCV, LassoCV, and metrics (R², MAE, MSE) [Software documentation].

Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to linear regression analysis (5th ed.). Wiley.

https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction

Colab paid products - Cancel contracts here
