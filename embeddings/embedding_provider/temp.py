import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier

# 1. Chargement des données
path = "/your/base/path" # <-- Remplace par ton chemin réel
df = pd.read_csv(f"{path}/Datasets/03 Health_and_Sleep.csv")

# 2. Nettoyage et préparation
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["Sleep Disorder"] = df["Sleep Disorder"].replace({
    "None": "Pas de trouble",
    "Sleep Apnea": "Apnée",
    "Insomnia": "Insomnie"
})
df.drop(columns=["Person ID"], inplace=True)
df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True).astype(float)
df.drop(columns=["Blood Pressure"], inplace=True)

# 3. Séparation X/Y
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"].astype("category").cat.codes

# 4. Détection des colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 5. Pipeline de prétraitement
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Modèle DNN à 2 couches
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42))
])

# 7. Courbe d'apprentissage
train_sizes = np.linspace(0.1, 1.0, 8)

train_sizes_abs, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=train_sizes,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = np.nanmean(train_scores, axis=1)
train_std = np.nanstd(train_scores, axis=1)
test_mean = np.nanmean(test_scores, axis=1)
test_std = np.nanstd(test_scores, axis=1)

# 8. Affichage
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_mean, 'o-', label="Training score")
plt.plot(train_sizes_abs, test_mean, 's-', label="Validation score")
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.title("Courbe d'Apprentissage à DNN (2 couches cachées)")
plt.xlabel("Taille de l'échantillon d'entraénement")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
