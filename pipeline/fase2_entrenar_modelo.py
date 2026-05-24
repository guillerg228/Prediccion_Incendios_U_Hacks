"""
FASE 2 — Entrenamiento del modelo Random Forest + SHAP
Hackathon - Predicción de Incendios

Lo que hace:
- Carga el dataset generado en Fase 1
- Entrena un Random Forest
- Evalúa el modelo (accuracy, precision, recall)
- Calcula importancia de variables con SHAP
- Guarda el modelo entrenado como modelo_incendios.pkl

Requiere:
    pip install scikit-learn shap pandas numpy matplotlib joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
CSV_PATH  = os.path.join(DATA_DIR, "dataset_entrenamiento.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_incendios.pkl")

FEATURES = [
    "temperatura",
    "humedad",
    "viento",
    "precipitacion",
    "cobertura_num",
    "en_anp_num",
    "mes",
]

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================

def cargar_datos():
    print("📂 Cargando dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"   ✅ {len(df)} registros cargados")
    print(f"   Positivos: {(df['label']==1).sum()}")
    print(f"   Negativos: {(df['label']==0).sum()}")
    return df

# =============================================================================
# 2. ENTRENAR MODELO
# =============================================================================

def entrenar(df):
    print("\n🤖 Entrenando Random Forest...")

    X = df[FEATURES].copy()
    y = df["label"]

    # Llenar cualquier nulo restante
    X = X.fillna(X.median())

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",  # compensa desbalance positivos/negativos
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    print("   ✅ Modelo entrenado")

    # Evaluación
    print("\n📊 Evaluación del modelo:")
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Sin incendio", "Incendio"]))
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Matriz de confusión:")
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

    return modelo, X_train, X_test

# =============================================================================
# 3. ANÁLISIS SHAP
# =============================================================================

def analizar_shap(modelo, X_train, X_test):
    print("\n🔍 Calculando valores SHAP...")

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)

    # Compatibilidad con distintas versiones de SHAP
    # Versiones antiguas: lista [clase0, clase1]
    # Versiones nuevas: array 3D (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "values"):
        sv = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Importancia global
    importancia = pd.DataFrame({
        "feature":    FEATURES,
        "importancia": np.abs(sv).mean(axis=0)
    }).sort_values("importancia", ascending=False)

    print("\n   Importancia de variables:")
    for _, row in importancia.iterrows():
        barra = "█" * int(row["importancia"] * 100)
        print(f"   {row['feature']:20s} {barra} {row['importancia']:.3f}")

    # Guardar gráfica
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test.values, feature_names=FEATURES, show=False)
    plt.tight_layout()
    grafica_path = os.path.join(os.path.dirname(MODEL_PATH), "shap_importancia.png")
    plt.savefig(grafica_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ✅ Gráfica SHAP guardada: {grafica_path}")

    return explainer, importancia

# =============================================================================
# 4. GUARDAR MODELO
# =============================================================================

def guardar_modelo(modelo, explainer):
    joblib.dump({
        "modelo":    modelo,
        "explainer": explainer,
        "features":  FEATURES,
    }, MODEL_PATH)
    print(f"\n💾 Modelo guardado: {MODEL_PATH}")

# =============================================================================
# MAIN
# =============================================================================

def run():
    print("\n" + "="*60)
    print("  FASE 2 — ENTRENAMIENTO DEL MODELO")
    print("="*60 + "\n")

    df = cargar_datos()
    modelo, X_train, X_test = entrenar(df)
    explainer, importancia = analizar_shap(modelo, X_train, X_test)
    guardar_modelo(modelo, explainer)

    print("\n" + "="*60)
    print("  FASE 2 COMPLETADA — ejecuta fase3_predecir.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    run()
