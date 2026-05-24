"""
FASE 3 — Predicción de zonas de riesgo con el modelo entrenado
Hackathon - Predicción de Incendios

Lo que hace:
- Genera una cuadrícula de puntos sobre Morelos y EdoMex
- Consulta clima ACTUAL de Open-Meteo para cada punto
- Cruza con cobertura vegetal y ANP
- El modelo predice probabilidad de incendio
- SHAP explica por qué cada zona es de riesgo
- Guarda predicciones en SQL Server (tabla: predicciones_riesgo)

Requiere:
    pip install requests geopandas shapely pandas numpy scikit-learn shap joblib pyodbc
"""

import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import shap
import pyodbc
import os
from shapely.geometry import Point
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_incendios.pkl")

# SQL Server
DB_SERVER   = "servidor-incedios.database.windows.net"  # cambia por tu servidor
DB_NAME     = "incedios100real"                              # cambia por tu base de datos
DB_USER     = "urigod"                            # cambia por tu usuario
DB_PASSWORD = "Castelan1-"

CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
)

GEOJSON = {
    "anp_morelos":       os.path.join(DATA_DIR, "Areas_Naturales_Morelos.geojson"),
    "anp_edomex":        os.path.join(DATA_DIR, "Areas_Naturales_EdoMex.geojson"),
    "cobertura_morelos": os.path.join(DATA_DIR, "Cobertura_Morelos.geojson"),
    "cobertura_edomex":  os.path.join(DATA_DIR, "Cobertura_EdoMex.geojson"),
}

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# Resolución de la cuadrícula (0.1° ~ 11km)
GRID_STEP = 0.1

BBOXES = {
    "morelos": (-99.6, 18.4, -98.6, 19.1),
    "edomex":  (-100.6, 18.7, -98.4, 20.3),
}

COBERTURA_MAP = {
    "bosque":   5,
    "selva":    4,
    "matorral": 3,
    "pastizal": 2,
    "agricola": 1,
    "urbano":   0,
    "agua":     0,
}

# =============================================================================
# 1. CARGAR MODELO Y CAPAS
# =============================================================================

def cargar_modelo():
    print("🤖 Cargando modelo entrenado...")
    data = joblib.load(MODEL_PATH)
    print("   ✅ Modelo cargado")
    return data["modelo"], data["explainer"], data["features"]

def cargar_capas():
    print("📂 Cargando capas GeoJSON...")
    capas = {}
    for nombre, ruta in GEOJSON.items():
        if os.path.exists(ruta):
            capas[nombre] = gpd.read_file(ruta).to_crs("EPSG:4326")
        else:
            capas[nombre] = None
    return capas

# =============================================================================
# 2. GENERAR CUADRÍCULA
# =============================================================================

def generar_cuadricula(estado, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = np.arange(lat_min, lat_max, GRID_STEP)
    lons = np.arange(lon_min, lon_max, GRID_STEP)

    puntos = [{"lat": lat, "lon": lon, "estado": estado}
              for lat in lats for lon in lons]

    print(f"   📍 {estado}: {len(puntos)} puntos en cuadrícula")
    return pd.DataFrame(puntos)

# =============================================================================
# 3. CLIMA ACTUAL
# =============================================================================

def obtener_clima_actual(lat, lon):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone":  "America/Mexico_City",
    }
    try:
        resp = requests.get(OPENMETEO_URL, params=params, timeout=10)
        data = resp.json().get("current", {})
        return {
            "temperatura":   data.get("temperature_2m"),
            "humedad":       data.get("relative_humidity_2m"),
            "viento":        data.get("wind_speed_10m"),
            "precipitacion": data.get("precipitation"),
        }
    except:
        return {"temperatura": None, "humedad": None, "viento": None, "precipitacion": None}

def enriquecer_clima(df):
    print("🌤️  Obteniendo clima actual...")

    # Agrupar por coordenadas redondeadas
    df["lat_r"] = df["lat"].round(1)
    df["lon_r"] = df["lon"].round(1)
    puntos_unicos = df[["lat_r", "lon_r"]].drop_duplicates()

    cache = {}
    for _, row in puntos_unicos.iterrows():
        key = (row["lat_r"], row["lon_r"])
        cache[key] = obtener_clima_actual(row["lat_r"], row["lon_r"])

    for col in ["temperatura", "humedad", "viento", "precipitacion"]:
        df[col] = df.apply(lambda r: cache[(r["lat_r"], r["lon_r"])][col], axis=1)

    print("   ✅ Clima obtenido")
    return df

# =============================================================================
# 4. CRUCE ESPACIAL
# =============================================================================

def cruzar_espacial(df, capas):
    print("🗺️  Cruzando con cobertura y ANP...")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )

    # Cobertura
    coberturas = [c for k, c in capas.items() if "cobertura" in k and c is not None]
    if coberturas:
        cob_all = gpd.GeoDataFrame(pd.concat(coberturas), crs="EPSG:4326")
        col_desc = "DESC_SAMOF" if "DESC_SAMOF" in cob_all.columns else cob_all.columns[1]
        joined = gpd.sjoin(gdf, cob_all[["geometry", col_desc]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")].reindex(gdf.index)
        df["cobertura"] = joined[col_desc].values
    else:
        df["cobertura"] = None

    # ANP
    anps = [c for k, c in capas.items() if "anp" in k and c is not None]
    if anps:
        anp_all = gpd.GeoDataFrame(pd.concat(anps), crs="EPSG:4326")
        joined2 = gpd.sjoin(gdf, anp_all[["geometry", "NOMBRE"]], how="left", predicate="within")
        joined2 = joined2[~joined2.index.duplicated(keep="first")].reindex(gdf.index)
        df["en_anp"]     = (~joined2["NOMBRE"].isna()).values
        df["nombre_anp"] = joined2["NOMBRE"].values
    else:
        df["en_anp"]     = False
        df["nombre_anp"] = None

    def cobertura_a_num(c):
        if pd.isna(c): return 1
        c = str(c).lower()
        for key, val in COBERTURA_MAP.items():
            if key in c: return val
        return 1

    df["cobertura_num"] = df["cobertura"].apply(cobertura_a_num)
    df["en_anp_num"]    = df["en_anp"].astype(int)
    df["mes"]           = datetime.now().month

    print("   ✅ Cruce completado")
    return df

# =============================================================================
# 5. PREDECIR Y EXPLICAR CON SHAP
# =============================================================================

def predecir(df, modelo, explainer, features):
    print("⚠️  Prediciendo zonas de riesgo...")

    X = df[features].fillna(df[features].median()).reset_index(drop=True)

    # Probabilidad de incendio
    df = df.reset_index(drop=True)
    df["prob_incendio"] = modelo.predict_proba(X)[:, 1]

    # Nivel de riesgo
    def nivel(p):
        if p >= 0.75:  return "CRÍTICO"
        elif p >= 0.50: return "ALTO"
        elif p >= 0.25: return "MEDIO"
        else:           return "BAJO"

    df["nivel_riesgo"] = df["prob_incendio"].apply(nivel)

    # SHAP — explicación por punto (compatible con todas las versiones)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "values"):
        sv = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Guardar el factor más importante por punto
    sv_df = pd.DataFrame(sv, columns=features).reset_index(drop=True)
    df["factor_principal"]  = sv_df.abs().idxmax(axis=1).values
    df["factor_valor_shap"] = sv_df.abs().max(axis=1).round(3).values

    # Resumen
    print(f"\n   📊 Distribución de riesgo:")
    print(df["nivel_riesgo"].value_counts().to_string())

    criticos = df[df["nivel_riesgo"] == "CRÍTICO"]
    if not criticos.empty:
        print(f"\n   🚨 Zonas CRÍTICAS:")
        for _, r in criticos.head(5).iterrows():
            anp = f" — dentro de {r['nombre_anp']}" if r["en_anp"] else ""
            print(f"   ({r['lat']:.2f}, {r['lon']:.2f}) {r['estado']}{anp}")
            print(f"   Factor principal: {r['factor_principal']} ({r['prob_incendio']:.0%})")

    return df

# =============================================================================
# 6. GUARDAR EN SQL SERVER
# =============================================================================

def guardar_predicciones(df):
    print("\n💾 Guardando predicciones en SQL Server...")

    cols = [
        "lat", "lon", "estado",
        "temperatura", "humedad", "viento", "precipitacion",
        "cobertura", "en_anp", "nombre_anp",
        "prob_incendio", "nivel_riesgo",
        "factor_principal", "factor_valor_shap"
    ]
    df_save = df[cols].copy()
    df_save["fecha_prediccion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Solo guardar los de riesgo MEDIO o superior para no saturar la BD
    df_save = df_save[df_save["nivel_riesgo"].isin(["MEDIO", "ALTO", "CRÍTICO"])]

    if df_save.empty:
        print("   ℹ️  Sin zonas de riesgo medio o superior hoy.")
        return

    conn = pyodbc.connect(CONNECTION_STRING, timeout=30)
    cursor = conn.cursor()

    tabla = "predicciones_riesgo"
    cols_def = ", ".join([f"[{c}] NVARCHAR(255)" for c in df_save.columns])
    cursor.execute(
        f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tabla}') "
        f"CREATE TABLE {tabla} ({cols_def})"
    )
    conn.commit()

    placeholders = ", ".join(["?" for _ in df_save.columns])
    cols_str = ", ".join([f"[{c}]" for c in df_save.columns])
    insert_sql = f"INSERT INTO {tabla} ({cols_str}) VALUES ({placeholders})"

    for _, row in df_save.iterrows():
        cursor.execute(insert_sql, [str(v) if v is not None else None for v in row])

    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ {len(df_save)} zonas de riesgo guardadas en tabla '{tabla}'")

# =============================================================================
# MAIN
# =============================================================================

def run():
    print("\n" + "="*60)
    print("  FASE 3 — PREDICCIÓN DE ZONAS DE RIESGO")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    modelo, explainer, features = cargar_modelo()
    capas = cargar_capas()

    print("\n📍 Generando cuadrícula de puntos...")
    dfs = []
    for estado, bbox in BBOXES.items():
        dfs.append(generar_cuadricula(estado, bbox))
    df = pd.concat(dfs, ignore_index=True)

    df = enriquecer_clima(df)
    df = cruzar_espacial(df, capas)
    df = predecir(df, modelo, explainer, features)
    guardar_predicciones(df)

    print("\n" + "="*60)
    print("  FASE 3 COMPLETADA")
    print("  Conecta Power BI a la tabla 'predicciones_riesgo'")
    print("="*60 + "\n")

if __name__ == "__main__":
    run()
