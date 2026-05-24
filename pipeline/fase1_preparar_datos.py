"""
FASE 1 — Preparación de datos para modelo de predicción
Hackathon - Predicción de Incendios

Lo que hace:
- Toma incendios históricos de los GeoJSON como ejemplos positivos (label=1)
- Genera puntos aleatorios sin incendio como ejemplos negativos (label=0)
- Para cada punto obtiene clima de Open-Meteo (histórico o actual)
- Cruza con cobertura vegetal y ANP
- Guarda el dataset listo para entrenar en dataset_entrenamiento.csv

Requiere:
    pip install requests geopandas shapely pandas numpy scikit-learn
"""

import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from datetime import datetime, timedelta
import os
import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

GEOJSON = {
    "anp_morelos":       os.path.join(DATA_DIR, "Areas_Naturales_Morelos.geojson"),
    "anp_edomex":        os.path.join(DATA_DIR, "Areas_Naturales_EdoMex.geojson"),
    "cobertura_morelos": os.path.join(DATA_DIR, "Cobertura_Morelos.geojson"),
    "cobertura_edomex":  os.path.join(DATA_DIR, "Cobertura_EdoMex.geojson"),
    "incendios_morelos": os.path.join(DATA_DIR, "Incendios_Morelos.geojson"),
    "incendios_edomex":  os.path.join(DATA_DIR, "Incendios_EdoMex.geojson"),
}

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"  # API histórica
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "data", "dataset_entrenamiento.csv")

# Cuántos puntos negativos generar por estado
NEGATIVOS_POR_ESTADO = 500

# =============================================================================
# 1. CARGAR CAPAS
# =============================================================================

def cargar_capas():
    print("📂 Cargando GeoJSON...")
    capas = {}
    for nombre, ruta in GEOJSON.items():
        if os.path.exists(ruta):
            capas[nombre] = gpd.read_file(ruta).to_crs("EPSG:4326")
            print(f"   ✅ {nombre}: {len(capas[nombre])} features")
        else:
            print(f"   ⚠️  No encontrado: {ruta}")
            capas[nombre] = None
    return capas

# =============================================================================
# 2. EXTRAER INCENDIOS HISTÓRICOS (POSITIVOS)
# =============================================================================

def extraer_positivos(capas):
    print("\n🔥 Extrayendo incendios históricos (label=1)...")
    dfs = []

    for estado in ["morelos", "edomex"]:
        gdf = capas.get(f"incendios_{estado}")
        if gdf is None:
            continue

        df = pd.DataFrame()
        # Extraer centroide si la geometría no es punto
        geom = gdf.geometry
        if not all(geom.geom_type == "Point"):
            geom = geom.centroid
        df["lat"] = geom.y
        df["lon"] = geom.x
        df["estado"] = estado
        df["label"]  = 1

        # Intentar extraer fecha si existe en los atributos
        for col in gdf.columns:
            if "fecha" in col.lower() or "date" in col.lower() or "anio" in col.lower() or "año" in col.lower():
                df["fecha"] = pd.to_datetime(gdf[col], errors="coerce")
                break
        
        if "fecha" not in df.columns:
            # Si no hay fecha, asignar fechas aleatorias en temporada de incendios
            # (marzo-mayo son los meses más comunes en México)
            años = np.random.choice(range(2020, 2025), size=len(df))
            meses = np.random.choice([2, 3, 4, 5], size=len(df))
            dias = np.random.randint(1, 28, size=len(df))
            df["fecha"] = pd.to_datetime({
                "year": años, "month": meses, "day": dias
            }, errors="coerce")

        dfs.append(df)
        print(f"   ✅ {estado}: {len(df)} incendios históricos")

    return pd.concat(dfs, ignore_index=True)

# =============================================================================
# 3. GENERAR PUNTOS NEGATIVOS
# =============================================================================

def generar_negativos(capas, n_por_estado=NEGATIVOS_POR_ESTADO):
    print(f"\n⬜ Generando puntos sin incendio (label=0)...")
    dfs = []

    # Bounding boxes de cada estado
    bboxes = {
        "morelos": (-99.6, 18.4, -98.6, 19.1),
        "edomex":  (-100.6, 18.7, -98.4, 20.3),
    }

    for estado, (lon_min, lat_min, lon_max, lat_max) in bboxes.items():
        gdf_incendios = capas.get(f"incendios_{estado}")
        
        puntos_negativos = []
        intentos = 0
        
        while len(puntos_negativos) < n_por_estado and intentos < n_por_estado * 10:
            intentos += 1
            lat = np.random.uniform(lat_min, lat_max)
            lon = np.random.uniform(lon_min, lon_max)
            punto = Point(lon, lat)

            # Verificar que no coincida con un incendio histórico (buffer 0.01° ~ 1km)
            if gdf_incendios is not None:
                distancias = gdf_incendios.geometry.distance(punto)
                if distancias.min() < 0.01:
                    continue

            puntos_negativos.append({"lat": lat, "lon": lon})

        df = pd.DataFrame(puntos_negativos)
        df["estado"] = estado
        df["label"]  = 0

        # Fechas en todos los meses para que el modelo aprenda por clima y cobertura
        años = np.random.choice(range(2020, 2025), size=len(df))
        meses = np.random.choice(range(1, 13), size=len(df))
        dias = np.random.randint(1, 28, size=len(df))
        df["fecha"] = pd.to_datetime({
            "year": años, "month": meses, "day": dias
        }, errors="coerce")

        dfs.append(df)
        print(f"   ✅ {estado}: {len(df)} puntos negativos generados")

    return pd.concat(dfs, ignore_index=True)

# =============================================================================
# 4. OBTENER CLIMA HISTÓRICO
# =============================================================================

def obtener_clima_historico(lat, lon, fecha):
    """
    Consulta Open-Meteo Archive API para clima en una fecha específica.
    """
    try:
        fecha_str = fecha.strftime("%Y-%m-%d") if pd.notna(fecha) else "2023-04-15"
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": fecha_str,
            "end_date":   fecha_str,
            "daily":      "temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum",
            "timezone":   "America/Mexico_City",
        }
        resp = requests.get(OPENMETEO_URL, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json().get("daily", {})
        return {
            "temperatura": data.get("temperature_2m_max", [None])[0],
            "humedad":     data.get("relative_humidity_2m_mean", [None])[0],
            "viento":      data.get("wind_speed_10m_max", [None])[0],
            "precipitacion": data.get("precipitation_sum", [None])[0],
        }
    except:
        return None

def enriquecer_con_clima(df):
    print(f"\n🌤️  Obteniendo clima histórico para {len(df)} puntos...")
    print("   (Esto puede tardar varios minutos, agrupando puntos cercanos...)")

    # Agrupar por coordenadas redondeadas + fecha para reducir llamadas API
    df["lat_r"]    = df["lat"].round(1)
    df["lon_r"]    = df["lon"].round(1)
    df["fecha_r"]  = df["fecha"].dt.strftime("%Y-%m-%d")

    grupos = df[["lat_r", "lon_r", "fecha_r"]].drop_duplicates()
    print(f"   {len(grupos)} consultas únicas a Open-Meteo")

    clima_cache = {}
    for i, (_, row) in enumerate(grupos.iterrows()):
        key = (row["lat_r"], row["lon_r"], row["fecha_r"])
        fecha = pd.to_datetime(row["fecha_r"])
        clima = obtener_clima_historico(row["lat_r"], row["lon_r"], fecha)
        clima_cache[key] = clima if clima else {
            "temperatura": None, "humedad": None,
            "viento": None, "precipitacion": None
        }
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(grupos)} consultas completadas...")
            time.sleep(0.5)  # respetar rate limit

    for col in ["temperatura", "humedad", "viento", "precipitacion"]:
        df[col] = df.apply(
            lambda r: clima_cache.get(
                (r["lat_r"], r["lon_r"], r["fecha_r"]), {}
            ).get(col), axis=1
        )

    print(f"   ✅ Clima agregado")
    return df

# =============================================================================
# 5. CRUZAR CON COBERTURA Y ANP
# =============================================================================

def cruzar_espacial(df, capas):
    print("\n🗺️  Cruzando con cobertura vegetal y ANP...")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    # Cobertura vegetal
    coberturas = []
    for estado in ["morelos", "edomex"]:
        cob = capas.get(f"cobertura_{estado}")
        if cob is not None:
            coberturas.append(cob)

    if coberturas:
        cob_all = pd.concat(coberturas, ignore_index=True)
        cob_gdf = gpd.GeoDataFrame(cob_all, crs="EPSG:4326")
        col_desc = "DESC_SAMOF" if "DESC_SAMOF" in cob_gdf.columns else cob_gdf.columns[1]

        joined = gpd.sjoin(gdf, cob_gdf[["geometry", col_desc]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")].reindex(gdf.index)
        df["cobertura"] = joined[col_desc].values
    else:
        df["cobertura"] = None

    # ANP
    anps = []
    for estado in ["morelos", "edomex"]:
        anp = capas.get(f"anp_{estado}")
        if anp is not None:
            anps.append(anp)

    if anps:
        anp_all = pd.concat(anps, ignore_index=True)
        anp_gdf = gpd.GeoDataFrame(anp_all, crs="EPSG:4326")
        joined2 = gpd.sjoin(gdf, anp_gdf[["geometry", "NOMBRE"]], how="left", predicate="within")
        joined2 = joined2[~joined2.index.duplicated(keep="first")].reindex(gdf.index)
        df["en_anp"] = (~joined2["NOMBRE"].isna()).values
    else:
        df["en_anp"] = False

    print("   ✅ Cruce completado")
    return df

# =============================================================================
# 6. FEATURES FINALES
# =============================================================================

# Mapeo de cobertura a número
COBERTURA_MAP = {
    "bosque":    5,
    "selva":     4,
    "matorral":  3,
    "pastizal":  2,
    "agricola":  1,
    "urbano":    0,
    "agua":      0,
}

def preparar_features(df):
    print("\n⚙️  Preparando features finales...")

    # Mes del año (estacionalidad)
    df["mes"] = df["fecha"].dt.month

    # Cobertura a número
    def cobertura_a_num(c):
        if pd.isna(c):
            return 1
        c = str(c).lower()
        for key, val in COBERTURA_MAP.items():
            if key in c:
                return val
        return 1

    df["cobertura_num"] = df["cobertura"].apply(cobertura_a_num)
    df["en_anp_num"]    = df["en_anp"].astype(int)

    # Llenar nulos con medianas
    for col in ["temperatura", "humedad", "viento", "precipitacion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    print(f"   ✅ Dataset final: {len(df)} registros")
    print(f"   Positivos (incendio=1): {(df['label']==1).sum()}")
    print(f"   Negativos (incendio=0): {(df['label']==0).sum()}")

    return df

# =============================================================================
# MAIN
# =============================================================================

def run():
    print("\n" + "="*60)
    print("  FASE 1 — PREPARACIÓN DE DATOS")
    print("="*60 + "\n")

    capas = cargar_capas()

    # Positivos y negativos
    df_pos = extraer_positivos(capas)
    df_neg = generar_negativos(capas)
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # Clima
    df = enriquecer_con_clima(df)

    # Cruce espacial
    df = cruzar_espacial(df, capas)

    # Features
    df = preparar_features(df)

    # Guardar
    cols_finales = [
        "lat", "lon", "estado", "mes",
        "temperatura", "humedad", "viento", "precipitacion",
        "cobertura", "cobertura_num", "en_anp_num", "label"
    ]
    df[cols_finales].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Dataset guardado en: {OUTPUT_CSV}")

    print("\n" + "="*60)
    print("  FASE 1 COMPLETADA — ejecuta fase2_entrenar_modelo.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    run()