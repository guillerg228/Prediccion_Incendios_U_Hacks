"""
Pipeline de datos - Predicción de Incendios
Hackathon

Fuentes:
- NASA FIRMS: incendios activos
- Open-Meteo: condiciones meteorológicas
- GeoJSON locales: ANP y cobertura vegetal

Requiere:
    pip install requests geopandas shapely sqlalchemy pyodbc pandas
"""

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import json
import os

# =============================================================================
# CONFIGURACIÓN — edita estos valores
# =============================================================================

# NASA FIRMS
FIRMS_API_KEY = "e6d51347ffd4693cc03888bc3816e4f6"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Área de interés — bounding box
# formato: W,S,E,N
BBOX_MORELOS  = "-99.6,18.4,-98.6,19.1"
BBOX_EDOMEX   = "-100.6,18.7,-98.4,20.3"

# Open-Meteo (sin API key, es gratuita)
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# SQL Server — autenticación SQL clásica
DB_SERVER   = "servidor-incedios.database.windows.net"  # cambia por tu servidor
DB_NAME     = "incedios100real"                              # cambia por tu base de datos
DB_USER     = "urigod"                            # cambia por tu usuario
DB_PASSWORD = "Castelan1-"                            # cambia por tu contraseña

CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
)

# Rutas a tus GeoJSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON = {
    "anp_morelos":       os.path.join(BASE_DIR, "..", "data", "Areas_Naturales_Morelos.geojson"),
    "anp_edomex":        os.path.join(BASE_DIR, "..", "data", "Areas_Naturales_EdoMex.geojson"),
    "cobertura_morelos": os.path.join(BASE_DIR, "..", "data", "Cobertura_Morelos.geojson"),
    "cobertura_edomex":  os.path.join(BASE_DIR, "..", "data", "Cobertura_EdoMex.geojson"),
    "incendios_morelos": os.path.join(BASE_DIR, "..", "data", "Incendios_Morelos.geojson"),
    "incendios_edomex":  os.path.join(BASE_DIR, "..", "data", "Incendios_EdoMex.geojson"),
}

# =============================================================================
# 1. CARGAR GEOJSON
# =============================================================================

def cargar_geojson():
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
# 2. NASA FIRMS — incendios activos
# =============================================================================

def obtener_firms(bbox, dias=7):
    """
    Descarga puntos de incendios activos de NASA FIRMS.
    dias: cuántos días hacia atrás consultar (1-10)
    """
    print(f"🔥 Consultando NASA FIRMS (bbox: {bbox})...")
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        f"/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/{bbox}/{dias}"
    )
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        print(f"   ❌ Error FIRMS: {resp.status_code} - {resp.text[:200]}")
        return gpd.GeoDataFrame()

    if "latitude" not in resp.text:
        print(f"   ⚠️  Respuesta inesperada: {resp.text[:200]}")
        return gpd.GeoDataFrame()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    if df.empty:
        print("   ℹ️  Sin incendios activos en el área.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    print(f"   ✅ {len(gdf)} puntos de incendio encontrados")
    return gdf

# =============================================================================
# 3. OPEN-METEO — clima para cada punto
# =============================================================================

def obtener_clima(lat, lon):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone":  "America/Mexico_City",
    }
    resp = requests.get(OPENMETEO_URL, params=params, timeout=10)
    if resp.status_code != 200:
        return {"temperatura": None, "humedad": None, "viento": None, "precipitacion": None}

    data = resp.json().get("current", {})
    return {
        "temperatura":   data.get("temperature_2m"),
        "humedad":       data.get("relative_humidity_2m"),
        "viento":        data.get("wind_speed_10m"),
        "precipitacion": data.get("precipitation"),
    }

def enriquecer_con_clima(gdf_incendios):
    print("🌤️  Obteniendo datos climáticos...")

    if gdf_incendios.empty:
        return gdf_incendios

    gdf_incendios["lat_r"] = gdf_incendios.geometry.y.round(1)
    gdf_incendios["lon_r"] = gdf_incendios.geometry.x.round(1)
    puntos_unicos = gdf_incendios[["lat_r", "lon_r"]].drop_duplicates()

    clima_cache = {}
    for _, row in puntos_unicos.iterrows():
        key = (row["lat_r"], row["lon_r"])
        clima_cache[key] = obtener_clima(row["lat_r"], row["lon_r"])

    for col in ["temperatura", "humedad", "viento", "precipitacion"]:
        gdf_incendios[col] = gdf_incendios.apply(
            lambda r: clima_cache[(r["lat_r"], r["lon_r"])][col], axis=1
        )

    print(f"   ✅ Clima agregado a {len(gdf_incendios)} puntos")
    return gdf_incendios

# =============================================================================
# 4. CRUCE ESPACIAL — ANP y cobertura vegetal
# =============================================================================

def cruzar_con_capas(gdf_incendios, capas, estado):
    print(f"🗺️  Cruzando datos espaciales ({estado})...")

    if gdf_incendios.empty:
        return gdf_incendios

    anp_key       = f"anp_{estado}"
    cobertura_key = f"cobertura_{estado}"

    if capas.get(anp_key) is not None:
        joined = gpd.sjoin(
            gdf_incendios,
            capas[anp_key][["geometry", "NOMBRE", "CAT_MANEJO"]],
            how="left",
            predicate="within"
        )
        gdf_incendios["en_anp"]        = ~joined["NOMBRE"].isna()
        gdf_incendios["nombre_anp"]    = joined["NOMBRE"].values
        gdf_incendios["categoria_anp"] = joined["CAT_MANEJO"].values
    else:
        gdf_incendios["en_anp"]        = False
        gdf_incendios["nombre_anp"]    = None
        gdf_incendios["categoria_anp"] = None

    if capas.get(cobertura_key) is not None:
        col_desc = "DESC_SAMOF" if "DESC_SAMOF" in capas[cobertura_key].columns else \
                   capas[cobertura_key].columns[1]
        joined2 = gpd.sjoin(
            gdf_incendios,
            capas[cobertura_key][["geometry", col_desc]],
            how="left",
            predicate="within"
        )
        gdf_incendios["cobertura"] = joined2[col_desc].values
    else:
        gdf_incendios["cobertura"] = None

    print(f"   ✅ Cruce completado")
    return gdf_incendios

# =============================================================================
# 5. ÍNDICE DE RIESGO
# =============================================================================

RIESGO_COBERTURA = {
    "bosque":   1.5,
    "selva":    1.4,
    "matorral": 1.3,
    "pastizal": 1.2,
    "agricola": 0.8,
    "urbano":   0.5,
    "agua":     0.0,
}

def calcular_riesgo(row):
    score = 0

    temp = row.get("temperatura")
    if temp is not None:
        if temp >= 40:   score += 30
        elif temp >= 35: score += 20
        elif temp >= 30: score += 10

    hum = row.get("humedad")
    if hum is not None:
        if hum <= 10:    score += 30
        elif hum <= 20:  score += 20
        elif hum <= 30:  score += 10

    viento = row.get("viento")
    if viento is not None:
        if viento >= 50:   score += 20
        elif viento >= 30: score += 12
        elif viento >= 15: score += 6

    precip = row.get("precipitacion")
    if precip is not None and precip == 0:
        score += 10

    cobertura = str(row.get("cobertura", "")).lower()
    multiplicador = 1.0
    for key, val in RIESGO_COBERTURA.items():
        if key in cobertura:
            multiplicador = val
            break

    score = min(100, score * multiplicador)

    if score >= 70:   nivel = "CRÍTICO"
    elif score >= 50: nivel = "ALTO"
    elif score >= 30: nivel = "MEDIO"
    else:             nivel = "BAJO"

    return round(score, 1), nivel

def agregar_riesgo(gdf):
    if gdf.empty:
        return gdf
    print("⚠️  Calculando índice de riesgo...")
    resultados = gdf.apply(calcular_riesgo, axis=1)
    gdf["indice_riesgo"] = [r[0] for r in resultados]
    gdf["nivel_riesgo"]  = [r[1] for r in resultados]
    criticos = (gdf["nivel_riesgo"] == "CRÍTICO").sum()
    print(f"   ✅ {criticos} puntos en nivel CRÍTICO")
    return gdf

# =============================================================================
# 6. GUARDAR EN SQL SERVER
# =============================================================================

def guardar_en_bd(gdf, tabla, conn_str):
    if gdf.empty:
        print(f"   ℹ️  Sin datos para guardar en {tabla}")
        return

    print(f"💾 Guardando en SQL Server → tabla: {tabla}...")

    import pyodbc

    df = gdf.drop(columns="geometry", errors="ignore")
    df["fecha_ingesta"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["lat"] = gdf.geometry.y
    df["lon"] = gdf.geometry.x
    df.columns = [c.lower().replace(" ", "_")[:50] for c in df.columns]

    cols_def = ", ".join([f"[{c}] NVARCHAR(255)" for c in df.columns])
    create_sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tabla}') CREATE TABLE {tabla} ({cols_def})"

    conn = pyodbc.connect(conn_str, timeout=30)
    cursor = conn.cursor()
    cursor.execute(create_sql)
    conn.commit()

    placeholders = ", ".join(["?" for _ in df.columns])
    cols_str = ", ".join([f"[{c}]" for c in df.columns])
    insert_sql = f"INSERT INTO {tabla} ({cols_str}) VALUES ({placeholders})"

    for _, row in df.iterrows():
        cursor.execute(insert_sql, [str(v) if v is not None else None for v in row])

    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ {len(df)} registros guardados")

# =============================================================================
# MAIN
# =============================================================================

def run_pipeline():
    print("\n" + "="*60)
    print("  PIPELINE DE INCENDIOS — INICIO")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    capas = cargar_geojson()

    print("🔌 Conectando a SQL Server...")
    engine = None
    try:
        import pyodbc
        test_conn = pyodbc.connect(CONNECTION_STRING, timeout=30)
        test_conn.close()
        engine = CONNECTION_STRING
        print("   ✅ Conexión exitosa\n")
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        print("   Continuando sin guardar en BD (modo debug)\n")
        engine = None

    # Morelos en lugar de Jalisco
    for estado, bbox in [("morelos", BBOX_MORELOS), ("edomex", BBOX_EDOMEX)]:
        print(f"\n{'─'*40}")
        print(f"  PROCESANDO: {estado.upper()}")
        print(f"{'─'*40}")

        gdf = obtener_firms(bbox, dias=5)
        if gdf.empty:
            print(f"  Sin incendios activos en {estado} (últimos 7 días).\n")
            continue

        gdf = enriquecer_con_clima(gdf)
        gdf = cruzar_con_capas(gdf, capas, estado)
        gdf = agregar_riesgo(gdf)

        if engine:
            guardar_en_bd(gdf, f"incendios_activos_{estado}", engine)

        print(f"\n  📊 Resumen {estado}:")
        if "nivel_riesgo" in gdf.columns:
            print(gdf["nivel_riesgo"].value_counts().to_string())

    print("\n" + "="*60)
    print("  PIPELINE COMPLETADO")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_pipeline()