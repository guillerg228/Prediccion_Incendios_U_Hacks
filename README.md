# 🔥 Sistema Inteligente de Predicción de Incendios Forestales

Sistema de monitoreo y predicción de incendios forestales en tiempo real para los estados de Morelos y Estado de México, desarrollado durante el hackathon U-Hacks.

🌐 **Demo en vivo:** [https://erickzghz-crypto.github.io/mapa-incendios-pruebas/](https://erickzghz-crypto.github.io/mapa-incendios-pruebas/)

---

## 🌎 Problema que resuelve

México enfrenta miles de incendios forestales cada año, afectando Áreas Naturales Protegidas, ecosistemas y comunidades. La detección tardía y la falta de predicción preventiva agravan el impacto ambiental y social.

Este sistema combina datos satelitales, meteorológicos y geoespaciales para:
- **Monitorear** incendios activos en tiempo real
- **Predecir** zonas de alto riesgo antes de que ocurran
- **Alertar** cuando un incendio activo se encuentra dentro de un Área Natural Protegida

---

## 🏗️ Arquitectura del sistema

```
NASA FIRMS + Open-Meteo + GeoJSON locales
            ↓
        pipeline.py  (monitoreo diario)
            ↓
      Azure SQL Server
      ├── incendios_activos_morelos
      ├── incendios_activos_edomex
      └── predicciones_riesgo
            ↓
    ┌───────────────────────┐
    │   Modelo ML (fase1-3) │    Power BI Dashboard
    │   Random Forest + SHAP│ ←─ Monitoreo + Predicción
    └───────────────────────┘
            ↓
    Azure Static Web App
```

---

## 🤖 Tecnologías utilizadas

| Categoría | Tecnología |
|---|---|
| Datos satelitales | NASA FIRMS (VIIRS SNPP) |
| Clima | Open-Meteo API |
| Datos geoespaciales | GeoJSON (CONANP, INEGI) |
| Machine Learning | Random Forest + SHAP (scikit-learn) |
| Base de datos | Azure SQL Server |
| Visualización | Power BI |
| Frontend | Azure Static Web Apps |
| Lenguaje | Python 3 |

---

## 📊 Machine Learning

Se utilizó un modelo **Random Forest** con explicabilidad **SHAP**:

- **Entrenamiento:** 2,235 incendios históricos (positivos) + 1,000 puntos sin incendio (negativos)
- **Features:** temperatura, humedad, viento, precipitación, tipo de cobertura vegetal, mes del año, dentro de ANP
- **Output:** probabilidad de incendio (0-100%) + factor principal que lo causa
- **Niveles de riesgo:** BAJO / MEDIO / ALTO / CRÍTICO

SHAP permite explicar cada predicción individualmente:
> *"Esta zona tiene 85% de probabilidad de incendio debido a temperatura de 38°C (+32%), humedad del 12% (+28%) y cobertura de bosque de pino (+18%)"*

---

## 🗂️ Estructura del proyecto

```
proyecto/
├── data/                    # GeoJSON (no incluidos en repo, ver sección de datos)
├── pipeline/
│   ├── pipeline.py          # Monitoreo diario de incendios activos
│   ├── fase1_preparar_datos.py   # Preparación del dataset de entrenamiento
│   ├── fase2_entrenar_modelo.py  # Entrenamiento Random Forest + SHAP
│   └── fase3_predecir.py         # Generación de predicciones de riesgo
├── web/
│   └── index.html           # Página web del proyecto
├── powerbi/                 # Dashboard de Power BI
├── .gitignore
└── README.md
```

---

## 🚀 Cómo correrlo

### 1. Instalar dependencias

```bash
pip install requests geopandas shapely sqlalchemy pyodbc pandas numpy scikit-learn shap joblib matplotlib
```

### 2. Configurar credenciales

En `pipeline/pipeline.py` y `pipeline/fase3_predecir.py` edita:

```python
FIRMS_API_KEY = "tu_api_key_de_nasa_firms"
DB_SERVER     = "tu_servidor.database.windows.net"
DB_NAME       = "tu_base_de_datos"
DB_USER       = "tu_usuario"
DB_PASSWORD   = "tu_password"
```

### 3. Obtener los datos GeoJSON

Descarga los archivos desde [CONANP](https://www.gob.mx/conanp) e [INEGI](https://www.inegi.org.mx) y colócalos en la carpeta `data/`:
- `Areas_Naturales_Morelos.geojson`
- `Areas_Naturales_EdoMex.geojson`
- `Cobertura_Morelos.geojson`
- `Cobertura_EdoMex.geojson`
- `Incendios_Morelos.geojson`
- `Incendios_EdoMex.geojson`

### 4. Ejecutar el pipeline de monitoreo

```bash
python pipeline/pipeline.py
```

### 5. Entrenar el modelo de predicción (una sola vez)

```bash
python pipeline/fase1_preparar_datos.py
python pipeline/fase2_entrenar_modelo.py
```

### 6. Generar predicciones de riesgo

```bash
python pipeline/fase3_predecir.py
```

---

## 👥 Equipo

| Nombre |
|---|
| Guerrero Aguilar Guillermo Antonio |
| Castelan Meliton Uriel Agustin |
| Mendoza Rios Nadia Miranda |
| Hernandez Gonzalez Erick Julian |

---

## 📄 Licencia

MIT License — libre para usar, modificar y distribuir.
