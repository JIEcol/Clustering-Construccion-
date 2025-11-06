import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
# Eliminamos 'from kneed import KneeLocator' para solucionar el ModuleNotFoundError.

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuración y Carga de Datos ---

GITHUB_EXCEL_URL = (
    "https://raw.githubusercontent.com/JIEcol/Clustering-Construccion-/main/mapa_3y4_cs_v2_sep25.xlsx"
)

st.title("🏗️ Análisis de Clustering para Proyectos de Construcción")
st.markdown("---")

@st.cache_data
def load_data(url):
    """Carga los datos y realiza limpieza básica."""
    try:
        df = pd.read_excel(url, engine='openpyxl')
        
        # Relleno de NaN y coerción de tipo para todas las variables numéricas esperadas
        # Asegúrate de que el nombre de tus columnas sea exacto
        numeric_cols_to_clean = [
            'longitud', 'latitud', 'area_lote', 'area_construida', 'area_vendible', 
            'numero_etapas', 'numero_unidades', 'numero_bloques', 'total_parqueaderos', 
            '#_parqueaderos_propietarios', '#_parqueaderos_visitantes',
            # Nuevas variables D5 y D6 (asumiendo que ya están consolidadas a nivel P)
            'precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias', 
            'area_por_tipo', 'alcobas', 'baños'
        ]
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                # Convertir a numérico, rellenar NaNs con la media de la columna
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
            
        # Rellenar NaN en categóricas con 'Missing' o 'N/A'
        for col in df.select_dtypes(include=['object', 'category']).columns:
             df[col] = df[col].fillna('N/A')
             
        return df
    except Exception as e:
        st.error(f"Error al cargar/limpiar el archivo: {e}")
        return None

df_raw = load_data(GITHUB_EXCEL_URL)

if df_raw is None:
    st.stop()
else:
    st.success("✅ Archivo cargado y listo para el preprocesamiento con las nuevas dimensiones.")

# --- 1. Definición de Variables (Dimensiones 1-7) ---

# D1-D4 (Variables ya definidas)
location_categorical = ['regional', 'ciudad', 'zona', 'barrio', 'localidad_comuna', 'estrato']
location_numerical = ['longitud', 'latitud']
magnitude_numerical = [
    'numero_etapas', 'numero_unidades', 'area_lote', 'area_construida', 
    'area_vendible', 'numero_bloques', 'total_parqueaderos', 
    '#_parqueaderos_propietarios', '#_parqueaderos_visitantes'
]
quality_categorical = [
    'sistema_constructivo', 'cimentación', 'divison_interior', 
    'placa_entre_piso', 'fachada', 'ventanas'
]
amenities_columns = [col for col in df_raw.columns if col.startswith('zonacomun')]
market_categorical = amenities_columns + ['marca', 'insumos', 'destino', 'uso_general']


# D5: Atributos Transaccionales (Nuevas variables)
# ASUMIMOS que 'fecha' ya fue transformada a una métrica numérica de tiempo, ej: 'antiguedad_dato'
transactional_numerical = ['precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias']
transactional_categorical = ['fase', 'estado', 'modalidad'] # Asumimos OHE de la fase/estado predominante

# D6: Características de Diseño (Nuevas variables)
design_numerical = ['area_por_tipo', 'alcobas', 'baños'] # Promedio Ponderado
design_categorical = ['uso', 'tipo_vivienda', 'nombre_tipo'] 
dotaciones_categorical = [col for col in df_raw.columns if col.startswith('dotaciones_asociadas')] # Asumimos OHE por dotación

# D7: Calidad y Lujo de Acabados (Nuevas variables)
finishes_categorical = [
    'condicion_entrega', 'meson_cocina', 'muebles_cocina', 
    'pisos_alcobas', 'pisos_baño', 'pisos_cocina', 
    'puerta_principal', 'tipo_cocina'
]

# Consolidar listas finales
numerical_features = (
    location_numerical + magnitude_numerical + 
    transactional_numerical + design_numerical
)
categorical_features = (
    location_categorical + quality_categorical + market_categorical + 
    transactional_categorical + design_categorical + dotaciones_categorical + 
    finishes_categorical
)

# Filtrar solo las columnas que realmente existen en el DataFrame
numerical_features = [col for col in numerical_features if col in df_raw.columns]
categorical_features = [col for col in categorical_features if col in df_raw.columns]

st.info(
    f"Total de Dimensiones de entrada: **{len(numerical_features)}** numéricas, "
    f"**{len(categorical_features)}** categóricas."
)

# --- 2. Pipeline de Preprocesamiento ---

# Paso 1: Escalado Numérico
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Paso 2: Transformación Categórica
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combinar los pasos usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 3. Ejecución y Visualización del Clustering ---

# 3.1. Método del Codo (Elbow Method) - Sin kneed
st.subheader("1. Determinación de K (Método del Codo)")
k_range = st.slider("Selecciona un rango máximo para K", 2, 20, 10)

@st.cache_data(show_spinner="Calculando WCSS para el método del codo...")
def run_elbow_method(df_data, preprocessor, max_k):
    """Calcula la Suma de Cuadrados Dentro del Clúster (WCSS) para diferentes K."""
    pipeline_elbow = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline_elbow.fit_transform(df_data)
    
    wcss = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        kmeans.fit(X_processed)
        wcss.append(kmeans.inertia_)
    
    return k_values, wcss

k_values, wcss = run_elbow_method(df_raw, preprocessor, k_range)

# Plotear el resultado (Manual)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, wcss, 'bx-')
ax.set_xlabel("Número de Clústeres (K)")
ax.set_ylabel("WCSS")
ax.set_title("Método del Codo para Encontrar K Óptimo")
st.pyplot(fig)

st.write("Observa el gráfico y elige el punto donde la curva se 'dobla' significativamente. Por defecto sugeriremos 4, pero tú puedes elegir otro valor.")

# 3.2. Ejecución de K-Means
st.subheader("2. Ejecución y Resultados del Clustering")

selected_k = st.number_input(
    "Selecciona el número de Clústeres (K) a usar:", 
    min_value=2, 
    max_value=20, 
    value=4, 
    step=1
)

if st.button(f"Ejecutar K-Means con K={selected_k}"):

    @st.cache_data(show_spinner=f"Entrenando K-Means con {selected_k} clústeres...")
    def run_kmeans_clustering(df_data, preprocessor, k):
        """Ejecuta la pipeline de preprocesamiento y clustering."""
        kmeans_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('cluster', KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300))
        ])
        
        kmeans_pipeline.fit(df_data)
        df_data['Cluster'] = kmeans_pipeline.named_steps['cluster'].labels_
        return df_data
    
    df_clustered = run_kmeans_clustering(df_raw.copy(), preprocessor, selected_k)
    
    st.success(f"Clustering completado. Se crearon {selected_k} clústeres. ¡Trabajo hecho!")

    # A. Distribución de Proyectos por Clúster
    st.markdown("### A. Distribución de Proyectos por Clúster")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("N° Proyectos"))

    # B. Análisis de Centroides (Resumen de Características)
    st.markdown("### B. Análisis de Centroides (Perfil del Clúster)")
    
    st.markdown("#### Promedios de Variables Numéricas")
    # Mostrar promedios de las NUEVAS y viejas numéricas
    all_numeric_cols = list(set(numerical_features) & set(df_clustered.columns))
    numeric_summary = df_clustered.groupby('Cluster')[all_numeric_cols].mean().T
    st.dataframe(numeric_summary.style.background_gradient(cmap='viridis', axis=1))

    # Resumen de variables categóricas clave (incluyendo VIS/No VIS y Acabados)
    st.markdown("#### Categorías Más Frecuentes (Moda) por Clúster")
    key_categorical_summary = [
        'estrato', 'tipo_vivienda', 'condicion_entrega', 
        'uso_general', 'meson_cocina', 'ventas' 
    ]
    categorical_summary_dict = {}
    for col in key_categorical_summary:
        if col in df_clustered.columns:
            # Calcular la moda (valor más frecuente) para cada clúster
            modes = df_clustered.groupby('Cluster')[col].agg(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
            categorical_summary_dict[col] = modes
    
    if categorical_summary_dict:
        st.dataframe(pd.DataFrame(categorical_summary_dict).T)

    # C. Descargar los resultados
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar DataFrame con Etiquetas de Clúster",
        data=csv,
        file_name='proyectos_clusterizados_final.csv',
        mime='text/csv',
    )
