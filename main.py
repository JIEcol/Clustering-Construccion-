import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
# No usamos kneed, evitamos el ModuleNotFoundError.

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuración y Carga de Datos ---

GITHUB_EXCEL_URL = (
    "https://raw.githubusercontent.com/JIEcol/Clustering-Construccion-/main/mapa_3y4_cs_v2_sep25.xlsx"
)

st.set_page_config(layout="wide") # Para que se vea mejor
st.title("🏗️ Análisis de Clustering (K-Means) de Proyectos de Construcción")
st.markdown("---")
st.markdown("**¡Cargando y Preparando las 7 Dimensiones de Datos!**")


@st.cache_data
def load_data(url):
    """Carga los datos y realiza limpieza básica."""
    try:
        df = pd.read_excel(url, engine='openpyxl')
        
        # Lista de todas las columnas numéricas esperadas para limpiar NaNs con la media
        numeric_cols_to_clean = [
            'longitud', 'latitud', 'numero_etapas', 'numero_unidades', 'area_lote', 
            'area_construida', 'area_vendible', 'numero_bloques', 'total_parqueaderos', 
            '#_parqueaderos_propietarios', '#_parqueaderos_visitantes',
            # D5 y D6 Numéricas
            'precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias', 
            'area_por_tipo', 'alcobas', 'baños'
        ]
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Reemplazamos NaNs con la media para evitar problemas en el escalado
                df[col] = df[col].fillna(df[col].mean()) 
            
        # Rellenar NaN en categóricas con 'N/A' para el OHE
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
    st.success(f"✅ Archivo cargado. {df_raw.shape[0]} filas y {df_raw.shape[1]} columnas disponibles.")


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

# D5: Atributos Transaccionales
# Omitimos 'fecha' asumiendo que ya fue transformada o no está presente.
transactional_numerical = ['precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias']
transactional_categorical = ['fase', 'estado', 'modalidad'] 

# D6: Características de Diseño
design_numerical = ['area_por_tipo', 'alcobas', 'baños']
design_categorical = ['uso', 'tipo_vivienda', 'nombre_tipo'] 
dotaciones_categorical = [col for col in df_raw.columns if col.startswith('dotaciones_asociadas')] 

# D7: Calidad y Lujo de Acabados
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
    f"Se usarán **{len(numerical_features)}** variables numéricas (Estandarización) y "
    f"**{len(categorical_features)}** variables categóricas (One-Hot Encoding)."
)

# --- 2. Pipeline de Preprocesamiento ---

# Escalado Numérico (Estandarización)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Transformación Categórica (One-Hot Encoding)
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

# 3.1. Método del Codo
st.subheader("1. Determinación de K (Método del Codo)")
k_range = st.slider("Selecciona el rango máximo de K a evaluar:", 2, 20, 10)

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

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, wcss, 'bx-')
    ax.set_xlabel("Número de Clústeres (K)")
    ax.set_ylabel("WCSS (Varianza Intra-Clúster)")
    ax.set_title("Método del Codo")
    st.pyplot(fig)

with col2:
    st.markdown("#### Interpretación del Codo:")
    st.info("El 'codo' es el punto donde el descenso del WCSS comienza a ser marginal. Elige un K justo antes de que la curva se aplane.")
    selected_k = st.number_input(
        "Selecciona el número de Clústeres (K) a usar:", 
        min_value=2, 
        max_value=20, 
        value=4, 
        step=1
    )

# 3.2. Ejecución de K-Means y Análisis

if st.button(f"🚀 Ejecutar K-Means y Segmentar con K={selected_k}", type="primary"):

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
    
    st.success(f"Clustering completado. Se crearon {selected_k} segmentos de mercado.")

    # A. Distribución de Proyectos por Clúster
    st.subheader("1. Distribución de Proyectos")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("N° Proyectos"))

    # B. Análisis de Centroides
    st.subheader("2. Perfil de los Clústeres (Centroides)")
    
    st.markdown("#### Promedios de Variables Numéricas")
    all_numeric_cols = list(set(numerical_features) & set(df_clustered.columns))
    numeric_summary = df_clustered.groupby('Cluster')[all_numeric_cols].mean().T
    st.dataframe(
        numeric_summary.style.background_gradient(cmap='viridis', axis=1), 
        caption="Valores promedio (sin desescalar) de las métricas de cada clúster."
    )

    st.markdown("#### Categorías Más Frecuentes (Moda)")
    key_categorical_summary = [
        'estrato', 'tipo_vivienda', 'condicion_entrega', 
        'uso_general', 'meson_cocina', 'regional'
    ]
    categorical_summary_dict = {}
    for col in key_categorical_summary:
        if col in df_clustered.columns:
            modes = df_clustered.groupby('Cluster')[col].agg(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
            categorical_summary_dict[col] = modes
    
    if categorical_summary_dict:
        st.dataframe(
            pd.DataFrame(categorical_summary_dict).T,
            caption="Valores más comunes (Moda) de las categorías clave."
        )

    # C. Descargar los resultados
    st.markdown("---")
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar DataFrame con Etiquetas de Clúster",
        data=csv,
        file_name='proyectos_clusterizados_final.csv',
        mime='text/csv',
    )
