import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # Agregamos Plotly para visualizaciones interactivas
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuración y Carga de Datos ---

GITHUB_EXCEL_URL = (
    "https://raw.githubusercontent.com/JIEcol/Clustering-Construccion-/main/mapa_3y4_cs_v2_sep25.xlsx"
)

st.set_page_config(layout="wide")
st.title("🏗️ Análisis de Clustering (K-Means) de Proyectos de Construcción")
st.markdown("---")
st.markdown("#### Paso 1: Carga y Preparación de las 7 Dimensiones")


@st.cache_data
def load_data(url):
    """Carga los datos y realiza limpieza básica."""
    try:
        df = pd.read_excel(url, engine='openpyxl')
        
        # Lista de todas las columnas numéricas esperadas
        numeric_cols_to_clean = [
            'longitud', 'latitud', 'numero_etapas', 'numero_unidades', 'area_lote', 
            'area_construida', 'area_vendible', 'numero_bloques', 'total_parqueaderos', 
            '#_parqueaderos_propietarios', '#_parqueaderos_visitantes',
            'precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias', 
            'area_por_tipo', 'alcobas', 'baños'
        ]
        
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Relleno de NaNs con la media (Estrategia simple para el modelo)
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
    st.success(f"✅ Archivo cargado. {df_raw.shape[0]} filas listas para K-Means.")


# --- 1. Definición de Variables (Dimensiones 1-7) ---

# D1-D4
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

# Filtrar solo las columnas que realmente existen
numerical_features = [col for col in numerical_features if col in df_raw.columns]
categorical_features = [col for col in categorical_features if col in df_raw.columns]


# --- 2. Pipeline de Preprocesamiento (Creación fuera de funciones cacheables) ---

def build_preprocessor(numerical_features, categorical_features):
    """Construye el ColumnTransformer basado en las listas de columnas."""
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

preprocessor = build_preprocessor(numerical_features, categorical_features)


# SOLUCIÓN a UnhashableParamError: Cacheamos la data procesada
@st.cache_data(show_spinner="Aplicando preprocesamiento (Estandarización y OHE)...")
def get_processed_data(df_data, preprocessor):
    """Aplica el preprocesamiento a los datos y devuelve una matriz numpy."""
    # Usamos fit_transform para entrenar el escalador/OHE y transformar
    return preprocessor.fit_transform(df_data)

X_processed = get_processed_data(df_raw, preprocessor) 


# --- 3. Ejecución y Visualización del Clustering ---

st.markdown("#### Paso 2: Evaluación y Entrenamiento del Modelo")

# 3.1. Método del Codo
st.subheader("1. Determinación de K (Método del Codo)")
k_range = st.slider("Selecciona el rango máximo de K a evaluar:", 2, 20, 10)

def run_elbow_method(X_data, max_k):
    """Calcula la Suma de Cuadrados Dentro del Clúster (WCSS) para diferentes K."""
    wcss = []
    k_values = range(1, max_k + 1)
    
    # Esta función ya no es cacheada, por eso es rápida (usa X_processed cacheado)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        kmeans.fit(X_data)
        wcss.append(kmeans.inertia_)
    
    return k_values, wcss

k_values, wcss = run_elbow_method(X_processed, k_range)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, wcss, 'bx-')
    ax.set_xlabel("Número de Clústeres (K)")
    ax.set_ylabel("WCSS (Varianza Intra-Clúster)")
    ax.set_title("Método del Codo: Elige el punto de inflexión")
    st.pyplot(fig)

with col2:
    st.markdown("#### Elige K")
    st.info("El 'codo' es donde la mejora al añadir un clúster se estanca. ¡Sé eficiente!")
    selected_k = st.number_input(
        "Número de Clústeres (K):", 
        min_value=2, 
        max_value=20, 
        value=4, 
        step=1
    )

# 3.2. Ejecución de K-Means y Análisis

if st.button(f"🚀 Ejecutar K-Means y Segmentar con K={selected_k}", type="primary"):

    @st.cache_data(show_spinner=f"Entrenando K-Means con {selected_k} clústeres...")
    def run_kmeans_clustering(X_data, df_original, k):
        """Ejecuta K-Means en los datos ya procesados y asigna etiquetas."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        kmeans.fit(X_data) 
        
        # Copiamos el DF para añadir la nueva columna sin afectar el DF cacheado
        df_clustered = df_original.copy()
        df_clustered['Cluster'] = kmeans.labels_
        return df_clustered
    
    df_clustered = run_kmeans_clustering(X_processed, df_raw, selected_k)
    
    st.success(f"Clustering completado. Se crearon **{selected_k} segmentos** de mercado.")

    # A. Distribución de Proyectos por Clúster
    st.subheader("2. Perfiles y Distribución de Clústeres")
    
    col_dist, col_map = st.columns(2)
    
    with col_dist:
        st.markdown("##### A. Distribución y Perfiles Numéricos")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("N° Proyectos"))

        # Análisis de Centroides Numéricos
        all_numeric_cols = list(set(numerical_features) & set(df_clustered.columns))
        numeric_summary = df_clustered.groupby('Cluster')[all_numeric_cols].mean().T
        st.dataframe(
            numeric_summary.style.background_gradient(cmap='viridis', axis=1), 
            caption="Valores promedio clave por clúster (e.g., Precioenmiles promedio)."
        )

    with col_map:
        st.markdown("##### B. Visualización Geográfica (Latitud vs. Longitud)")
        if 'latitud' in df_clustered.columns and 'longitud' in df_clustered.columns:
            fig_map = px.scatter(
                df_clustered, 
                x='longitud', 
                y='latitud', 
                color='Cluster', 
                hover_data=['regional', 'estrato', 'precioenmiles'],
                title=f"Segmentación de Proyectos (K={selected_k})",
                template="streamlit"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Faltan las columnas 'latitud' o 'longitud' para el mapa.")
            
    # Resumen Categórico
    st.markdown("#### C. Perfiles Categóricos Clave (Moda)")
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
            caption="Valores más comunes (Moda) para entender el 'lujo' y el 'nicho' de cada clúster."
        )

    # D. Descargar los resultados
    st.markdown("---")
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar DataFrame con Etiquetas de Clúster",
        data=csv,
        file_name='proyectos_clusterizados_final.csv',
        mime='text/csv',
    )
