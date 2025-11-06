import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from kneed import KneeLocator # Usaremos kneed para identificar el codo automáticamente (opcional, si falla lo quitamos)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Configuración y Carga de Datos (Reutilizando tu código) ---

GITHUB_EXCEL_URL = (
    "https://raw.githubusercontent.com/JIEcol/Clustering-Construccion-/main/mapa_3y4_cs_v2_sep25.xlsx"
)

st.title("🏗️ Análisis de Clustering para Proyectos de Construcción")
st.markdown("---")

@st.cache_data
def load_data(url):
    """Carga los datos desde la URL de GitHub."""
    try:
        df = pd.read_excel(url, engine='openpyxl')
        # Limpieza básica: rellenar NaN en columnas numéricas y convertir tipos
        for col in ['longitud', 'latitud', 'area_lote', 'area_construida', 'area_vendible']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

df_raw = load_data(GITHUB_EXCEL_URL)

if df_raw is None:
    st.stop()
else:
    st.success("✅ Archivo cargado y listo para el preprocesamiento.")

# --- 1. Definición de Variables de la Estrategia ---

# D1: Ubicación y Contexto Socioeconómico
location_categorical = ['regional', 'ciudad', 'zona', 'barrio', 'localidad_comuna', 'estrato']
location_numerical = ['longitud', 'latitud']

# D2: Magnitud, Escala y Tipología
magnitude_numerical = [
    'numero_etapas', 'numero_unidades', 'area_lote', 'area_construida', 
    'area_vendible', 'numero_bloques', 'total_parqueaderos', 
    '#_parqueaderos_propietarios', '#_parqueaderos_visitantes'
]

# D3: Calidad Estructural y Costo Base
quality_categorical = [
    'sistema_constructivo', 'cimentación', 'divison_interior', 
    'placa_entre_piso', 'fachada', 'ventanas'
]

# D4: Segmento de Mercado y Lujo (Amenidades y Marcas)
# Generamos la lista de zonacomunXX automáticamente
amenities_columns = [col for col in df_raw.columns if col.startswith('zonacomun')]
market_categorical = amenities_columns + ['marca', 'insumos', 'destino', 'uso_general']


# Consolidar listas
numerical_features = location_numerical + magnitude_numerical
categorical_features = location_categorical + quality_categorical + market_categorical

# Filtrar solo las columnas que realmente existen en el DataFrame
numerical_features = [col for col in numerical_features if col in df_raw.columns]
categorical_features = [col for col in categorical_features if col in df_raw.columns]

st.info(f"Dimensiones de entrada: {len(numerical_features)} numéricas, {len(categorical_features)} categóricas.")


# --- 2. Pipeline de Preprocesamiento (Siguiendo la Estrategia) ---

# Paso 1: Escalado Numérico (Normalizar/Estandarizar)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Estandarización
])

# Paso 2: Transformación Categórica (One-Hot Encoding - OHE)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # OHE
])

# Combinar los pasos usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Ignorar cualquier columna no especificada
)

# --- 3. Ejecución y Visualización del Clustering ---

# 3.1. Método del Codo para elegir K

st.subheader("1. Determinación de K (Método del Codo)")
k_range = st.slider("Selecciona un rango máximo para K", 2, 20, 10)

@st.cache_data(show_spinner="Calculando WCSS para el método del codo...")
def run_elbow_method(df_data, preprocessor, max_k):
    """Calcula la Suma de Cuadrados Dentro del Clúster (WCSS) para diferentes K."""
    pipeline_elbow = Pipeline(steps=[('preprocessor', preprocessor)])
    # Transformar los datos una sola vez
    X_processed = pipeline_elbow.fit_transform(df_data)
    
    wcss = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        if k == 1:
            # WCSS para K=1 es la suma de cuadrados totales
            wcss.append(np.sum(np.sum(X_processed - X_processed.mean(axis=0))**2))
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
            kmeans.fit(X_processed)
            wcss.append(kmeans.inertia_)

    return k_values, wcss

k_values, wcss = run_elbow_method(df_raw, preprocessor, k_range)

# Encontrar el "codo" (punto de inflexión)
try:
    kneedle = KneeLocator(k_values, wcss, S=1.0, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    if optimal_k is None: optimal_k = 4 # Fallback
except:
    optimal_k = 4 # Fallback si kneed falla

# Plotear el resultado
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, wcss, 'bx-')
ax.set_xlabel("Número de Clústeres (K)")
ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
ax.set_title("Método del Codo para Encontrar K Óptimo")
if optimal_k and optimal_k > 1:
    ax.vlines(optimal_k, min(wcss), max(wcss), linestyles='--', colors='r', label=f'K sugerido: {optimal_k}')
    ax.legend()
st.pyplot(fig)

st.write(f"Según el método del codo, un buen valor inicial para K es: **{optimal_k if optimal_k and optimal_k > 1 else 4}**")


# 3.2. Ejecución de K-Means y Análisis de Resultados

st.subheader("2. Ejecución y Resultados del Clustering")

# Selector de K con el valor sugerido
selected_k = st.number_input(
    "Selecciona el número de Clústeres (K) a usar:", 
    min_value=2, 
    max_value=20, 
    value=optimal_k if optimal_k and optimal_k > 1 else 4, 
    step=1
)

if st.button(f"Ejecutar K-Means con K={selected_k}"):

    @st.cache_data(show_spinner=f"Entrenando K-Means con {selected_k} clústeres...")
    def run_kmeans_clustering(df_data, preprocessor, k):
        """Ejecuta la pipeline de preprocesamiento y clustering."""
        # Pipeline completa: Preprocesamiento + K-Means
        kmeans_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('cluster', KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300))
        ])
        
        # Entrenar el modelo
        kmeans_pipeline.fit(df_data)
        
        # Obtener las etiquetas de clúster y el DataFrame transformado
        df_data['Cluster'] = kmeans_pipeline.named_steps['cluster'].labels_
        
        # Extraer los datos escalados para calcular los centroides en el espacio de características
        X_scaled = kmeans_pipeline.named_steps['preprocessor'].transform(df_data)
        
        # Calcular los centroides reales
        centroids_scaled = kmeans_pipeline.named_steps['cluster'].cluster_centers_
        
        return df_data, centroids_scaled, X_scaled
    
    df_clustered, centroids, X_scaled = run_kmeans_clustering(df_raw.copy(), preprocessor, selected_k)
    
    st.success(f"Clustering completado. Se crearon {selected_k} clústeres.")

    # A. Distribución de Proyectos por Clúster
    st.markdown("### A. Distribución de Proyectos por Clúster")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("N° Proyectos"))
    
    # B. Análisis de Centroides (Resumen de Características)
    st.markdown("### B. Análisis de Centroides (El 'Perfil' de cada Clúster)")
    
    # Crea un DataFrame de los centroides para variables numéricas (desescaladas)
    # y las categorías más frecuentes para las variables categóricas.
    
    # Paso 1: Invertir la transformación de las características numéricas
    num_cols_processed = [f'num__{col}' for col in numerical_features]
    
    st.markdown("""
        Los centroides muestran los valores promedio de las variables numéricas y 
        las categorías más comunes para las variables categóricas en cada clúster.
    """)

    # Para un análisis más simple en Streamlit, mostramos el promedio por clúster del DF original:
    numeric_summary = df_clustered.groupby('Cluster')[numerical_features].mean().T
    st.dataframe(numeric_summary.style.background_gradient(cmap='viridis', axis=1), 
                 caption="Valores Promedio de Variables Numéricas por Clúster (Sin desescalar)")

    # Resumen de variables categóricas (Moda)
    st.markdown("#### Categorías Más Frecuentes (Moda) por Clúster")
    
    # Seleccionamos algunas variables categóricas clave para el resumen
    key_categorical = ['regional', 'estrato', 'sistema_constructivo', 'fachada', 'uso_general', 'marca']
    categorical_summary = {}
    for col in key_categorical:
        if col in df_clustered.columns:
            # Calcular la moda (valor más frecuente) para cada clúster
            modes = df_clustered.groupby('Cluster')[col].agg(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
            categorical_summary[col] = modes
    
    if categorical_summary:
        st.dataframe(pd.DataFrame(categorical_summary).T, 
                     caption="Moda (Valor más Común) de Variables Categóricas Clave por Clúster")

    # C. Descargar los resultados
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar DataFrame con Etiquetas de Clúster",
        data=csv,
        file_name='proyectos_clusterizados.csv',
        mime='text/csv',
    )
