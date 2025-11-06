import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ... (El código de importaciones, warnings, y la carga inicial de datos 'df_raw' se mantiene igual) ...

# --- Código para construir el preprocesador (Movido dentro de la función cacheada) ---

# ... (Las listas numerical_features y categorical_features se mantienen igual) ...

# ----------------------------------------------------------------------------------
# ¡SOLUCIÓN FINAL!: La función cacheada recibe solo strings/hashes y reconstruye el preprocesador.
# ----------------------------------------------------------------------------------

def build_preprocessor(numerical_features, categorical_features):
    """Construye y devuelve el ColumnTransformer."""
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

@st.cache_data(show_spinner="Aplicando preprocesamiento (Estandarización y OHE)...")
def get_processed_data(df_data, num_cols, cat_cols):
    """
    Aplica el preprocesamiento a los datos.
    Recibe listas de columnas (hashable) en lugar del objeto ColumnTransformer.
    """
    # 1. Reconstruir el preprocesador dentro de la función cacheada
    preprocessor_internal = build_preprocessor(num_cols, cat_cols)
    
    # 2. Aplicar la transformación
    X_transformed = preprocessor_internal.fit_transform(df_data)
    
    # 3. Guardar el preprocesador entrenado como atributo para su uso posterior (e.g., para desescalar)
    # Aunque no lo usaremos inmediatamente, es una buena práctica.
    # Nota: No devolvemos el preprocesador, solo la data transformada.
    return X_transformed, preprocessor_internal 

# Procesar los datos una sola vez. Ahora pasamos listas de strings (hashable)
X_processed, preprocessor_fitted = get_processed_data(df_raw, numerical_features, categorical_features) 


# --- El resto del código continúa desde la sección 3.1 ---

# 3.1. Método del Codo
st.subheader("1. Determinación de K (Método del Codo)")
k_range = st.slider("Selecciona el rango máximo de K a evaluar:", 2, 20, 10)

def run_elbow_method(X_data, max_k):
    """Calcula la Suma de Cuadrados Dentro del Clúster (WCSS) para diferentes K."""
    wcss = []
    k_values = range(1, max_k + 1)
    
    with st.spinner(f"Calculando inercia para K de 1 a {max_k}. Esto puede tardar unos segundos..."):
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
            kmeans.fit(X_data)
            wcss.append(kmeans.inertia_)
    
    return k_values, wcss

k_values, wcss = run_elbow_method(X_processed, k_range)

# ... (El resto del código de visualización del codo, el selector de K y el entrenamiento de K-Means es el mismo) ...

# 3.2. Ejecución de K-Means y Análisis

selected_k = st.number_input(
    "Selecciona el número de Clústeres (K) a usar:", 
    min_value=2, 
    max_value=20, 
    value=4, 
    step=1
)

if st.button(f"🚀 Ejecutar K-Means y Segmentar con K={selected_k}", type="primary"):

    @st.cache_data(show_spinner=f"Entrenando K-Means con {selected_k} clústeres...")
    def run_kmeans_clustering(X_data, df_original, k):
        """Ejecuta K-Means en los datos ya procesados y asigna etiquetas."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        kmeans.fit(X_data) 
        
        df_clustered = df_original.copy()
        df_clustered['Cluster'] = kmeans.labels_
        return df_clustered
    
    df_clustered = run_kmeans_clustering(X_processed, df_raw, selected_k)
    
    # ... (El resto de la sección de visualización y descarga se mantiene igual) ...
    
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
                color=df_clustered['Cluster'].astype(str), # Convertir a string para color discreto
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
