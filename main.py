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

# --- Configuración y Carga de Datos ---

st.set_page_config(layout="wide")
st.title("🏗️ Análisis de Clustering (K-Means) de Proyectos de Construcción")
st.markdown("---")

# URL del Diccionario de Datos (Excel) como referencia en GitHub
GITHUB_EXCEL_DICT_URL = (
    "https://github.com/JIEcol/Clustering-Construccion-/blob/main/Diccionario%20de%20datos%20CU%20sin%20estrategia.xlsx"
)
st.sidebar.info(f"📚 El **Diccionario de Datos** está disponible en este [enlace de GitHub]({GITHUB_EXCEL_DICT_URL}).")


# 1. Función para la carga del archivo (Sin caché para evitar conflictos con el FileUploader)
def load_data(uploaded_file):
    """Carga los datos desde el archivo Parquet subido y realiza limpieza."""
    try:
        # 🛑 ESTRATEGIA FINAL: Pasamos el objeto de Streamlit directamente a pd.read_parquet
        df = pd.read_parquet(uploaded_file) 
        
        # Lista de todas las columnas numéricas esperadas para limpieza
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
                df[col] = df[col].fillna(df[col].mean()) 
            
        # Rellenar NaN en categóricas con 'N/A'
        for col in df.select_dtypes(include=['object', 'category']).columns:
             df[col] = df[col].fillna('N/A')
             
        return df
    except Exception as e:
        # Mostramos el error si la lectura falla
        st.error(f"❌ Falló la lectura del archivo Parquet. Detalle: {e}")
        return None

# 2. El Uploader de archivos
st.markdown("#### Paso 1: Carga de Datos y Preprocesamiento")
uploaded_file = st.file_uploader(
    "📤 Sube aquí el archivo de datos consolidado (`.parquet`)",
    type="parquet"
)

# 3. Lógica principal condicionada a la carga del archivo
if uploaded_file is not None:
    # Usamos st.spinner para mostrar el estado de la carga
    with st.spinner("Cargando y limpiando datos..."):
        df_raw = load_data(uploaded_file)

    if df_raw is None or df_raw.empty:
        # st.error ya se llamó dentro de load_data
        st.stop()
    else:
        st.success(f"✅ Archivo Parquet cargado. {df_raw.shape[0]} filas listas para K-Means.")

        # ----------------------------------------------------------------------------------
        # --- 2. DEFINICIÓN DE VARIABLES ---
        # ----------------------------------------------------------------------------------

        # Variables Categóricas y Numéricas consolidadas de las 7 dimensiones
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
        transactional_numerical = ['precioenmiles', 'preciomc', 'saldo', 'ventas', 'renuncias']
        transactional_categorical = ['fase', 'estado', 'modalidad'] 
        design_numerical = ['area_por_tipo', 'alcobas', 'baños']
        design_categorical = ['uso', 'tipo_vivienda', 'nombre_tipo'] 
        dotaciones_categorical = [col for col in df_raw.columns if col.startswith('dotaciones_asociadas')] 
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

        st.info(
            f"Se usarán **{len(numerical_features)}** variables numéricas (Estandarización) y **{len(categorical_features)}** variables categóricas (OHE)."
        )

        # ----------------------------------------------------------------------------------
        # --- 3. PIPELINE DE PREPROCESAMIENTO Y SOLUCIÓN DE CACHING ---
        # ----------------------------------------------------------------------------------
        
        # Mantenemos el caché en el preprocesamiento de la matriz (X_processed) para eficiencia
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
            """Aplica el preprocesamiento, cacheando la matriz X_processed."""
            preprocessor_internal = build_preprocessor(num_cols, cat_cols)
            X_transformed = preprocessor_internal.fit_transform(df_data)
            
            return X_transformed, preprocessor_internal 

        # Procesamos los datos
        X_processed, preprocessor_fitted = get_processed_data(df_raw, numerical_features, categorical_features) 


        # ----------------------------------------------------------------------------------
        # --- 4. EJECUCIÓN Y VISUALIZACIÓN DEL CLUSTERING ---
        # ----------------------------------------------------------------------------------

        st.markdown("#### Paso 2: Evaluación y Entrenamiento del Modelo (K-Means)")

        # 4.1. Método del Codo con Detección Automática de K
        st.subheader("1. Determinación Automática de K (Método del Codo)")
        k_range = st.slider("Selecciona el rango máximo de K a evaluar:", 2, 20, 10)

        def run_elbow_method(X_data, max_k):
            """Calcula WCSS para K."""
            wcss = []
            k_values = range(1, max_k + 1)
            
            with st.spinner(f"Calculando inercia para K de 1 a {max_k}..."):
                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
                    kmeans.fit(X_data)
                    wcss.append(kmeans.inertia_)
            
            return k_values, wcss

        def find_optimal_k(k_values, wcss):
            """Heurística para encontrar el codo."""
            if len(k_values) < 3: return 4
            
            diffs = np.diff(wcss)
            k_suggested = np.argmin(diffs) + 1 
            
            return max(2, k_suggested)


        k_values, wcss = run_elbow_method(X_processed, k_range)
        optimal_k_auto = find_optimal_k(k_values, wcss)


        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_values, wcss, 'bx-')
            ax.vlines(optimal_k_auto, min(wcss), max(wcss), linestyles='--', colors='r', label=f'K Sugerido: {optimal_k_auto}')
            ax.set_xlabel("Número de Clústeres (K)")
            ax.set_ylabel("WCSS (Varianza Intra-Clúster)")
            ax.set_title("Método del Codo con Detección Automática")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.markdown("#### 🎯 K Sugerido")
            st.info(f"El algoritmo sugiere un K de **{optimal_k_auto}**. ¡Puedes ajustarlo visualmente!")
            selected_k = st.number_input(
                "Número de Clústeres (K):", 
                min_value=2, 
                max_value=20, 
                value=optimal_k_auto, 
                step=1
            )

        # 4.2. Ejecución de K-Means y Análisis

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
            
            st.success(f"Clustering completado. Se crearon **{selected_k} segmentos** de mercado. ¡Listo para interpretar!")

            # A. Distribución y Perfiles (Visualización)
            st.subheader("2. Perfiles y Distribución de Clústeres")
            
            col_dist, col_map = st.columns(2)
            
            with col_dist:
                st.markdown("##### A. Distribución y Perfiles Numéricos")
                cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
                st.dataframe(cluster_counts.rename("N° Proyectos"))

                all_numeric_cols = list(set(numerical_features) & set(df_clustered.columns))
                numeric_summary = df_clustered.groupby('Cluster')[all_numeric_cols].mean().T
                st.dataframe(
                    numeric_summary.style.background_gradient(cmap='viridis', axis=1), 
                    caption="Valores promedio clave por clúster."
                )

            with col_map:
                st.markdown("##### B. Visualización Geográfica (Latitud vs. Longitud)")
                if 'latitud' in df_clustered.columns and 'longitud' in df_clustered.columns:
                    fig_map = px.scatter(
                        df_clustered, 
                        x='longitud', 
                        y='latitud', 
                        color=df_clustered['Cluster'].astype(str), 
                        hover_data=['regional', 'estrato', 'precioenmiles'],
                        title=f"Segmentación Geográfica (K={selected_k})",
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

else:
    st.info("⬆️ Esperando a que subas el archivo de datos `.parquet` para comenzar el análisis de clustering.")
