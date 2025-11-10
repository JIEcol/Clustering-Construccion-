import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 0. Configuración Inicial y Definición de Variables ---

st.set_page_config(layout="wide")
st.title("🏗️ Análisis de Clustering (K-Means) de Proyectos de Construcción")
st.markdown("---")

# **<-- CORRECCIÓN 1: DEFINICIÓN DE CLAVES -->**
PROJECT_ID_COL = 'identificador'
WEIGHT_COL = 'unidades'

# Variables de Nivel P (Ajustadas a una estructura general)
P_COLS = [PROJECT_ID_COL, 'regional', 'ciudad', 'zona', 'barrio', 'estrato', 
          'longitud', 'latitud', 'numero_etapas', 'area_lote', 'area_construida', 
          'area_vendible', 'numero_bloques', 'total_parqueaderos']

# Variables de Nivel E/T que requieren Agregación

# 1. Variables a SUMAR (Volumen) - Usamos la nueva columna de peso
SUM_COLS = ['saldo', 'ventas', 'renuncias', WEIGHT_COL] 

# 2. Variables a PROMEDIAR PONDERADAMENTE (Valor/Dimensión)
WEIGHTED_MEAN_COLS = ['precioenmiles', 'preciomc', 'area_por_tipo', 'alcobas', 'baños']

# 3. Variables a obtener la MODA PONDERADA (Calidad/Nicho)
WEIGHTED_MODE_COLS = [
    'sistema_constructivo', 'cimentación', 'divison_interior', 'placa_entre_piso', 
    'fachada', 'ventanas', 'fase', 'estado', 'modalidad', 'uso', 'tipo_vivienda', 
    'nombre_tipo', 'condicion_entrega', 'meson_cocina', 'muebles_cocina', 
    'pisos_alcobas', 'pisos_baño', 'pisos_cocina', 'puerta_principal', 'tipo_cocina'
]


# --- 1. Funciones de Agregación (Lógica Nivel E/T -> P) ---

def general_mode(series, weights, is_weighted):
    """Calcula la Moda Ponderada o Simple (Nueva función de respaldo)."""
    df_temp = pd.DataFrame({'value': series, 'weight': weights})
    df_temp.dropna(subset=['value'], inplace=True) 
    if df_temp.empty:
        return 'N/A'
    
    if is_weighted:
        # Moda Ponderada
        weighted_counts = df_temp.groupby('value')['weight'].sum()
        return weighted_counts.idxmax() if not weighted_counts.empty else 'N/A'
    else:
        # Moda Simple
        return series.mode()[0] if not series.mode().empty else 'N/A'


@st.cache_data(show_spinner="⏳ Aplicando lógica de agregación (Nivel E/T a Nivel P)...")
def aggregate_data_to_project_level(df_full):
    """
    Agrega los datos, usando la Media Aritmética simple si 'unidades' no está presente,
    pero asumiendo 'identificador' como ID de proyecto.
    """
    
    # **<-- CORRECCIÓN 2: VERIFICACIÓN DE CLAVES -->**
    if PROJECT_ID_COL not in df_full.columns:
        st.error(f"🚨 Faltan la columna clave: **'{PROJECT_ID_COL}'**. El proceso debe detenerse.")
        return pd.DataFrame()
    
    # **<-- CORRECCIÓN 3: VERIFICACIÓN DE COLUMNA DE PESO (Respaldo) -->**
    is_weighted = WEIGHT_COL in df_full.columns
    weight_col_name = WEIGHT_COL
    
    if not is_weighted:
        st.warning(f"⚠️ **ATENCIÓN:** Falta la columna **'{WEIGHT_COL}'**. La agregación usará la **Media Aritmética Simple**.")
        df_full['peso_unitario_respaldo'] = 1
        weight_col_name = 'peso_unitario_respaldo'
    else:
        # Usar el nombre original para la columna de peso si existe
        df_full[WEIGHT_COL] = pd.to_numeric(df_full[WEIGHT_COL], errors='coerce').fillna(0)
    
    # 1. Limpieza inicial para la columna de peso
    df_full[weight_col_name] = df_full[weight_col_name].replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Separar variables de Nivel P (solo tomamos la primera fila por proyecto)
    df_p = df_full.drop_duplicates(subset=[PROJECT_ID_COL], keep='first')[P_COLS]
    
    # 3. Preparar la agregación de Nivel E/T
    agg_dict = {}
    temp_df = df_full.copy()
    temp_df['total_peso'] = temp_df[weight_col_name]

    # Agregación por SUMA
    for col in SUM_COLS:
        if col in df_full.columns:
            agg_dict[col] = 'sum'

    # Agregación por PROMEDIO PONDERADO / MEDIA SIMPLE
    for col in WEIGHTED_MEAN_COLS:
        if col in df_full.columns:
            # Calcular el numerador (col * peso)
            temp_df[f'{col}_num'] = pd.to_numeric(temp_df[col], errors='coerce') * temp_df[weight_col_name]
            agg_dict[f'{col}_num'] = 'sum'

    agg_dict['total_peso'] = 'sum'
    
    # Ejecutar el GroupBy para SUMA y PROMEDIOS PONDERADOS
    df_aggregated_num = temp_df.groupby(PROJECT_ID_COL).agg(agg_dict).reset_index()

    # Calcular el PROMEDIO FINAL
    for col in WEIGHTED_MEAN_COLS:
        if col in df_full.columns:
            # El promedio ponderado / simple se calcula aquí
            df_aggregated_num[col] = (
                df_aggregated_num[f'{col}_num'] / df_aggregated_num['total_peso']
            ).fillna(0)
            df_aggregated_num.drop(columns=[f'{col}_num'], inplace=True, errors='ignore')
            
    df_aggregated_num.drop(columns=['total_peso'], inplace=True, errors='ignore')


    # 4. Agregación por MODA PONDERADA / SIMPLE (Iterativa)
    df_mode = df_aggregated_num[[PROJECT_ID_COL]].copy()
    for col in WEIGHTED_MODE_COLS:
        if col in df_full.columns:
            # Aplicar la función de modo general (ponderado o simple)
            df_mode[col] = df_full.groupby(PROJECT_ID_COL).apply(
                lambda x: general_mode(x[col], x[weight_col_name], is_weighted)
            ).reset_index(level=0, drop=True)
            
    # 5. Combinar todos los resultados
    df_final = df_p.merge(df_aggregated_num, on=PROJECT_ID_COL, how='left')
    df_final = df_final.merge(df_mode, on=PROJECT_ID_COL, how='left')
    
    # Limpieza final de columnas auxiliares
    cols_to_drop = [col for col in df_final.columns if col.endswith(('_p', '_agg', '_final', '_mode', '_respaldo'))]
    df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return df_final.drop_duplicates(subset=[PROJECT_ID_COL])


# --- 2. Interfaz y Carga de Archivo ---

st.markdown("#### Paso 1: Carga y Agregación del Dataset (P, E, T)")

uploaded_file = st.file_uploader(
    "📂 Cargue el archivo Excel o CSV que contiene datos de Nivel P, E y T", 
    type=['xlsx', 'csv']
)

if uploaded_file:
    try:
        # Intenta leer el archivo
        if uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df_raw = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Archivo cargado con **{df_raw.shape[0]}** filas.")

        # APLICAR LA LÓGICA DE AGREGACIÓN
        df_processed = aggregate_data_to_project_level(df_raw)

        if not df_processed.empty:
            st.success(f"✅ Agregación completada. Dataset reducido a **{df_processed.shape[0]}** Proyectos (Nivel P).")
            st.dataframe(df_processed.head())
            
            # --- 3. Definición de Features para K-Means ---
            
            all_cols = P_COLS + SUM_COLS + WEIGHTED_MEAN_COLS + WEIGHTED_MODE_COLS
            
            # Las columnas deben existir en df_processed después de la agregación
            numerical_features = [col for col in all_cols if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64']]
            categorical_features = [col for col in all_cols if col in df_processed.columns and df_processed[col].dtype == 'object']
            
            # Filtrar las claves (identificador)
            numerical_features = [col for col in numerical_features if col not in [PROJECT_ID_COL]]
            categorical_features = [col for col in categorical_features if col not in [PROJECT_ID_COL]]
            
            # Rellenar NaNs en categóricas después de la agregación por si acaso
            for col in categorical_features:
                 df_processed[col] = df_processed[col].fillna('N/A')

            st.info(
                f"Se usarán **{len(numerical_features)}** variables numéricas (Escalado) y **{len(categorical_features)}** variables categóricas (OHE)."
            )
            
            # --- 4. Pipeline de Preprocesamiento (OHE y Estandarización) ---
            
            def build_preprocessor(num_cols, cat_cols):
                numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, num_cols),
                        ('cat', categorical_transformer, cat_cols)
                    ],
                    remainder='drop'
                )
                return preprocessor

            @st.cache_data(show_spinner="Aplicando OHE y Estandarización...")
            def get_transformed_data(df_data, num_cols, cat_cols):
                preprocessor_internal = build_preprocessor(num_cols, cat_cols)
                X_transformed = preprocessor_internal.fit_transform(df_data[num_cols + cat_cols])
                return X_transformed, preprocessor_internal

            X_transformed, preprocessor_fitted = get_transformed_data(df_processed, numerical_features, categorical_features)
            
            
            # --- 5. Ejecución y Visualización del Clustering ---
            
            st.markdown("---")
            st.markdown("#### Paso 2: Evaluación y Entrenamiento del Modelo (K-Means)")
            
            def run_elbow_method(X_data, max_k):
                wcss = []
                k_values = range(1, max_k + 1)
                for k in k_values:
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
                    kmeans_model.fit(X_data)
                    wcss.append(kmeans_model.inertia_)
                return k_values, wcss

            def find_optimal_k(k_values, wcss):
                if len(k_values) < 3: return 4
                diffs = np.diff(wcss)
                k_suggested = np.argmin(diffs) + 1
                return max(2, k_suggested)

            st.subheader("1. Determinación Automática de K (Método del Codo)")
            k_range = st.slider("Rango máximo de K a evaluar:", 2, 20, 10)
            k_values, wcss = run_elbow_method(X_transformed, k_range)
            optimal_k_auto = find_optimal_k(k_values, wcss)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(k_values, wcss, 'bx-')
                ax.vlines(optimal_k_auto, min(wcss), max(wcss), linestyles='--', colors='r', label=f'K Sugerido: {optimal_k_auto}')
                ax.set_title("Método del Codo")
                ax.set_xlabel("Número de Clústeres (K)")
                ax.set_ylabel("WCSS (Varianza Intra-Clúster)")
                st.pyplot(fig)

            with col2:
                st.info(f"K Sugerido: **{optimal_k_auto}**.")
                selected_k = st.number_input("Número de Clústeres (K):", min_value=2, max_value=20, value=optimal_k_auto, step=1)

            if st.button(f"🚀 Ejecutar K-Means y Segmentar con K={selected_k}", type="primary"):
                
                @st.cache_data(show_spinner=f"Entrenando K-Means con {selected_k} clústeres...")
                def run_kmeans_clustering(X_data, df_original, k):
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
                    kmeans_model.fit(X_data) 
                    df_clustered = df_original.copy()
                    df_clustered['Cluster'] = kmeans_model.labels_
                    return df_clustered
                
                df_clustered = run_kmeans_clustering(X_transformed, df_processed, selected_k)
                st.success(f"Clustering completado. **{selected_k} segmentos** creados.")

                # Sección de Visualización de Resultados 
                st.subheader("2. Perfiles y Distribución de Clústeres")
                
                col_dist, col_map = st.columns(2)
                
                with col_dist:
                    st.markdown("##### A. Perfiles Numéricos (Valores Promedio)")
                    numeric_summary = df_clustered.groupby('Cluster')[numerical_features].mean().T
                    st.dataframe(
                        numeric_summary.style.background_gradient(cmap='viridis', axis=1), 
                        caption="Valores promedio clave por clúster."
                    )
                
                with col_map:
                    # Se mantiene la visualización geográfica asumiendo que las columnas existen en el DF final
                    if 'latitud' in df_clustered.columns and 'longitud' in df_clustered.columns:
                        fig_map = px.scatter(
                            df_clustered, x='longitud', y='latitud', 
                            color=df_clustered['Cluster'].astype(str),
                            hover_data=['regional', 'estrato', 'precioenmiles'],
                            title="Segmentación Geográfica"
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.warning("Faltan las columnas 'latitud' o 'longitud' para el mapa. Asegúrese de que existan en su base original o en P_COLS.")

                # Descargar los resultados
                st.markdown("---")
                csv = df_clustered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar DataFrame con Etiquetas de Clúster",
                    data=csv,
                    file_name='proyectos_clusterizados_final.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"❌ Error durante la ejecución del proceso. Mensaje: {e}")
