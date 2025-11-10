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

# **IMPORTANTE:** Definiciones de variables según su estrategia de agregación
# Se asume que la columna 'id_proyecto' es la clave única para el Nivel P

# Variables de Nivel P (No requieren agregación, solo limpieza)
P_COLS = ['id_proyecto', 'regional', 'ciudad', 'zona', 'barrio', 'localidad_comuna', 'estrato', 
          'longitud', 'latitud', 'numero_etapas', 'area_lote', 'area_construida', 
          'area_vendible', 'numero_bloques', 'total_parqueaderos']

# Variables de Nivel E/T que requieren Agregación

# 1. Variables a SUMAR (Volumen)
SUM_COLS = ['saldo', 'ventas', 'renuncias', 'numero_unidades'] 

# 2. Variables a PROMEDIAR PONDERADAMENTE (Valor/Dimensión). Usaremos 'numero_unidades' como peso.
WEIGHTED_MEAN_COLS = ['precioenmiles', 'preciomc', 'area_por_tipo', 'alcobas', 'baños']

# 3. Variables a obtener la MODA PONDERADA (Calidad/Nicho). Usaremos 'numero_unidades' como peso.
WEIGHTED_MODE_COLS = [
    'sistema_constructivo', 'cimentación', 'divison_interior', 'placa_entre_piso', 
    'fachada', 'ventanas', 'fase', 'estado', 'modalidad', 'uso', 'tipo_vivienda', 
    'nombre_tipo', 'condicion_entrega', 'meson_cocina', 'muebles_cocina', 
    'pisos_alcobas', 'pisos_baño', 'pisos_cocina', 'puerta_principal', 'tipo_cocina'
]

# Nota: Las variables de zonacomun/dotaciones son binarias y se pueden manejar como MODA PONDERADA,
# o simplemente tomando el MÁXIMO (1) si la característica existe en cualquier unidad/etapa.
# Usaremos Moda Ponderada para simplificar.

# --- 1. Funciones de Agregación (Lógica Nivel E/T -> P) ---

def weighted_mode(series, weights):
    """Calcula la Moda Ponderada para variables categóricas/discretas."""
    df_temp = pd.DataFrame({'value': series, 'weight': weights})
    # Eliminar NaNs para evitar errores de agrupación
    df_temp.dropna(subset=['value'], inplace=True) 
    if df_temp.empty:
        return 'N/A'
    
    # Sumar los pesos por cada valor único
    weighted_counts = df_temp.groupby('value')['weight'].sum()
    # Devolver el valor con el mayor peso
    return weighted_counts.idxmax() if not weighted_counts.empty else 'N/A'


@st.cache_data(show_spinner="⏳ Aplicando lógica de agregación (Nivel E/T a Nivel P)...")
def aggregate_data_to_project_level(df_full):
    """
    Agrega los datos de Nivel E (Etapa) y T (Tipo) al Nivel P (Proyecto)
    siguiendo las reglas de Suma, Promedio Ponderado y Moda Ponderada.
    """
    # 1. Limpieza inicial para la columna de peso y la clave de proyecto
    if 'id_proyecto' not in df_full.columns or 'numero_unidades' not in df_full.columns:
        st.error("🚨 Faltan las columnas clave: 'id_proyecto' o 'numero_unidades'.")
        return pd.DataFrame()

    df_full['numero_unidades'] = pd.to_numeric(df_full['numero_unidades'], errors='coerce').fillna(0)
    
    # 2. Separar variables de Nivel P que solo necesitan el primer/último valor (o la moda simple si hay repetición)
    # y asegurarse de que solo se tome una fila por proyecto.
    df_p = df_full.drop_duplicates(subset=['id_proyecto'], keep='first')[P_COLS]
    
    # 3. Preparar la agregación de Nivel E/T
    agg_dict = {}

    # Agregación por SUMA
    for col in SUM_COLS:
        if col in df_full.columns:
            agg_dict[col] = 'sum'

    # Agregación por PROMEDIO PONDERADO
    temp_df = df_full.copy()
    for col in WEIGHTED_MEAN_COLS:
        if col in df_full.columns:
            # Calcular el numerador (col * peso) y el denominador (peso total)
            temp_df[f'{col}_num'] = temp_df[col] * temp_df['numero_unidades']
            agg_dict[f'{col}_num'] = 'sum'
            agg_dict['numero_unidades'] = 'sum' # Para obtener el divisor total

    # Agregación por MODA PONDERADA (se hace en un paso posterior para evitar problemas de GroupBy)
    
    # Ejecutar el GroupBy para SUMA y PROMEDIOS PONDERADOS (numerador)
    df_aggregated_num = temp_df.groupby('id_proyecto').agg(agg_dict).reset_index()

    # Calcular el PROMEDIO PONDERADO final
    for col in WEIGHTED_MEAN_COLS:
        if col in df_full.columns:
            df_aggregated_num[col] = (
                df_aggregated_num[f'{col}_num'] / df_aggregated_num['numero_unidades']
            ).fillna(0)
            df_aggregated_num.drop(columns=[f'{col}_num'], inplace=True)
            
    # Renombrar la columna de suma de unidades para evitar colisión si ya existe en SUM_COLS
    df_aggregated_num.rename(columns={'numero_unidades': 'total_unidades_agregadas'}, inplace=True)
    if 'numero_unidades' in df_aggregated_num.columns and 'numero_unidades' in SUM_COLS:
         df_aggregated_num.rename(columns={'total_unidades_agregadas': 'numero_unidades'}, inplace=True)


    # 4. Agregación por MODA PONDERADA (Iterativa)
    df_mode = df_aggregated_num[['id_proyecto']].copy()
    for col in WEIGHTED_MODE_COLS:
        if col in df_full.columns:
            df_mode[col] = df_full.groupby('id_proyecto').apply(
                lambda x: weighted_mode(x[col], x['numero_unidades'])
            ).reset_index(level=0, drop=True)
            
    # 5. Combinar todos los resultados
    # Fusionar Nivel P + Agregaciones Numéricas + Agregaciones Categóricas
    df_final = df_p.merge(df_aggregated_num, on='id_proyecto', how='left', suffixes=('_p', '_agg'))
    df_final = df_final.merge(df_mode, on='id_proyecto', how='left', suffixes=('_final', '_mode'))
    
    # Limpieza final: Eliminar columnas duplicadas/innecesarias de la fusión
    cols_to_drop = [col for col in df_final.columns if col.endswith(('_p', '_agg', '_final', '_mode')) and col.split('_')[0] in P_COLS]
    df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Renombrar 'total_unidades_agregadas' si no es una columna final deseada
    if 'total_unidades_agregadas' in df_final.columns:
         df_final.drop(columns=['total_unidades_agregadas'], inplace=True)

    return df_final.drop_duplicates(subset=['id_proyecto'])


# --- 2. Interfaz y Carga de Archivo (Requisito 2) ---

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
            
            # Las features finales son las que sobrevivieron a la agregación
            all_cols = P_COLS + SUM_COLS + WEIGHTED_MEAN_COLS + WEIGHTED_MODE_COLS
            
            numerical_features = [col for col in all_cols if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64']]
            categorical_features = [col for col in all_cols if col in df_processed.columns and df_processed[col].dtype == 'object']
            
            # Filtrar 'id_proyecto' y otras claves que puedan haber quedado
            numerical_features = [col for col in numerical_features if col not in ['id_proyecto']]
            categorical_features = [col for col in categorical_features if col not in ['id_proyecto']]

            st.info(
                f"Se usarán **{len(numerical_features)}** variables numéricas (Escalado) y **{len(categorical_features)}** variables categóricas (OHE)."
            )
            
            # --- 4. Pipeline de Preprocesamiento y Ejecución K-Means ---
            
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
                # Solo usamos las columnas que se van a transformar
                X_transformed = preprocessor_internal.fit_transform(df_data[num_cols + cat_cols])
                return X_transformed, preprocessor_internal

            X_transformed, preprocessor_fitted = get_transformed_data(df_processed, numerical_features, categorical_features)
            
            
            # --- 5. Ejecución y Visualización del Clustering (Mismo código anterior) ---
            
            st.markdown("---")
            st.markdown("#### Paso 2: Evaluación y Entrenamiento del Modelo (K-Means)")
            
            # Función para el método del codo (se asume que existe o se define aquí)
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

            k_range = st.slider("Rango máximo de K a evaluar:", 2, 20, 10)
            k_values, wcss = run_elbow_method(X_transformed, k_range)
            optimal_k_auto = find_optimal_k(k_values, wcss)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(k_values, wcss, 'bx-')
                ax.vlines(optimal_k_auto, min(wcss), max(wcss), linestyles='--', colors='r')
                ax.set_title("Método del Codo")
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

                # Sección de Visualización de Resultados (similar a la original)
                st.subheader("2. Perfiles y Distribución de Clústeres")
                
                col_dist, col_map = st.columns(2)
                with col_dist:
                    numeric_summary = df_clustered.groupby('Cluster')[numerical_features].mean().T
                    st.dataframe(numeric_summary.style.background_gradient(cmap='viridis', axis=1), caption="Valores promedio clave por clúster.")
                
                with col_map:
                    fig_map = px.scatter(
                        df_clustered, x='longitud', y='latitud', 
                        color=df_clustered['Cluster'].astype(str),
                        hover_data=['regional', 'estrato', 'precioenmiles'],
                        title="Segmentación Geográfica"
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                
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
        st.error(f"❌ Error durante la ejecución del proceso. Asegúrese de que el archivo contiene las columnas clave 'id_proyecto' y 'numero_unidades'. Mensaje: {e}")
