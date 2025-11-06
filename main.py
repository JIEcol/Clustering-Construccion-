import streamlit as st
import pandas as pd

# La URL "raw" del archivo en GitHub
# (Simplemente reemplaza 'blob' por 'raw' en la URL original del archivo en GitHub)
GITHUB_EXCEL_URL = (
    "https://raw.githubusercontent.com/JIEcol/Clustering-Construccion-/main/mapa_3y4_cs_v2_sep25.xlsx"
)

# Título de la aplicación
st.title("🏗️ Lector de Datos de Construcción")
st.markdown("---")

# Usamos st.cache_data para que Streamlit solo cargue el archivo una vez,
# lo que es más rápido y eficiente.

@st.cache_data
def load_data(url):
    """Carga los datos desde la URL de GitHub."""
    try:
        # Usamos pd.read_excel para leer el archivo directamente desde la URL
        df = pd.read_excel(url, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Llamar a la función para cargar los datos
df = load_data(GITHUB_EXCEL_URL)

# Mostrar los datos si se cargaron correctamente
if df is not None:
    st.success("✅ ¡Archivo Excel cargado exitosamente!")
    
    # Muestra las primeras filas del DataFrame
    st.subheader("Vista Previa del DataFrame")
    st.dataframe(df.head())
    
    # Muestra información básica
    st.subheader("Información General")
    st.write(f"Número de filas: **{df.shape[0]}**")
    st.write(f"Número de columnas: **{df.shape[1]}**")
    
    # Podrías agregar más análisis o visualizaciones aquí
else:
    st.info("⚠️ No se pudo cargar el archivo. Por favor, verifica la URL.")
