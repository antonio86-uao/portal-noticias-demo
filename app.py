# =============================================================================
# INTEGRACIÓN DEL MODELO REAL EN LA APP
# Pasos para reemplazar el simulador con DistilBERT fine-tuneado
# =============================================================================

# PASO 1: COMPRIMIR Y SUBIR EL MODELO
# En Colab, ejecuta esto para preparar el modelo:

import shutil
import os

# Comprimir la carpeta del modelo
modelo_path = "./distilbert-ag-news-finetuned"
shutil.make_archive("modelo_finetuned", 'zip', modelo_path)
print("✅ Modelo comprimido en modelo_finetuned.zip")

# Verificar archivos
print("\n📁 Archivos del modelo:")
for file in os.listdir(modelo_path):
    size = os.path.getsize(os.path.join(modelo_path, file)) / (1024*1024)
    print(f"  {file}: {size:.1f} MB")

# PASO 2: DESCARGAR EL MODELO
# Ejecuta esto en Colab para descargar:
from google.colab import files
files.download("modelo_finetuned.zip")

# =============================================================================
# PASO 3: NUEVA VERSIÓN DE LA APP CON MODELO REAL
# =============================================================================

# Este es el código actualizado para app.py:

import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import numpy as np
from datetime import datetime
import time
import zipfile
import os

# Configuración de la página
st.set_page_config(
    page_title="Portal de Clasificación de Noticias",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (igual que antes)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .real-model-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Función para descargar y extraer modelo desde GitHub Releases
@st.cache_data
def download_model():
    """Descarga el modelo desde GitHub Releases"""
    import urllib.request
    
    model_dir = "distilbert-ag-news-finetuned"
    
    if not os.path.exists(model_dir):
        st.info("⬇️ Descargando modelo fine-tuneado...")
        
        # URL de tu release en GitHub
        model_url = "https://github.com/antonio86-uao/portal-noticias-demo/releases/download/v1.0/modelo_finetuned.zip"
        
        try:
            urllib.request.urlretrieve(model_url, "modelo_finetuned.zip")
            
            with zipfile.ZipFile("modelo_finetuned.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.remove("modelo_finetuned.zip")
            st.success("✅ Modelo descargado exitosamente!")
            
        except Exception as e:
            st.error(f"❌ Error descargando modelo: {e}")
            return False
    
    return True

# Función para cargar el modelo REAL
@st.cache_resource
def load_real_model():
    """Carga el modelo DistilBERT fine-tuneado REAL"""
    
    if not download_model():
        return None, None
    
    try:
        model_path = "distilbert-ag-news-finetuned"
        
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Crear pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return classifier, tokenizer
        
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

# Función para clasificar texto con modelo REAL
def classify_text_real(text, classifier):
    """Clasifica un texto usando el modelo DistilBERT fine-tuneado REAL"""
    if not classifier:
        return None
    
    try:
        # Hacer predicción
        results = classifier(text)
        
        # Mapear resultados
        label_map = {
            'LABEL_0': 'World',
            'LABEL_1': 'Sports', 
            'LABEL_2': 'Business',
            'LABEL_3': 'Sci/Tech'
        }
        
        predictions = {}
        for result in results[0]:
            label = result['label']
            category = label_map.get(label, label)
            predictions[category] = result['score']
        
        return predictions
        
    except Exception as e:
        st.error(f"Error en la clasificación: {str(e)}")
        return None

# Función para crear gráfico de confianza (CORREGIDA)
def create_confidence_chart(predictions):
    """Crea un gráfico de barras con las predicciones"""
    if not predictions:
        return None
    
    df = pd.DataFrame(list(predictions.items()), columns=['Categoría', 'Confianza'])
    df = df.sort_values('Confianza', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Categoría'],
            x=df['Confianza'],
            orientation='h',
            marker=dict(
                color=df['Confianza'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Confianza")
            ),
            text=[f'{x:.1%}' for x in df['Confianza']],
            textposition='auto'  # ✅ CORREGIDO
        )
    ])
    
    fig.update_layout(
        title="Confianza por Categoría",
        xaxis_title="Confianza",
        yaxis_title="Categoría",
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='.0%')
    )
    
    return fig

# INTERFAZ PRINCIPAL
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">📰 Portal de Clasificación de Noticias</h1>', 
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="real-model-badge">🔥 MODELO REAL ACTIVO</span></div>', 
                    unsafe_allow_html=True)
    
    st.markdown("### Clasificación automática usando DistilBERT Fine-tuneado")
    st.info("🎯 **Modelo Real:** DistilBERT fine-tuneado en AG News con 94.72% de accuracy!")
    
    # Cargar modelo
    with st.spinner("🔄 Cargando modelo DistilBERT fine-tuneado..."):
        classifier, tokenizer = load_real_model()
    
    if not classifier:
        st.error("❌ No se pudo cargar el modelo. Verifique la configuración.")
        return
    
    st.success("✅ Modelo DistilBERT cargado exitosamente! (94.72% accuracy)")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Modelo Fine-tuneado")
        st.markdown("""
        **🔥 Modelo Real Activo**
        
        **Arquitectura:** DistilBERT  
        **Dataset:** AG News (120K ejemplos)  
        **Accuracy:** 94.72%  
        **Épocas:** 3  
        **Tiempo entrenamiento:** ~25 minutos
        
        **Autor:** Gabriel Antonio Vallejo Loaiza  
        **Universidad:** Autónoma de Occidente
        """)
        
        st.header("🎯 Ejemplos de Prueba")
        examples = {
            "World 🌍": "Breaking news: International summit discusses climate change policies",
            "Sports ⚽": "Manchester United defeats Chelsea 3-1 in Premier League final", 
            "Business 💼": "Apple stock rises 5% after quarterly earnings exceed expectations",
            "Sci/Tech 🔬": "Scientists develop breakthrough AI algorithm for medical diagnosis"
        }
        
        for category, text in examples.items():
            if st.button(f"Probar {category}"):
                st.session_state.example_text = text
    
    # Input de texto
    st.header("📝 Clasificar Noticia con DistilBERT")
    
    default_text = st.session_state.get('example_text', '')
    
    text_input = st.text_area(
        "Ingresa el texto de la noticia:",
        value=default_text,
        height=150,
        placeholder="Ejemplo: Scientists discover new exoplanet using advanced AI telescopes..."
    )
    
    # Botón de clasificación
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_button = st.button("🚀 Clasificar con DistilBERT", type="primary")
    
    # Procesamiento con modelo REAL
    if classify_button and text_input.strip():
        with st.spinner("🤖 Clasificando con DistilBERT..."):
            start_time = time.time()
            predictions = classify_text_real(text_input, classifier)
            processing_time = time.time() - start_time
        
        if predictions:
            best_category = max(predictions, key=predictions.get)
            best_confidence = predictions[best_category]
            
            st.header("🎯 Resultado del Modelo Fine-tuneado")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                icons = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}
                
                st.markdown(f"""
                **📂 Categoría:** {icons.get(best_category, '📰')} {best_category}  
                **🎯 Confianza:** {best_confidence:.2%}  
                **⚡ Tiempo:** {processing_time:.3f}s  
                **🤖 Modelo:** DistilBERT Real (94.72% accuracy)
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Todas las Predicciones")
                for category, confidence in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                    icon = icons.get(category, '📰')
                    st.write(f"**{icon} {category}:** {confidence:.2%}")
                    st.progress(confidence)
            
            # Gráfico
            st.subheader("📈 Visualización de Confianza")
            fig = create_confidence_chart(predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif classify_button:
        st.warning("⚠️ Por favor, ingresa texto para clasificar.")
    
    # Footer
    st.markdown("---")
    st.markdown("**🎓 Taller Final - Procesamiento de Datos Secuenciales | 👨‍🎓 Gabriel Vallejo | 🏫 UAO**")

if __name__ == "__main__":
    main()
