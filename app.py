# =============================================================================
# PORTAL INTELIGENTE DE CLASIFICACIÓN DE NOTICIAS
# Aplicación Streamlit con DistilBERT desde Hugging Face
# Versión: 2.0.0 - Modelo Real Integrado
# =============================================================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Portal de Clasificación de Noticias",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo desde Hugging Face
@st.cache_resource
def load_model_from_hf():
    """Carga el modelo DistilBERT desde Hugging Face"""
    try:
        from transformers import pipeline
        
        # Cargar el pipeline desde tu repositorio de Hugging Face
        classifier = pipeline(
            "text-classification",
            model="gaanvalo/distilbert-finetuned-noticias",
            return_all_scores=True
        )
        
        return classifier
        
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None

# Función para clasificar texto con modelo real
def classify_text_real(text, classifier):
    """Clasifica un texto usando el modelo DistilBERT desde Hugging Face"""
    if not classifier:
        return None
    
    try:
        # Hacer predicción
        results = classifier(text)
        
        # Mapear resultados (ajustar según las etiquetas de tu modelo)
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

# Función para crear gráfico de confianza
def create_confidence_chart(predictions):
    """Crea un gráfico de barras con las predicciones"""
    if not predictions:
        return None
    
    df = pd.DataFrame(list(predictions.items()), columns=['Categoría', 'Confianza'])
    df = df.sort_values('Confianza', ascending=True)
    
    # Crear gráfico
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
            textposition='auto'
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
    
    # Cargar modelo
    with st.spinner("🔄 Cargando modelo DistilBERT desde Hugging Face..."):
        classifier = load_model_from_hf()
    
    if not classifier:
        st.error("❌ No se pudo cargar el modelo desde Hugging Face.")
        st.info("Verifique que el modelo 'gaanvalo/distilbert-finetuned-noticias' esté público y correctamente subido.")
        return
    
    st.success("✅ Modelo DistilBERT cargado exitosamente desde Hugging Face! (94.72% accuracy)")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Modelo Real Activo")
        st.markdown("""
        **🔥 DistilBERT Fine-tuneado**
        
        **🤖 Fuente:** Hugging Face Hub  
        **📊 Accuracy:** 94.72%  
        **📚 Dataset:** AG News (120K ejemplos)  
        **⚡ Épocas:** 3  
        **🔗 Modelo:** gaanvalo/distilbert-finetuned-noticias
        
        **👨‍🎓 Autor:** Gabriel Antonio Vallejo Loaiza  
        **🏫 Universidad:** Autónoma de Occidente
        
        **Categorías:**
        - 🌍 World (Internacional)
        - ⚽ Sports (Deportes) 
        - 💼 Business (Negocios)
        - 🔬 Sci/Tech (Ciencia/Tecnología)
        """)
        
        st.header("🎯 Ejemplos de Prueba")
        examples = {
            "World 🌍": "International climate summit reaches historic agreement on carbon emissions reduction",
            "Sports ⚽": "Manchester United defeats Real Madrid 3-1 in Champions League final", 
            "Business 💼": "Apple stock surges 8% after quarterly earnings exceed Wall Street expectations",
            "Sci/Tech 🔬": "Scientists develop breakthrough AI algorithm for early cancer detection"
        }
        
        for category, text in examples.items():
            if st.button(f"Probar {category}"):
                st.session_state.example_text = text
    
    # Características del sistema
    st.header("🚀 Características del Portal")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>⚡ Tiempo Real</h4>
        <p>Clasificación con modelo real en menos de 3 segundos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>📊 Análisis Visual</h4>
        <p>Gráficos interactivos de confianza por categoría</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h4>🎯 Alta Precisión</h4>
        <p>Modelo fine-tuneado con 94.72% de accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input de texto
    st.header("📝 Clasificar Noticia con DistilBERT")
    
    default_text = st.session_state.get('example_text', '')
    
    text_input = st.text_area(
        "Ingresa el texto de la noticia:",
        value=default_text,
        height=150,
        placeholder="Ejemplo: Tesla announces revolutionary breakthrough in autonomous driving technology..."
    )
    
    # Botón de clasificación
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_button = st.button("🚀 Clasificar con DistilBERT", type="primary")
    
    # Procesamiento con modelo real
    if classify_button and text_input.strip():
        with st.spinner("🤖 Clasificando con DistilBERT desde Hugging Face..."):
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
                
                # Determinar color según confianza
                if best_confidence > 0.7:
                    confidence_class = "confidence-high"
                elif best_confidence > 0.4:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                icons = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}
                
                st.markdown(f"""
                **📂 Categoría:** {icons.get(best_category, '📰')} {best_category}  
                **🎯 Confianza:** <span class="{confidence_class}">{best_confidence:.2%}</span>  
                **⚡ Tiempo:** {processing_time:.3f}s  
                **🤖 Fuente:** Hugging Face - DistilBERT Real
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
            
            # Historial
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'Hora': datetime.now().strftime("%H:%M:%S"),
                'Texto': text_input[:80] + "..." if len(text_input) > 80 else text_input,
                'Categoría': f"{icons.get(best_category, '📰')} {best_category}",
                'Confianza': f"{best_confidence:.2%}",
                'Tiempo': f"{processing_time:.3f}s"
            })
            
            # Mostrar historial
            if len(st.session_state.history) > 0:
                st.subheader("📝 Historial de Clasificaciones")
                history_df = pd.DataFrame(st.session_state.history[-10:])
                st.dataframe(history_df, use_container_width=True)
                
                if st.button("🗑️ Limpiar Historial"):
                    st.session_state.history = []
                    st.rerun()
    
    elif classify_button:
        st.warning("⚠️ Por favor, ingresa texto para clasificar.")
    
    # Estadísticas
    if st.session_state.get('history'):
        st.header("📊 Estadísticas de Uso")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clasificaciones", len(st.session_state.history))
        
        with col2:
            categories = [item['Categoría'].split()[1] for item in st.session_state.history]
            most_common = max(set(categories), key=categories.count) if categories else "N/A"
            st.metric("Categoría Más Común", most_common)
        
        with col3:
            times = [float(item['Tiempo'].replace('s', '')) for item in st.session_state.history]
            avg_time = sum(times) / len(times) if times else 0
            st.metric("Tiempo Promedio", f"{avg_time:.3f}s")
        
        with col4:
            st.metric("Accuracy del Modelo", "94.72%")
    
    # Información técnica
    st.header("🔧 Información Técnica del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Métricas Reales")
        metrics_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Valor': ['94.72%', '94.8%', '94.6%', '94.7%']
        }
        st.table(pd.DataFrame(metrics_data))
    
    with col2:
        st.subheader("⚙️ Configuración")
        config_data = {
            'Parámetro': ['Modelo', 'Fuente', 'Épocas', 'Learning Rate'],
            'Valor': ['DistilBERT', 'Hugging Face', '3', '2e-5']
        }
        st.table(pd.DataFrame(config_data))
    
    # Enlace al modelo
    st.info("🔗 **Modelo disponible en:** https://huggingface.co/gaanvalo/distilbert-finetuned-noticias")
    
    # Footer con versión
    st.markdown("---")
    st.markdown("""
    **Taller Final - Módulo 2:** Procesamiento de Datos Secuenciales con Deep Learning  
    **Estudiante:** Gabriel Antonio Vallejo Loaiza - Código: 2250145  
    **Universidad:** Autónoma de Occidente  
    **Fecha:** Agosto 2025  
    
    **Estado:** Modelo DistilBERT real activo desde Hugging Face Hub  
    **Versión:** 2.0.0 - Modelo Real Integrado
    """)

if __name__ == "__main__":
    main()
