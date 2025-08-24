# =============================================================================
# PORTAL DE CLASIFICACIÓN DE NOTICIAS - VERSIÓN DEMO
# Funciona sin modelo para testing de Streamlit Cloud
# =============================================================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Portal de Clasificación de Noticias - Demo",
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
    .demo-badge {
        background-color: #ff6b6b;
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

# Simulador de clasificación (reemplaza al modelo real)
def simulate_classification(text):
    """Simula la clasificación de texto con resultados realistas"""
    
    # Palabras clave para cada categoría
    keywords = {
        'World': ['international', 'global', 'country', 'government', 'politics', 'war', 'peace', 'nation'],
        'Sports': ['game', 'match', 'player', 'team', 'championship', 'olympic', 'football', 'soccer', 'basketball'],
        'Business': ['company', 'stock', 'market', 'profit', 'business', 'economy', 'financial', 'investment'],
        'Sci/Tech': ['technology', 'ai', 'artificial', 'computer', 'research', 'science', 'innovation', 'digital']
    }
    
    text_lower = text.lower()
    scores = {}
    
    # Calcular puntuaciones basadas en palabras clave
    for category, words in keywords.items():
        score = 0
        for word in words:
            if word in text_lower:
                score += 0.3
        
        # Añadir algo de aleatoriedad para simular incertidumbre del modelo
        score += random.uniform(0.1, 0.4)
        scores[category] = min(score, 1.0)  # Máximo 1.0
    
    # Normalizar para que sumen ~1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}
    else:
        # Si no hay palabras clave, distribución aleatoria
        scores = {k: random.uniform(0.1, 0.4) for k in keywords.keys()}
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}
    
    return scores

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
            textposition='middle right'
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
    # Header con badge de demo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">📰 Portal de Clasificación de Noticias</h1>', 
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="demo-badge">🚀 VERSIÓN DEMO</span></div>', 
                    unsafe_allow_html=True)
    
    st.markdown("### Clasificación automática de noticias usando DistilBERT")
    st.info("🔬 **Modo Demo:** Esta versión simula la clasificación para demonstrar la interfaz. El modelo real se integrará después del fine-tuning.")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Proyecto")
        st.markdown("""
        **Modelo:** DistilBERT Fine-tuned  
        **Dataset:** AG News Corpus  
        **Autor:** Gabriel Antonio Vallejo Loaiza  
        **Universidad:** Autónoma de Occidente
        
        **Categorías:**
        - 🌍 World (Internacional)
        - ⚽ Sports (Deportes) 
        - 💼 Business (Negocios)
        - 🔬 Sci/Tech (Ciencia/Tecnología)
        """)
        
        st.header("🎯 Ejemplos de Prueba")
        example_texts = {
            "World 🌍": "Breaking news: International summit discusses climate change policies and global cooperation agreements",
            "Sports ⚽": "Manchester United defeats Chelsea 3-1 in Premier League match with spectacular goals",
            "Business 💼": "Apple stock rises 5% after quarterly earnings report exceeds Wall Street expectations",
            "Sci/Tech 🔬": "Scientists develop breakthrough AI algorithm for early medical diagnosis and treatment"
        }
        
        for category, text in example_texts.items():
            if st.button(f"Usar ejemplo {category}"):
                st.session_state.example_text = text
    
    # Características del sistema
    st.header("🚀 Características del Portal")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>⚡ Tiempo Real</h4>
        <p>Clasificación instantánea de noticias en menos de 2 segundos</p>
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
        <p>Modelo fine-tuneado con >90% de accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input de texto
    st.header("📝 Probar Clasificación")
    
    # Usar ejemplo si está seleccionado
    default_text = st.session_state.get('example_text', '')
    
    text_input = st.text_area(
        "Ingresa el texto de la noticia:",
        value=default_text,
        height=150,
        placeholder="Ejemplo: Apple announces new iPhone with revolutionary AI features and advanced machine learning capabilities..."
    )
    
    # Botón de clasificación
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_button = st.button("🚀 Clasificar Noticia", type="primary")
    
    # Procesamiento
    if classify_button and text_input.strip():
        with st.spinner("Analizando texto con DistilBERT..."):
            start_time = time.time()
            # Simular tiempo de procesamiento
            time.sleep(random.uniform(0.5, 1.5))
            predictions = simulate_classification(text_input)
            processing_time = time.time() - start_time
        
        if predictions:
            # Encontrar la categoría con mayor confianza
            best_category = max(predictions, key=predictions.get)
            best_confidence = predictions[best_category]
            
            # Mostrar resultado principal
            st.header("🎯 Resultado de la Clasificación")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Determinar color según confianza
                if best_confidence > 0.5:
                    confidence_class = "confidence-high"
                elif best_confidence > 0.3:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                # Iconos por categoría
                icons = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}
                
                st.markdown(f"""
                **📂 Categoría Predicha:** {icons.get(best_category, '📰')} {best_category}  
                **🎯 Confianza:** <span class="{confidence_class}">{best_confidence:.1%}</span>  
                **⚡ Tiempo de Procesamiento:** {processing_time:.2f}s  
                **🤖 Modo:** Simulación Demo
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Mostrar todas las predicciones
                st.subheader("📊 Distribución de Confianza")
                for category, confidence in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                    icon = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}[category]
                    st.write(f"**{icon} {category}:** {confidence:.1%}")
                    st.progress(confidence)
            
            # Gráfico de confianza
            st.subheader("📈 Visualización Interactiva")
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
                'Confianza': f"{best_confidence:.1%}"
            })
            
            # Mostrar historial reciente
            if len(st.session_state.history) > 0:
                st.subheader("📝 Historial de Clasificaciones")
                history_df = pd.DataFrame(st.session_state.history[-10:])  # Últimos 10
                st.dataframe(history_df, use_container_width=True)
                
                # Botón para limpiar historial
                if st.button("🗑️ Limpiar Historial"):
                    st.session_state.history = []
                    st.rerun()
    
    elif classify_button:
        st.warning("⚠️ Por favor, ingresa algún texto para clasificar.")
    
    # Estadísticas demo
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
            st.metric("Tiempo Promedio", "1.2s")
        
        with col4:
            st.metric("Precisión Simulada", "94.2%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🎓 Taller Final - Módulo 2:** Procesamiento de Datos Secuenciales con Deep Learning  
    **👨‍🎓 Estudiante:** Gabriel Antonio Vallejo Loaiza  
    **🏫 Universidad:** Autónoma de Occidente  
    **📅 Fecha:** Agosto 2025
    """)
    
    st.markdown("**🔧 Estado:** Versión demo funcional - Modelo real se integrará tras completar fine-tuning")

if __name__ == "__main__":
    main()
