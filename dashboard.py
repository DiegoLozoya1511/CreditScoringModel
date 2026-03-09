import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Modelo ADA - Scoring de Crédito",
    page_icon="💰",
    layout="centered"
)

# Diccionario con los pesos de las variables
PESOS = {
    'Credit_Mix_Good': 0.725155,
    'Payment_of_Min_Amount_Yes': 0.385820,
    'Credit_Mix_Standard': 0.349903,
    'High_spent': 0.282668,
    'Num_Credit_Card': 0.103531,
    'Interest_Rate': 0.042818,
    'Delay_from_due_date': 0.019019,
    'Changed_Credit_Limit': 0.015326,
    'Outstanding_Debt': 0.000081
}

# Diccionario con descripciones de las variables en español
DESCRIPCIONES = {
    'Outstanding_Debt': 'Deuda Pendiente',
    'Interest_Rate': 'Tasa de Interés',
    'Delay_from_due_date': 'Días de Retraso',
    'Num_Credit_Card': 'Número de Tarjetas de Crédito',
    'Credit_Mix_Standard': 'Mix de Crédito Estándar',
    'Changed_Credit_Limit': 'Cambio en Límite de Crédito',
    'Credit_Mix_Good': 'Mix de Crédito Bueno',
    'Payment_of_Min_Amount_Yes': 'Pago del Mínimo',
    'High_spent': 'Gasto Alto'
}

# Inicializar el estado de la sesión
if 'page' not in st.session_state:
    st.session_state.page = 'inicio'
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

def calcular_score_final(suma_ponderada):

    primer_calculo = (suma_ponderada + 4.26) * 100
    score_final = np.floor((primer_calculo / 538.79) * 600)

    return primer_calculo, score_final

def crear_semaforo_simple(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,

        number={
            "font": {"size": 42}
        },

        gauge={
            "axis": {
                "range": [0,600],
                "tickmode": "array",
                "tickvals": [0,220,410,600],
                "ticktext": ["0","220","410","600"]
            },

            "bar": {"color": "black","thickness":0.25},

            "steps": [
                {"range":[0,220], "color":"#e74c3c"},
                {"range":[220,410], "color":"#f1c40f"},
                {"range":[410,600], "color":"#27ae60"}
            ],

            "threshold": {
                "line": {"color":"black","width":6},
                "thickness":0.8,
                "value": score
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=40,r=40,t=40,b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def pagina_inicio():
    """Página de inicio del modelo ADA"""
    
    st.markdown("""
        <style>
        .title-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
            text-align: center;
        }
        .main-title {
            color: white;
            font-size: 5rem;
            font-weight: 900;
            margin: 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            letter-spacing: 5px;
        }
        .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.3rem;
            margin-top: 0.5rem;
            font-weight: 300;
            letter-spacing: 2px;
        }
        .card-container {
            background: white;
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
            margin: 2rem 0;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.3rem;
            padding: 1rem 3rem;
            border-radius: 50px;
            border: none;
            font-weight: bold;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">MODELO ADA</h1>
        <p class="subtitle">Sistema de Scoring Crediticio Clásico</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 COMENZAR", key="start"):
        st.session_state.page = 'formulario'
        st.rerun()

def pagina_formulario():
    """Página del formulario de entrada de datos"""
    
    st.markdown("""
        <style>
        .form-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .form-header h2 {
            font-size: 2.2rem;
            margin: 0;
        }
        .form-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 0.5rem 0 0 0;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: #1976d2;
            font-size: 1rem;
            border-left: 5px solid #1976d2;
        }
        .variable-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        .section-title {
            color: #2d3748;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.3rem;
            padding: 1rem;
            border-radius: 10px;
            border: none;
            font-weight: bold;
            width: 100%;
        }
        .back-button {
            background: #718096 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← VOLVER", key="back"):
            st.session_state.page = 'inicio'
            st.rerun()
    
    with col2:
        st.markdown("""
            <div class="form-header">
                <h2>📊 DATOS DEL SOLICITANTE</h2>
                <p>Complete todos los campos con la información requerida</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            ℹ️ Todos los campos aceptan valores positivos y negativos
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="variable-section">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📈 VARIABLES FINANCIERAS</p>', unsafe_allow_html=True)
            
            outstanding_debt = st.number_input(
                "💰 Deuda Pendiente",
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="debt"
            )
            
            interest_rate = st.number_input(
                "📊 Tasa de Interés (%)",
                value=0.0,
                step=0.5,
                format="%.2f",
                key="rate"
            )
            
            delay_days = st.number_input(
                "⏰ Días de Retraso",
                value=0,
                step=1,
                key="delay"
            )
            
            num_cards = st.number_input(
                "💳 Número de Tarjetas",
                value=0,
                step=1,
                key="cards"
            )
            
            high_spent = st.number_input(
                "💸 Gasto Alto",
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="spent"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="variable-section">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📋 VARIABLES DE COMPORTAMIENTO</p>', unsafe_allow_html=True)
            
            credit_mix_std = st.number_input(
                "🔄 Mix Crédito Estándar",
                value=0.0,
                step=0.1,
                format="%.2f",
                key="mix_std"
            )
            
            changed_limit = st.number_input(
                "📈 Cambio en Límite",
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="limit_change"
            )
            
            credit_mix_good = st.number_input(
                "✨ Mix Crédito Bueno",
                value=0.0,
                step=0.1,
                format="%.2f",
                key="mix_good"
            )
            
            payment_min = st.number_input(
                "💵 Pago Mínimo",
                value=0.0,
                step=0.1,
                format="%.2f",
                key="payment"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("📊 CALCULAR SCORE", key="calculate"):
        st.session_state.user_inputs = {
            'Outstanding_Debt': outstanding_debt,
            'Interest_Rate': interest_rate,
            'Delay_from_due_date': delay_days,
            'Num_Credit_Card': num_cards,
            'Credit_Mix_Standard': credit_mix_std,
            'Changed_Credit_Limit': changed_limit,
            'Credit_Mix_Good': credit_mix_good,
            'Payment_of_Min_Amount_Yes': payment_min,
            'High_spent': high_spent
        }
        st.session_state.page = 'resultado'
        st.rerun()

def pagina_resultado():
    """Página de resultados con semáforo simple"""
    
    st.markdown("""
        <style>
        .result-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .result-header h2 {
            font-size: 2.2rem;
            margin: 0;
        }
        .result-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 0.5rem 0 0 0;
        }
        .calc-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        .calc-title {
            color: #2d3748;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .calc-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.2rem;
            padding: 0.8rem;
            border-radius: 10px;
            border: none;
            font-weight: bold;
            width: 100%;
        }
        .semaforo-container {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Calcular la suma ponderada
    suma_ponderada = 0
    for var_name, valor in st.session_state.user_inputs.items():
        if var_name in PESOS:
            suma_ponderada += valor * PESOS[var_name]
    
    # Calcular scores
    primer_calculo, score_final = calcular_score_final(suma_ponderada)
    
    # Determinar color y texto según el score
    if score_final < 220:
        color_score = "#FF4B4B"
        nivel_riesgo = "ALTO"
        emoji = "🔴"
    elif score_final < 410:
        color_score = "#FFD700"
        nivel_riesgo = "MEDIO"
        emoji = "🟡"
    else:
        color_score = "#4CAF50"
        nivel_riesgo = "BAJO"
        emoji = "🟢"
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← NUEVA EVALUACIÓN", key="new"):
            st.session_state.page = 'formulario'
            st.rerun()
    
    with col2:
        st.markdown("""
            <div class="result-header">
                <h2>📊 RESULTADO DEL SCORING</h2>
                <p>Evaluación crediticia completada</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Score y nivel de riesgo ARRIBA
    st.markdown(f"""
    <div style="text-align: center; margin: 0.1rem 0;">
        <span style="font-size: 3rem; font-weight: 900; color: {color_score};">{score_final:.0f}</span>
        <span style="font-size: 2rem; margin-left: 1rem;">{emoji}</span>
        <div style="font-size: 1.5rem; color: #666; margin-top: 0.5rem;">
            Nivel de Riesgo: <span style="color: {color_score}; font-weight: bold;">{nivel_riesgo}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gráfica abajo
    st.markdown('<div class="semaforo-container">', unsafe_allow_html=True)
    fig = crear_semaforo_simple(score_final)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cálculos en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="calc-box">
                <div class="calc-title">📝 SUMA PONDERADA</div>
                <div class="calc-value">{suma_ponderada:.4f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="calc-box">
                <div class="calc-title">📊 PRIMER CÁLCULO</div>
                <div class="calc-value">{primer_calculo:.2f}</div>
                <div style="font-size:0.9rem; color:#666;">({suma_ponderada:.4f} + 4.26) × 100</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="calc-box">
                <div class="calc-title">🎯 SCORE FINAL</div>
                <div class="calc-value" style="color:{color_score};">{score_final:.0f}</div>
                <div style="font-size:0.9rem; color:#666;">⌊({primer_calculo:.2f}/538.79)×600⌋</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Decisión crediticia
    st.markdown("### 📋 DECISIÓN CREDITICIA")
    
    if score_final < 220:
        st.error(f"""
            ### 🔴 RIESGO ALTO
            - **Score:** {score_final:.0f} puntos
            - **Decisión:** Rechazar solicitud
            - **Motivo:** Alto riesgo de incumplimiento
            - **Recomendación:** No otorgar crédito
        """)
    elif score_final < 410:
        st.warning(f"""
            ### 🟡 RIESGO MEDIO
            - **Score:** {score_final:.0f} puntos
            - **Decisión:** Evaluar con detenimiento
            - **Motivo:** Riesgo moderado
            - **Recomendación:** Solicitar garantías adicionales
        """)
    else:
        st.success(f"""
            ### 🟢 RIESGO BAJO
            - **Score:** {score_final:.0f} puntos
            - **Decisión:** Aprobar solicitud
            - **Motivo:** Excelente perfil crediticio
            - **Recomendación:** Aprobación automática
        """)

# Control de navegación
if st.session_state.page == 'inicio':
    pagina_inicio()
elif st.session_state.page == 'formulario':
    pagina_formulario()
elif st.session_state.page == 'resultado':
    pagina_resultado()