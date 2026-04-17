import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for confidence scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "#27ae60"},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 75], 'color': "#f39c12"},
                {'range': [75, 100], 'color': "#27ae60"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_symptom_radar(symptom_data, title="Symptom Profile"):
    """Create radar chart for symptom patterns"""
    categories = list(symptom_data.keys())
    values = list(symptom_data.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='rgba(39, 174, 96, 0.8)'),
        line=dict(color='rgba(39, 174, 96, 1)', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        height=400
    )
    return fig

def display_metric_cards(col1, col2, col3, col4, metrics):
    """Display metric cards in columns"""
    with col1:
        st.metric(metrics[0]['label'], metrics[0]['value'], metrics[0].get('delta'))
    with col2:
        st.metric(metrics[1]['label'], metrics[1]['value'], metrics[1].get('delta'))
    with col3:
        st.metric(metrics[2]['label'], metrics[2]['value'], metrics[2].get('delta'))
    with col4:
        st.metric(metrics[3]['label'], metrics[3]['value'], metrics[3].get('delta'))

def add_logo():
    """Add logo to sidebar"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #2c3e50;">🏥 Disease Predictor</h2>
        <p style="color: #7f8c8d;">AI-Powered Medical Diagnosis Assistant</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    """Add footer to app"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only. 
        Always consult with a qualified healthcare provider for medical advice and diagnosis.</p>
        <p>© 2024 Disease Prediction System | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)