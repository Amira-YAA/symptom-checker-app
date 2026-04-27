"""Utility functions"""

import streamlit as st
import pandas as pd

def display_metric_card(title, value, delta=None, color="#667eea"):
    """Display a styled metric card"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%);
                padding: 15px; border-radius: 10px; color: white; text-align: center;">
        <h4 style="margin: 0;">{title}</h4>
        <h2 style="margin: 5px 0;">{value}</h2>
        {f'<p style="margin: 0;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(value, label, color="#27ae60"):
    """Create a custom progress bar"""
    return f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>{label}</span>
            <span>{value:.1%}</span>
        </div>
        <div style="background-color: #ecf0f1; border-radius: 5px; overflow: hidden;">
            <div style="background-color: {color}; width: {value*100}%; height: 25px; border-radius: 5px;"></div>
        </div>
    </div>
    """

def reset_session():
    """Reset session state"""
    for key in list(st.session_state.keys()):
        if key not in ['df', 'model', 'features', 'disease_encoder', 'model_trained', 'model_accuracy']:
            del st.session_state[key]