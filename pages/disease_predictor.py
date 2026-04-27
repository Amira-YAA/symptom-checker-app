"""Disease Predictor Page"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.symptom_categories import SYMPTOM_CATEGORIES, get_symptom_category, get_all_symptoms
from src.model_trainer import predict_disease

def show():
    """Display disease predictor page"""
    
    st.markdown("*Select your symptoms below to get a disease prediction*")
    st.markdown("---")
    
    # Check if model is trained
    if not st.session_state.get('model_trained', False):
        st.info("🤖 Please wait for model to load...")
        return
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.reset_trigger = st.session_state.get('reset_trigger', 0) + 1
            st.rerun()
    
    with col2:
        selected_count = len(st.session_state.get('selected_symptoms', []))
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 10px; border-radius: 10px; text-align: center; color: white;">
            ✅ SELECTED: {selected_count} symptoms
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("📋 Clear All", use_container_width=True):
            st.session_state.reset_trigger = st.session_state.get('reset_trigger', 0) + 1
            st.rerun()
    
    st.markdown("---")
    
    # Get available symptoms
    available_symptoms = set(st.session_state.df.columns)
    reset_key = f"reset_{st.session_state.get('reset_trigger', 0)}"
    selected_symptoms = []
    
    # Category order
    category_order = [
        "🧠 Mental & Emotional", "❤️ Cardiovascular", "🫁 Respiratory",
        "🍽️ Digestive", "🧠 Neurological", "🚽 Genitourinary",
        "🦴 Musculoskeletal", "🩻 Skin & Appearance", "👁️ Eye & Vision",
        "🦻 Ear, Nose & Throat", "🩸 Systemic & General", "👶 Pregnancy & Reproductive"
    ]
    
    # Two columns for categories
    col1, col2 = st.columns(2)
    mid = len(category_order) // 2
    
    for idx, category in enumerate(category_order):
        with col1 if idx < mid else col2:
            if category in SYMPTOM_CATEGORIES:
                available = [s for s in SYMPTOM_CATEGORIES[category] if s in available_symptoms]
                if available:
                    with st.expander(f"{category} ({len(available)})", expanded=False):
                        cols = st.columns(2)
                        for i, symptom in enumerate(available):
                            name = symptom.replace('_', ' ').title()
                            key = f"{category}_{symptom}_{reset_key}"
                            if cols[i % 2].checkbox(name, key=key):
                                selected_symptoms.append(symptom)
    
    # Other symptoms
    all_categorized = set()
    for symptoms in SYMPTOM_CATEGORIES.values():
        all_categorized.update(symptoms)
    
    other = [s for s in available_symptoms if s != 'diseases' and s not in all_categorized]
    
    if other:
        with st.expander(f"📌 Other Symptoms ({len(other)})", expanded=False):
            search = st.text_input("🔍 Search", key=f"search_{reset_key}")
            filtered = [s for s in other if search.lower() in s.lower()] if search else other[:50]
            cols = st.columns(3)
            for i, symptom in enumerate(filtered):
                name = symptom.replace('_', ' ').title()
                key = f"other_{symptom}_{reset_key}"
                if cols[i % 3].checkbox(name, key=key):
                    selected_symptoms.append(symptom)
    
    # Store selected
    st.session_state.selected_symptoms = selected_symptoms
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 PREDICT DISEASE", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            with st.spinner("🔄 Analyzing symptoms with AI..."):
                result = predict_disease(
                    selected_symptoms,
                    st.session_state.model,
                    st.session_state.features,
                    st.session_state.disease_encoder
                )
                
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    conf = result['primary_confidence']
                    if conf > 0.7:
                        color = "#27ae60"
                        level = "HIGH"
                    elif conf > 0.4:
                        color = "#f39c12"
                        level = "MEDIUM"
                    else:
                        color = "#e74c3c"
                        level = "LOW"
                    
                    st.markdown(f"""
                    <div style="background: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h2>Most Likely Disease</h2>
                        <h1>{result['primary'].upper()}</h1>
                        <h3>{level} CONFIDENCE: {conf*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Top 7 Possible Diseases")
                    
                    for i, (disease, prob) in enumerate(result['top_7'], 1):
                        if i == 1:
                            bar_color = "#27ae60"
                            icon = "🏆"
                        elif i <= 3:
                            bar_color = "#3498db"
                            icon = "⭐"
                        elif i <= 5:
                            bar_color = "#f39c12"
                            icon = "📌"
                        else:
                            bar_color = "#95a5a6"
                            icon = "🔹"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <b>{i}. {icon} {disease}</b>
                                <span style="color: {bar_color};">{prob*100:.1f}%</span>
                            </div>
                            <div style="background: #ecf0f1; border-radius: 5px;">
                                <div style="background: {bar_color}; width: {prob*100}%; height: 25px; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("🔄 New Prediction"):
                        st.rerun()
                
                with col2:
                    st.markdown(f"### ✅ Selected Symptoms ({len(selected_symptoms)})")
                    by_cat = {}
                    for s in selected_symptoms:
                        cat = get_symptom_category(s)
                        by_cat.setdefault(cat, []).append(s)
                    
                    for cat, syms in by_cat.items():
                        with st.expander(f"{cat} ({len(syms)})"):
                            for s in syms[:15]:
                                st.markdown(f"- {s.replace('_', ' ').title()}")
                
                # Chart
                st.markdown("### 📈 Probability Distribution")
                
                top10_idx = np.argsort(result['all_probabilities'])[-10:][::-1]
                top10_diseases = result['all_diseases'][top10_idx]
                top10_probs = result['all_probabilities'][top10_idx]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c',
                         '#9b59b6', '#1abc9c', '#34495e', '#95a5a6', '#7f8c8d']
                
                ax.barh(range(len(top10_diseases)), top10_probs, color=colors[:len(top10_diseases)])
                ax.set_yticks(range(len(top10_diseases)))
                ax.set_yticklabels([d[:30] for d in top10_diseases])
                ax.set_xlabel('Probability')
                ax.set_xlim(0, 1)
                
                for i, prob in enumerate(top10_probs):
                    ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.caption("⚠️ **Medical Disclaimer:** For educational purposes only. Always consult a healthcare provider.")