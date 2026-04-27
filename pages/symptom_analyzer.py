"""Symptom Pattern Analyzer Page"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def show():
    """Display symptom pattern analyzer page"""
    
    st.markdown("## 🔬 Symptom Pattern Analyzer")
    st.markdown("Compare symptom patterns between two diseases")
    st.markdown("---")
    
    df = st.session_state.df
    symptom_cols = [c for c in df.columns if c != 'diseases']
    disease_options = sorted(df['diseases'].unique().tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease1 = st.selectbox("**Disease A**", disease_options, index=0)
    
    with col2:
        disease2 = st.selectbox("**Disease B**", disease_options, index=min(1, len(disease_options)-1))
    
    if st.button("🔍 COMPARE SYMPTOMS", type="primary", use_container_width=True):
        if disease1 == disease2:
            st.warning("⚠️ Please select two different diseases")
            return
        
        with st.spinner("Analyzing symptom patterns..."):
            # Get data
            data1 = df[df['diseases'] == disease1]
            data2 = df[df['diseases'] == disease2]
            
            # Calculate frequencies
            freq1 = data1[symptom_cols].sum() / len(data1)
            freq2 = data2[symptom_cols].sum() / len(data2)
            
            # Create comparison dataframe
            diff_df = pd.DataFrame({
                'symptom': symptom_cols,
                'disease1_freq': freq1.values,
                'disease2_freq': freq2.values,
                'difference': freq1.values - freq2.values,
                'abs_diff': np.abs(freq1.values - freq2.values)
            }).sort_values('abs_diff', ascending=False)
            
            # Header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3>{disease1.upper()} vs {disease2.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 20 table
            st.markdown("### 🔍 Top 20 Distinguishing Symptoms")
            
            display_df = diff_df.head(20).copy()
            display_df['disease1_freq'] = display_df['disease1_freq'].apply(lambda x: f"{x:.1%}")
            display_df['disease2_freq'] = display_df['disease2_freq'].apply(lambda x: f"{x:.1%}")
            display_df['difference'] = display_df['difference'].apply(lambda x: f"{x:+.1%}")
            display_df = display_df[['symptom', 'disease1_freq', 'disease2_freq', 'difference']]
            display_df.columns = ['Symptom', disease1[:25], disease2[:25], 'Difference']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📈 Top Symptoms in {disease1[:20]}")
                top1 = diff_df.nlargest(10, 'disease1_freq')
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(len(top1)), top1['disease1_freq'].values, color='#2ecc71')
                ax.set_yticks(range(len(top1)))
                ax.set_yticklabels([s[:30] for s in top1['symptom'].values])
                ax.set_xlabel('Prevalence')
                ax.set_xlim(0, 1)
                for i, val in enumerate(top1['disease1_freq'].values):
                    ax.text(val + 0.02, i, f'{val:.1%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown(f"### 📉 Top Symptoms in {disease2[:20]}")
                top2 = diff_df.nlargest(10, 'disease2_freq')
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(len(top2)), top2['disease2_freq'].values, color='#e74c3c')
                ax.set_yticks(range(len(top2)))
                ax.set_yticklabels([s[:30] for s in top2['symptom'].values])
                ax.set_xlabel('Prevalence')
                ax.set_xlim(0, 1)
                for i, val in enumerate(top2['disease2_freq'].values):
                    ax.text(val + 0.02, i, f'{val:.1%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Heatmap
            st.markdown("### 🗺️ Symptom Comparison Heatmap")
            
            heatmap_data = diff_df.head(15).set_index('symptom')[['disease1_freq', 'disease2_freq']]
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=[disease1[:20], disease2[:20]],
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(heatmap_data.values.T * 100, 1),
                texttemplate='%{text}%'
            ))
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Symptoms", len(symptom_cols))
            col2.metric(f"{disease1[:15]} Patients", len(data1))
            col3.metric(f"{disease2[:15]} Patients", len(data2))
            col4.metric("Avg Difference", f"{diff_df['abs_diff'].mean():.1%}")
            
            # Download
            st.download_button(
                "📥 Download Results (CSV)",
                diff_df.to_csv(index=False),
                f"comparison_{disease1}_{disease2}.csv",
                "text/csv"
            )