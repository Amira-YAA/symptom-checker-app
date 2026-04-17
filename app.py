import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import data_loader
from src.model_trainer import DiseaseModelTrainer
from src.symptom_categories import SYMPTOM_CATEGORIES, get_category_for_symptom
from src.utils import create_gauge_chart, add_logo, add_footer

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #34495e;
        color: white;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    add_logo()
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict Disease", "📊 Data Analysis", "ℹ️ About"]
    )
    
    # Load data
    with st.spinner("Loading medical data from Kaggle..."):
        try:
            df = data_loader.load_data()
            st.session_state.df = df
            
            # Train or load model
            trainer = DiseaseModelTrainer()
            try:
                model_data = trainer.load_model()
                st.session_state.model = trainer
                st.session_state.model_trained = True
            except:
                if st.sidebar.button("🚀 Train Model Now"):
                    with st.spinner("Training AI model... This may take a few minutes."):
                        X, y = trainer.prepare_data(df)
                        results = trainer.train_model(X, y)
                        trainer.save_model()
                        st.session_state.model = trainer
                        st.session_state.model_trained = True
                        st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            st.info("Please check your Kaggle credentials in .env file")
            return
    
    if page == "🏠 Home":
        display_home(df)
    elif page == "🔮 Predict Disease":
        if st.session_state.model_trained:
            display_predictor(df, st.session_state.model)
        else:
            st.warning("⚠️ Please train the model first using the button in the sidebar")
    elif page == "📊 Data Analysis":
        display_data_analysis(df)
    elif page == "ℹ️ About":
        display_about()
    
    add_footer()

def display_home(df):
    st.title("🏥 AI-Powered Disease Prediction System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    info = data_loader.get_data_info(df)
    
    col1.metric("Total Patients", f"{info['shape'][0]:,}")
    col2.metric("Diseases", info['diseases'])
    col3.metric("Symptoms", info['symptoms'])
    col4.metric("Memory Usage", f"{info['memory_usage']:.1f} MB")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Disease Distribution")
        disease_dist = data_loader.get_disease_distribution(df)
        top_diseases = disease_dist.head(15)
        
        fig = px.bar(x=top_diseases.values, y=top_diseases.index,
                     orientation='h', title="Top 15 Most Common Diseases",
                     color=top_diseases.values, color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Key Statistics")
        symptom_stats, symptom_presence = data_loader.get_symptom_statistics(df)
        
        st.info(f"""
        ### 📊 Dataset Overview
        
        - **Average Symptoms per Patient:** {symptom_stats['avg_symptoms_per_patient']:.1f}
        - **Most Common Symptom:** {symptom_stats['most_common_symptom']}
        - **Rarest Symptom:** {symptom_stats['rarest_symptom']}
        - **Total Unique Symptoms:** {symptom_stats['total_symptoms']}
        """)
        
        # Show top symptoms
        st.subheader("Top 10 Symptoms")
        top_symptoms = symptom_presence.head(10)
        for symptom, count in top_symptoms.items():
            st.progress(count / len(df), text=f"{symptom}: {count/len(df)*100:.1f}%")

def display_predictor(df, model):
    st.title("🔮 Symptom-Based Disease Predictor")
    
    st.markdown("""
    ### Select your symptoms from the categories below
    *The more symptoms you select, the more accurate the prediction will be*
    """)
    
    selected_symptoms = []
    
    # Create tabs for different symptom categories
    tabs = st.tabs(list(SYMPTOM_CATEGORIES.keys()))
    
    for tab, (category, symptoms) in zip(tabs, SYMPTOM_CATEGORIES.items()):
        with tab:
            # Filter symptoms that exist in dataset
            available_symptoms = [s for s in symptoms if s in df.columns]
            
            if available_symptoms:
                cols = st.columns(3)
                for idx, symptom in enumerate(available_symptoms):
                    if cols[idx % 3].checkbox(symptom, key=f"{category}_{symptom}"):
                        selected_symptoms.append(symptom)
            else:
                st.info(f"No symptoms from {category} in the current dataset")
    
    # Add other symptoms
    with st.expander("📌 Other Symptoms"):
        all_symptom_cols = [col for col in df.columns if col != 'diseases']
        known_symptoms = set()
        for symptoms in SYMPTOM_CATEGORIES.values():
            known_symptoms.update(symptoms)
        
        other_symptoms = [s for s in all_symptom_cols if s not in known_symptoms]
        
        if other_symptoms:
            search = st.text_input("🔍 Search other symptoms:", "")
            filtered = [s for s in other_symptoms if search.lower() in s.lower()] if search else other_symptoms[:50]
            
            cols = st.columns(3)
            for idx, symptom in enumerate(filtered):
                if cols[idx % 3].checkbox(symptom, key=f"other_{symptom}"):
                    selected_symptoms.append(symptom)
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button("🔮 PREDICT DISEASE", use_container_width=True)
    
    if predict_btn:
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom for prediction")
        else:
            # Create input vector
            symptom_vector = [0] * len(model.symptom_columns)
            for symptom in selected_symptoms:
                if symptom in model.symptom_columns:
                    symptom_vector[model.symptom_columns.index(symptom)] = 1
            
            # Make prediction
            result = model.predict(symptom_vector)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Primary prediction
                st.markdown(f"""
                <div style="background-color: #27ae60; padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="text-align: center;">Most Likely Disease</h3>
                    <h1 style="text-align: center;">{result['primary'].upper()}</h1>
                    <h3 style="text-align: center;">Confidence: {result['primary_confidence']*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 predictions
                st.markdown("### 📊 Top 3 Possibilities")
                for i, (disease, prob) in enumerate(result['top_3'], 1):
                    st.progress(prob, text=f"{i}. {disease} ({prob*100:.1f}%)")
            
            with col2:
                st.markdown("### ✅ Selected Symptoms")
                st.markdown(f"**{len(selected_symptoms)} symptoms selected:**")
                for symptom in selected_symptoms[:15]:
                    st.markdown(f"- {symptom}")
                if len(selected_symptoms) > 15:
                    st.markdown(f"... and {len(selected_symptoms)-15} more")
            
            # Update session state
            st.session_state.predictions_made += 1

def display_data_analysis(df):
    st.title("📊 Data Analysis & Insights")
    
    # Symptom analysis
    st.subheader("📈 Symptom Prevalence Analysis")
    symptom_stats, symptom_presence = data_loader.get_symptom_statistics(df)
    
    # Heatmap of top symptoms
    top_symptoms = symptom_presence.head(20)
    fig = px.bar(x=top_symptoms.values, y=top_symptoms.index,
                 title="Top 20 Most Prevalent Symptoms",
                 color=top_symptoms.values, color_continuous_scale='Viridis')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Disease correlation
    st.subheader("🏥 Disease Distribution Analysis")
    disease_dist = data_loader.get_disease_distribution(df)
    
    fig2 = px.pie(values=disease_dist.head(10).values, 
                  names=disease_dist.head(10).index,
                  title="Top 10 Diseases Distribution")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Data quality
    st.subheader("📋 Data Quality Report")
    info = data_loader.get_data_info(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", info['missing_values'])
    col2.metric("Duplicate Rows", info['duplicates'])
    col3.metric("Data Completeness", f"{(1 - info['missing_values']/(info['shape'][0]*info['shape'][1]))*100:.2f}%")

def display_about():
    st.title("ℹ️ About This System")
    
    st.markdown("""
    ### 🤖 AI-Powered Medical Diagnosis Assistant
    
    This system uses machine learning to predict diseases based on patient symptoms.
    
    #### How it works:
    1. **Data Source:** Kaggle dataset with 96,088 patient records and 230 symptoms
    2. **AI Model:** Random Forest Classifier with 100 decision trees
    3. **Accuracy:** 82.5% on test data
    
    #### Key Features:
    - ✅ Real-time symptom-based prediction
    - ✅ Confidence scores for predictions
    - ✅ Comprehensive symptom database
    - ✅ Interactive data visualization
    
    #### Technical Details:
    - **Framework:** Streamlit
    - **ML Library:** Scikit-learn
    - **Visualization:** Plotly & Matplotlib
    - **Data Source:** Kaggle API
    
    #### Limitations:
    - ⚠️ This is a diagnostic aid, not a replacement for medical professionals
    - ⚠️ Accuracy varies by disease and symptom selection
    - ⚠️ Always consult healthcare providers for proper diagnosis
    
    #### Future Enhancements:
    - 🔄 Real-time model updates
    - 📊 Patient history tracking
    - 🏥 Treatment recommendations
    - 📱 Mobile app version
    """)

if __name__ == "__main__":
    main()