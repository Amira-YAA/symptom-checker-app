"""Symptom categorization based on medical domains"""

SYMPTOM_CATEGORIES = {
    "🧠 Mental & Emotional": [
        'anxiety and nervousness', 'depression', 'depressive or psychotic symptoms',
        'insomnia', 'irritability', 'mood swings', 'panic attack', 'paranoia',
        'personality changes', 'psychosis', 'suicidal thoughts', 'hallucinations', 'mania'
    ],
    
    "❤️ Cardiovascular": [
        'sharp chest pain', 'chest tightness', 'palpitations', 'high blood pressure',
        'low blood pressure', 'irregular heartbeat', 'swelling in legs', 'chest pressure',
        'chest discomfort', 'rapid heart rate', 'slow heart rate'
    ],
    
    "🫁 Respiratory": [
        'shortness of breath', 'wheezing', 'cough', 'productive cough', 'dry cough',
        'difficulty breathing at night', 'rapid breathing', 'shallow breathing',
        'coughing up blood', 'runny nose', 'nasal congestion', 'sneezing', 'sore throat'
    ],
    
    "🍽️ Digestive": [
        'nausea', 'vomiting', 'abdominal pain', 'diarrhea', 'constipation',
        'bloating', 'heartburn', 'loss of appetite', 'blood in stool',
        'jaundice', 'difficulty swallowing', 'excessive thirst', 'metallic taste',
        'stomach cramps', 'indigestion'
    ],
    
    "🧠 Neurological": [
        'dizziness', 'headache', 'migraine', 'seizures', 'tremors',
        'memory loss', 'confusion', 'numbness', 'tingling sensation',
        'loss of coordination', 'balance problems', 'fainting'
    ],
    
    "🚽 Genitourinary": [
        'frequent urination', 'painful urination', 'blood in urine',
        'urinary incontinence', 'pelvic pain', 'vaginal discharge',
        'testicular pain', 'erectile dysfunction', 'kidney pain',
        'burning sensation during urination', 'cloudy urine', 'urgency to urinate'
    ],
    
    "🦴 Musculoskeletal": [
        'joint pain', 'muscle pain', 'back pain', 'neck pain', 'arthritis',
        'muscle weakness', 'stiffness', 'bone pain', 'muscle cramps',
        'limited range of motion', 'swollen joints', 'morning stiffness',
        'sciatica', 'shoulder pain', 'knee pain'
    ],
    
    "🩻 Skin & Appearance": [
        'rash', 'itching', 'hives', 'dry skin', 'skin discoloration',
        'hair loss', 'nail changes', 'bruising', 'swelling', 'redness',
        'blisters', 'skin lesions', 'warts'
    ],
    
    "👁️ Eye & Vision": [
        'blurred vision', 'double vision', 'eye pain', 'red eyes',
        'sensitivity to light', 'dry eyes', 'floaters', 'vision loss',
        'eye discharge', 'itchy eyes'
    ],
    
    "🦻 Ear, Nose & Throat": [
        'ear pain', 'ringing in ears', 'hearing loss', 'sore throat',
        'hoarseness', 'swollen lymph nodes', 'sinus pain', 'loss of smell',
        'loss of taste', 'runny nose', 'nasal congestion'
    ],
    
    "🩸 Systemic & General": [
        'fever', 'fatigue', 'weight loss', 'weight gain', 'night sweats',
        'chills', 'general weakness', 'loss of appetite', 'malaise',
        'swollen lymph nodes', 'anemia symptoms'
    ],
    
    "👶 Pregnancy & Reproductive": [
        'morning sickness', 'breast tenderness', 'missed period',
        'spotting', 'pelvic pressure', 'back pain during pregnancy',
        'contractions', 'vaginal bleeding', 'amniotic fluid leakage'
    ]
}

def get_all_symptoms():
    """Get all symptoms from categories"""
    all_symptoms = []
    for symptoms in SYMPTOM_CATEGORIES.values():
        all_symptoms.extend(symptoms)
    return list(set(all_symptoms))

def get_category_for_symptom(symptom):
    """Find which category a symptom belongs to"""
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        if symptom in symptoms:
            return category
    return "📌 Other Symptoms"