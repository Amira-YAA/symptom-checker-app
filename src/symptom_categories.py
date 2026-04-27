"""Symptom categories for disease prediction"""

SYMPTOM_CATEGORIES = {
    "🧠 Mental & Emotional": [
        "anxiety and nervousness", "depression", "restlessness", "excessive anger",
        "fears and phobias", "low self-esteem", "obsessions and compulsions",
        "hostile behavior", "antisocial behavior", "hysterical behavior",
        "temper problems", "insomnia", "sleepiness"
    ],
    "❤️ Cardiovascular": [
        "shortness of breath", "chest tightness", "palpitations", "irregular heartbeat",
        "sharp chest pain", "increased heart rate", "decreased heart rate",
        "chest pain", "peripheral edema"
    ],
    "🫁 Respiratory": [
        "cough", "wheezing", "difficulty breathing", "coughing up sputum",
        "hemoptysis", "congestion in chest", "abnormal breathing sounds",
        "nasal congestion", "sore throat", "hoarse voice", "sinus congestion"
    ],
    "🍽️ Digestive": [
        "nausea", "vomiting", "diarrhea", "abdominal pain", "heartburn",
        "constipation", "blood in stool", "upper abdominal pain",
        "stomach bloating", "changes in stool appearance", "melena",
        "difficulty in swallowing", "regurgitation", "burning abdominal pain"
    ],
    "🧠 Neurological": [
        "headache", "dizziness", "seizures", "loss of sensation",
        "paresthesia", "focal weakness", "problems with movement",
        "tremors", "memory disturbance", "delusions or hallucinations"
    ],
    "🚽 Genitourinary": [
        "painful urination", "frequent urination", "blood in urine",
        "vaginal discharge", "vaginal itching", "pelvic pain",
        "vaginal pain", "vaginal redness", "involuntary urination",
        "retention of urine", "pain during intercourse", "infertility"
    ],
    "🦴 Musculoskeletal": [
        "back pain", "joint pain", "muscle weakness", "leg pain",
        "hip pain", "knee pain", "shoulder pain", "neck pain",
        "arm pain", "wrist pain", "ankle pain", "muscle cramps",
        "low back pain", "side pain", "rib pain"
    ],
    "🩻 Skin & Appearance": [
        "skin rash", "skin lesion", "itching of skin", "acne or pimples",
        "skin growth", "abnormal appearing skin", "skin dryness",
        "skin swelling", "skin moles", "diaper rash"
    ],
    "👁️ Eye & Vision": [
        "diminished vision", "double vision", "pain in eye",
        "eye redness", "lacrimation", "itchiness of eye",
        "blindness", "spots or clouds in vision", "foreign body sensation"
    ],
    "🦻 Ear, Nose & Throat": [
        "ear pain", "ringing in ear", "plugged feeling in ear",
        "itchy ear(s)", "fluid in ear", "sore throat", 
        "hoarse voice", "difficulty speaking"
    ],
    "🩸 Systemic & General": [
        "fever", "fatigue", "weakness", "chills", "sweating",
        "weight gain", "loss of appetite", "flu-like syndrome", 
        "feeling ill", "ache all over"
    ],
    "👶 Pregnancy & Reproductive": [
        "pain during pregnancy", "spotting or bleeding during pregnancy",
        "uterine contractions", "recent pregnancy", "problems during pregnancy",
        "intermenstrual bleeding", "heavy menstrual flow", "painful menstruation",
        "long menstrual periods", "unpredictable menstruation"
    ]
}

def get_all_symptoms():
    """Get all symptoms from all categories"""
    all_symptoms = []
    for symptoms in SYMPTOM_CATEGORIES.values():
        all_symptoms.extend(symptoms)
    return list(set(all_symptoms))

def get_symptom_category(symptom):
    """Get category for a specific symptom"""
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        if symptom in symptoms:
            return category
    return "📌 Other Symptoms"

def get_category_symptoms(category):
    """Get symptoms for a specific category"""
    return SYMPTOM_CATEGORIES.get(category, [])

def get_category_count():
    """Get count of symptoms per category"""
    return {category: len(symptoms) for category, symptoms in SYMPTOM_CATEGORIES.items()}