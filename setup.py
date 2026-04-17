from setuptools import setup, find_packages

setup(
    name="disease-prediction-system",
    version="1.0.0",
    author="Amira YAA",
    description="AI-Powered Disease Prediction System using Symptom Analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
    ]
)
