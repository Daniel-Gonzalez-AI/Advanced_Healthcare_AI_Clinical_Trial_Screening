#!/usr/bin/env python3
"""
Clinical Trial Screening AI - Core Module
Advanced healthcare AI system for patient eligibility assessment.
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from typing import Dict, List, Tuple, Optional
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTrialClassifier:
    """
    Advanced AI system for clinical trial patient screening.
    
    This class implements a text-pair classification system that can determine
    patient eligibility for clinical trials based on medical records and trial criteria.
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize the clinical trial classifier.
        
        Args:
            model_name: HuggingFace model identifier for clinical BERT variant
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        # Model performance metrics
        self.metrics = {}
        
    def load_model(self) -> None:
        """Load the tokenizer and model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                problem_type="single_label_classification"
            )
            self.model.to(self.device)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text for better model performance.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed clinical text
        """
        # Remove potential PHI patterns (for safety, even with synthetic data)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)  # SSN pattern
        text = re.sub(r'\b\d{16}\b', '[CREDIT_CARD]', text)  # Credit card pattern
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)  # Phone pattern
        
        # Normalize medical abbreviations
        abbreviations = {
            'w/': 'with',
            'w/o': 'without',
            'h/o': 'history of',
            'r/o': 'rule out',
            'pt': 'patient',
            'pts': 'patients',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'sx': 'symptoms'
        }
        
        for abbrev, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_criteria_text(self, text: str) -> str:
        """
        Preprocess trial criteria text.
        
        Args:
            text: Raw trial criteria text
            
        Returns:
            Preprocessed criteria text
        """
        # Standardize criteria formatting
        text = re.sub(r'inclusion criteria:?', 'Inclusion Criteria:', text, flags=re.IGNORECASE)
        text = re.sub(r'exclusion criteria:?', 'Exclusion Criteria:', text, flags=re.IGNORECASE)
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_pairs(self, patient_texts: List[str], criteria_texts: List[str]) -> Dict:
        """
        Tokenize patient-criteria text pairs.
        
        Args:
            patient_texts: List of patient clinical notes
            criteria_texts: List of trial criteria
            
        Returns:
            Tokenized inputs for the model
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess texts
        processed_patients = [self.preprocess_clinical_text(text) for text in patient_texts]
        processed_criteria = [self.preprocess_criteria_text(text) for text in criteria_texts]
        
        # Tokenize text pairs
        tokenized = self.tokenizer(
            processed_patients,
            processed_criteria,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return tokenized
    
    def predict_eligibility(self, patient_text: str, criteria_text: str) -> Dict:
        """
        Predict patient eligibility for a clinical trial.
        
        Args:
            patient_text: Patient clinical notes
            criteria_text: Trial eligibility criteria
            
        Returns:
            Dictionary with eligibility prediction and confidence scores
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input pair
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract results
            prob_not_eligible = probabilities[0][0].item()
            prob_eligible = probabilities[0][1].item()
            
            prediction = {
                "eligible": prob_eligible > prob_not_eligible,
                "confidence": max(prob_eligible, prob_not_eligible),
                "probability_eligible": prob_eligible,
                "probability_not_eligible": prob_not_eligible,
                "risk_assessment": self._assess_risk(prob_eligible)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "eligible": False,
                "confidence": 0.0,
                "probability_eligible": 0.0,
                "probability_not_eligible": 1.0,
                "risk_assessment": "error",
                "error": str(e)
            }
    
    def _assess_risk(self, prob_eligible: float) -> str:
        """
        Assess the risk level of the eligibility decision.
        
        Args:
            prob_eligible: Probability of eligibility
            
        Returns:
            Risk assessment string
        """
        if prob_eligible >= 0.9:
            return "high_confidence_eligible"
        elif prob_eligible >= 0.7:
            return "moderate_confidence_eligible"
        elif prob_eligible >= 0.5:
            return "low_confidence_eligible"
        elif prob_eligible >= 0.3:
            return "low_confidence_not_eligible"
        elif prob_eligible >= 0.1:
            return "moderate_confidence_not_eligible"
        else:
            return "high_confidence_not_eligible"
    
    def batch_predict(self, patient_texts: List[str], criteria_texts: List[str]) -> List[Dict]:
        """
        Predict eligibility for multiple patient-criteria pairs.
        
        Args:
            patient_texts: List of patient clinical notes
            criteria_texts: List of trial criteria
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for patient_text, criteria_text in zip(patient_texts, criteria_texts):
            prediction = self.predict_eligibility(patient_text, criteria_text)
            predictions.append(prediction)
        
        return predictions
    
    def get_attention_weights(self, patient_text: str, criteria_text: str) -> np.ndarray:
        """
        Extract attention weights for interpretability.
        
        Args:
            patient_text: Patient clinical notes
            criteria_text: Trial criteria
            
        Returns:
            Attention weights array
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get attention weights
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions
            
            # Process attention weights (average across heads and layers)
            attention_weights = torch.stack(attentions).mean(dim=0).mean(dim=1)
            return attention_weights.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting attention weights: {e}")
            return np.array([])
    
    def create_synthetic_demo_data(self) -> List[Dict]:
        """
        Create synthetic patient and trial data for demonstration.
        
        Returns:
            List of synthetic patient-criteria pairs with labels
        """
        demo_data = [
            {
                "patient_text": """
                Patient: 45-year-old female with Type 2 diabetes diagnosed 3 years ago.
                Current medications: Metformin 1000mg twice daily, Lisinopril 10mg daily.
                HbA1c: 7.2% (last measured 2 months ago).
                Blood pressure: 135/85 mmHg.
                No history of cardiovascular events.
                BMI: 28.5 kg/m².
                Regular exercise routine, non-smoker.
                """,
                "criteria_text": """
                Inclusion Criteria:
                - Adults aged 18-65 years
                - Type 2 diabetes mellitus diagnosis
                - HbA1c between 7.0% and 10.0%
                - BMI 25-35 kg/m²
                
                Exclusion Criteria:
                - History of cardiovascular events
                - Current insulin therapy
                - Pregnancy or nursing
                """,
                "expected_eligible": True,
                "scenario": "Typical eligible diabetes patient"
            },
            {
                "patient_text": """
                Patient: 72-year-old male with Type 2 diabetes for 15 years.
                History of myocardial infarction 2 years ago.
                Current medications: Insulin glargine, Metformin, Aspirin, Atorvastatin.
                HbA1c: 8.1%.
                BMI: 32 kg/m².
                """,
                "criteria_text": """
                Inclusion Criteria:
                - Adults aged 18-65 years
                - Type 2 diabetes mellitus diagnosis
                - HbA1c between 7.0% and 10.0%
                - BMI 25-35 kg/m²
                
                Exclusion Criteria:
                - History of cardiovascular events
                - Current insulin therapy
                - Pregnancy or nursing
                """,
                "expected_eligible": False,
                "scenario": "Multiple exclusion criteria (age, CV history, insulin)"
            },
            {
                "patient_text": """
                Patient: 35-year-old pregnant female with gestational diabetes.
                Currently 28 weeks pregnant.
                Blood glucose managed with diet and exercise.
                BMI pre-pregnancy: 26 kg/m².
                No other medical conditions.
                """,
                "criteria_text": """
                Inclusion Criteria:
                - Adults aged 18-65 years
                - Type 2 diabetes mellitus diagnosis
                - HbA1c between 7.0% and 10.0%
                - BMI 25-35 kg/m²
                
                Exclusion Criteria:
                - History of cardiovascular events
                - Current insulin therapy
                - Pregnancy or nursing
                """,
                "expected_eligible": False,
                "scenario": "Gestational diabetes (wrong type) and pregnancy exclusion"
            }
        ]
        
        return demo_data

def compute_metrics(eval_pred) -> Dict:
    """
    Compute evaluation metrics for model training.
    
    Args:
        eval_pred: Predictions and labels from evaluation
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    """
    Main function for testing the clinical trial classifier.
    """
    # Initialize classifier
    classifier = ClinicalTrialClassifier()
    
    try:
        # Load model
        classifier.load_model()
        
        # Test with synthetic data
        demo_data = classifier.create_synthetic_demo_data()
        
        print("🏥 Clinical Trial Screening AI - Test Results")
        print("=" * 60)
        
        for i, example in enumerate(demo_data, 1):
            print(f"\n📋 Test Case {i}: {example['scenario']}")
            
            # Make prediction
            result = classifier.predict_eligibility(
                example["patient_text"], 
                example["criteria_text"]
            )
            
            # Display results
            print(f"Expected: {'Eligible' if example['expected_eligible'] else 'Not Eligible'}")
            print(f"Predicted: {'Eligible' if result['eligible'] else 'Not Eligible'}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Risk Assessment: {result['risk_assessment']}")
            
            # Check if prediction matches expectation
            correct = result['eligible'] == example['expected_eligible']
            print(f"Status: {'✅ Correct' if correct else '❌ Incorrect'}")
        
        print("\n✅ Clinical Trial Classifier test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
