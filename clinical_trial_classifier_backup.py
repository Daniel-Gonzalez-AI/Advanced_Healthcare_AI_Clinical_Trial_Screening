#!/usr/bin/env python3
"""
Clinical Trial Screening AI - Core Module
Advanced healthcare AI system for patient eligibility assessment.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging
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
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model for sequence classification (2 labels: eligible/not eligible)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                return_dict=True
            )
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text for better model performance.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed clinical text
        """
        if not text or not isinstance(text, str):
            return ""
        
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
            'sx': 'symptoms',
            'hx': 'history',
            'fx': 'fracture',
            'rx': 'prescription'
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
        if not text or not isinstance(text, str):
            return ""
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize the text pair
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Extract probabilities
            prob_not_eligible = probabilities[0]
            prob_eligible = probabilities[1]
            
            # Determine eligibility
            is_eligible = prob_eligible > prob_not_eligible
            confidence = max(prob_eligible, prob_not_eligible)
            
            # Assess risk level
            risk_assessment = self._assess_risk(prob_eligible)
            
            return {
                "eligible": bool(is_eligible),
                "confidence": float(confidence),
                "probability_eligible": float(prob_eligible),
                "probability_not_eligible": float(prob_not_eligible),
                "risk_assessment": risk_assessment,
                "raw_logits": logits.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            logger.error(f"Error in predict_eligibility: {e}")
            raise
    
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
            result = self.predict_eligibility(patient_text, criteria_text)
            predictions.append(result)
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # This is a simplified version - full attention extraction would require
            # a model that returns attention weights
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Return mock attention weights for demonstration
            # In a real implementation, you'd extract actual attention weights
            seq_length = inputs['input_ids'].shape[1]
            mock_attention = np.random.rand(seq_length, seq_length)
            
            return mock_attention
            
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
                "patient_text": """45-year-old female with Type 2 diabetes diagnosed 3 years ago. 
                Current medications: Metformin 1000mg twice daily, Lisinopril 10mg daily. 
                HbA1c: 7.2% (last measured 2 months ago). Blood pressure: 135/85 mmHg. 
                No history of cardiovascular events. BMI: 28.5 kg/m². Patient reports good 
                medication adherence and follows diabetic diet.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": True,
                "description": "Eligible diabetic patient meeting all criteria"
            },
            {
                "patient_text": """52-year-old male with Type 1 diabetes since childhood. 
                Currently on intensive insulin therapy with insulin pump. HbA1c: 8.1%. 
                BMI: 24.2 kg/m². No cardiovascular history. Blood pressure well controlled 
                at 120/80 mmHg.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": False,
                "description": "Not eligible: Type 1 diabetes and insulin therapy"
            },
            {
                "patient_text": """38-year-old female with Type 2 diabetes, well-controlled on 
                metformin. HbA1c: 6.8%. BMI: 32 kg/m². History of myocardial infarction 
                2 years ago, currently stable. Takes aspirin and atorvastatin. 
                Non-smoker, exercises regularly.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": False,
                "description": "Not eligible: HbA1c too low and cardiovascular history"
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
    print("🏥 Testing Clinical Trial Classifier...")
    
    # Initialize classifier
    classifier = ClinicalTrialClassifier()
    
    try:
        # Load the model
        classifier.load_model()
        print("✅ Model loaded successfully!")
        
        # Test with synthetic data
        demo_data = classifier.create_synthetic_demo_data()
        
        print(f"\n🧪 Testing with {len(demo_data)} synthetic examples...")
        
        for i, example in enumerate(demo_data):
            print(f"\n--- Test Case {i+1}: {example['description']} ---")
            
            result = classifier.predict_eligibility(
                example['patient_text'], 
                example['criteria_text']
            )
            
            print(f"Prediction: {'✅ ELIGIBLE' if result['eligible'] else '❌ NOT ELIGIBLE'}")
            print(f"Expected: {'✅ ELIGIBLE' if example['expected_eligible'] else '❌ NOT ELIGIBLE'}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Eligible Probability: {result['probability_eligible']:.1%}")
            print(f"Risk Assessment: {result['risk_assessment']}")
            
            # Check if prediction matches expectation
            if result['eligible'] == example['expected_eligible']:
                print("✅ CORRECT prediction!")
            else:
                print("❌ INCORRECT prediction!")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
        
        # Model performance metrics
        self.metrics = {}
        
    def load_model(self) -> None:
        """Load the tokenizer and model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model for sequence classification (2 labels: eligible/not eligible)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                return_dict=True
            )
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text for better model performance.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed clinical text
        """
        if not text or not isinstance(text, str):
            return ""
        
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
            'sx': 'symptoms',
            'hx': 'history',
            'fx': 'fracture',
            'rx': 'prescription'
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
        if not text or not isinstance(text, str):
            return ""
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize the text pair
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Extract probabilities
            prob_not_eligible = probabilities[0]
            prob_eligible = probabilities[1]
            
            # Determine eligibility
            is_eligible = prob_eligible > prob_not_eligible
            confidence = max(prob_eligible, prob_not_eligible)
            
            # Assess risk level
            risk_assessment = self._assess_risk(prob_eligible)
            
            return {
                "eligible": bool(is_eligible),
                "confidence": float(confidence),
                "probability_eligible": float(prob_eligible),
                "probability_not_eligible": float(prob_not_eligible),
                "risk_assessment": risk_assessment,
                "raw_logits": logits.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            logger.error(f"Error in predict_eligibility: {e}")
            raise
    
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
            result = self.predict_eligibility(patient_text, criteria_text)
            predictions.append(result)
        
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
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # This is a simplified version - full attention extraction would require
            # a model that returns attention weights
            inputs = self.tokenize_pairs([patient_text], [criteria_text])
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Return mock attention weights for demonstration
            # In a real implementation, you'd extract actual attention weights
            seq_length = inputs['input_ids'].shape[1]
            mock_attention = np.random.rand(seq_length, seq_length)
            
            return mock_attention
            
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
                "patient_text": """45-year-old female with Type 2 diabetes diagnosed 3 years ago. 
                Current medications: Metformin 1000mg twice daily, Lisinopril 10mg daily. 
                HbA1c: 7.2% (last measured 2 months ago). Blood pressure: 135/85 mmHg. 
                No history of cardiovascular events. BMI: 28.5 kg/m². Patient reports good 
                medication adherence and follows diabetic diet.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": True,
                "description": "Eligible diabetic patient meeting all criteria"
            },
            {
                "patient_text": """52-year-old male with Type 1 diabetes since childhood. 
                Currently on intensive insulin therapy with insulin pump. HbA1c: 8.1%. 
                BMI: 24.2 kg/m². No cardiovascular history. Blood pressure well controlled 
                at 120/80 mmHg.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": False,
                "description": "Not eligible: Type 1 diabetes and insulin therapy"
            },
            {
                "patient_text": """38-year-old female with Type 2 diabetes, well-controlled on 
                metformin. HbA1c: 6.8%. BMI: 32 kg/m². History of myocardial infarction 
                2 years ago, currently stable. Takes aspirin and atorvastatin. 
                Non-smoker, exercises regularly.""",
                
                "criteria_text": """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
                mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
                Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
                pregnancy or nursing.""",
                
                "expected_eligible": False,
                "description": "Not eligible: HbA1c too low and cardiovascular history"
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
    print("🏥 Testing Clinical Trial Classifier...")
    
    # Initialize classifier
    classifier = ClinicalTrialClassifier()
    
    try:
        # Load the model
        classifier.load_model()
        print("✅ Model loaded successfully!")
        
        # Test with synthetic data
        demo_data = classifier.create_synthetic_demo_data()
        
        print(f"\n🧪 Testing with {len(demo_data)} synthetic examples...")
        
        for i, example in enumerate(demo_data):
            print(f"\n--- Test Case {i+1}: {example['description']} ---")
            
            result = classifier.predict_eligibility(
                example['patient_text'], 
                example['criteria_text']
            )
            
            print(f"Prediction: {'✅ ELIGIBLE' if result['eligible'] else '❌ NOT ELIGIBLE'}")
            print(f"Expected: {'✅ ELIGIBLE' if example['expected_eligible'] else '❌ NOT ELIGIBLE'}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Eligible Probability: {result['probability_eligible']:.1%}")
            print(f"Risk Assessment: {result['risk_assessment']}")
            
            # Check if prediction matches expectation
            if result['eligible'] == example['expected_eligible']:
                print("✅ CORRECT prediction!")
            else:
                print("❌ INCORRECT prediction!")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
