#!/usr/bin/env python3
"""
Complete demonstration of Clinical Trial Screening AI system
"""

import time
from clinical_trial_classifier import ClinicalTrialClassifier

def run_complete_demo():
    """Run a complete demonstration of the system."""
    print("=" * 70)
    print("🏥 CLINICAL TRIAL SCREENING AI - COMPLETE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize system
    print("\n🔧 PHASE 1: System Initialization")
    print("-" * 40)
    
    classifier = ClinicalTrialClassifier()
    print("📥 Loading clinical BERT model...")
    start_time = time.time()
    classifier.load_model()
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    
    # Demonstrate with synthetic examples
    print("\n🧪 PHASE 2: Clinical Screening Demonstrations")
    print("-" * 40)
    
    demo_cases = classifier.create_synthetic_demo_data()
    
    for i, case in enumerate(demo_cases, 1):
        print(f"\n--- Case {i}: {case['description']} ---")
        
        # Run prediction
        start_time = time.time()
        result = classifier.predict_eligibility(case['patient_text'], case['criteria_text'])
        prediction_time = time.time() - start_time
        
        # Display results
        decision = "✅ ELIGIBLE" if result['eligible'] else "❌ NOT ELIGIBLE"
        expected = "✅ ELIGIBLE" if case['expected_eligible'] else "❌ NOT ELIGIBLE"
        correct = "✅ CORRECT" if result['eligible'] == case['expected_eligible'] else "❌ INCORRECT"
        
        print(f"🎯 AI Decision: {decision}")
        print(f"📋 Expected: {expected}")
        print(f"🔍 Accuracy: {correct}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚡ Processing Time: {prediction_time:.3f}s")
        print(f"🔬 Risk Assessment: {result['risk_assessment'].replace('_', ' ').title()}")
        
        # Brief pause for readability
        time.sleep(1)
    
    # Performance summary
    print("\n📈 PHASE 3: Performance Summary")
    print("-" * 40)
    
    # Test batch processing
    patient_texts = [case['patient_text'] for case in demo_cases]
    criteria_texts = [case['criteria_text'] for case in demo_cases]
    
    start_time = time.time()
    batch_results = classifier.batch_predict(patient_texts, criteria_texts)
    batch_time = time.time() - start_time
    
    # Calculate accuracy
    correct_predictions = sum(
        1 for result, case in zip(batch_results, demo_cases)
        if result['eligible'] == case['expected_eligible']
    )
    accuracy = correct_predictions / len(demo_cases)
    
    print(f"📊 Batch Processing: {len(demo_cases)} patients in {batch_time:.3f}s")
    print(f"⚡ Average Time per Patient: {batch_time/len(demo_cases):.3f}s")
    print(f"🎯 Demonstration Accuracy: {accuracy:.1%} ({correct_predictions}/{len(demo_cases)})")
    
    # System capabilities summary
    print(f"\n🔬 Model Information:")
    print(f"   • Base Model: {classifier.model_name}")
    print(f"   • Device: {classifier.device}")
    print(f"   • Task: Binary text-pair classification")
    print(f"   • Max Sequence Length: 512 tokens")
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 70)
    
    return classifier

if __name__ == "__main__":
    run_complete_demo()