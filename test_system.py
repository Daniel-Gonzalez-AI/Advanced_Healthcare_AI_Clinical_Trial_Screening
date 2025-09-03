#!/usr/bin/env python3
"""
Automated testing script for Clinical Trial Screening AI
"""

import sys
import traceback
from clinical_trial_classifier import ClinicalTrialClassifier

def test_model_loading():
    """Test if the model loads correctly."""
    print("🔧 Testing model loading...")
    try:
        classifier = ClinicalTrialClassifier()
        classifier.load_model()
        print("✅ Model loading: PASSED")
        return True, classifier
    except Exception as e:
        print(f"❌ Model loading: FAILED - {e}")
        return False, None

def test_prediction_functionality(classifier):
    """Test prediction functionality."""
    print("🔧 Testing prediction functionality...")
    try:
        # Test case
        patient_text = "45-year-old female with Type 2 diabetes, HbA1c 7.2%, BMI 28.5"
        criteria_text = "Inclusion: Type 2 diabetes, HbA1c 7.0-10.0%, BMI 25-35. Exclusion: cardiovascular events"
        
        result = classifier.predict_eligibility(patient_text, criteria_text)
        
        # Verify result structure
        required_keys = ['eligible', 'confidence', 'probability_eligible', 'probability_not_eligible', 'risk_assessment']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key in result: {key}")
        
        # Verify data types and ranges
        if not isinstance(result['eligible'], bool):
            raise ValueError("eligible should be boolean")
        if not (0 <= result['confidence'] <= 1):
            raise ValueError("confidence should be between 0 and 1")
        if not (0 <= result['probability_eligible'] <= 1):
            raise ValueError("probability_eligible should be between 0 and 1")
        
        print("✅ Prediction functionality: PASSED")
        print(f"   Result: {'Eligible' if result['eligible'] else 'Not eligible'} ({result['confidence']:.1%} confidence)")
        return True
        
    except Exception as e:
        print(f"❌ Prediction functionality: FAILED - {e}")
        print(traceback.format_exc())
        return False

def test_batch_processing(classifier):
    """Test batch processing."""
    print("🔧 Testing batch processing...")
    try:
        patients = [
            "Type 2 diabetes patient, HbA1c 7.5%, BMI 30",
            "Type 1 diabetes patient, insulin therapy, HbA1c 8.0%"
        ]
        criteria = [
            "Inclusion: Type 2 diabetes. Exclusion: insulin therapy",
            "Inclusion: Type 2 diabetes. Exclusion: insulin therapy"
        ]
        
        results = classifier.batch_predict(patients, criteria)
        
        if len(results) != 2:
            raise ValueError(f"Expected 2 results, got {len(results)}")
        
        print("✅ Batch processing: PASSED")
        print(f"   Processed {len(results)} patients successfully")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing: FAILED - {e}")
        return False

def test_synthetic_data_generation(classifier):
    """Test synthetic data generation."""
    print("🔧 Testing synthetic data generation...")
    try:
        demo_data = classifier.create_synthetic_demo_data()
        
        if not isinstance(demo_data, list):
            raise ValueError("Demo data should be a list")
        if len(demo_data) < 3:
            raise ValueError(f"Expected at least 3 examples, got {len(demo_data)}")
        
        # Verify structure of first example
        example = demo_data[0]
        required_.py
        required_keys = ['patient_text', 'criteria_text', 'expected_eligible', 'description']
        for key in required_keys:
            if key not in example:
                raise ValueError(f"Missing key in demo data: {key}")
        
        print("✅ Synthetic data generation: PASSED")
        print(f"   Generated {len(demo_data)} demo examples")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data generation: FAILED - {e}")
        return False

def run_comprehensive_test():
    """Run all tests."""
    print("=" * 60)
    print("🧪 CLINICAL TRIAL SCREENING AI - COMPREHENSIVE TEST")
    print("=" * 60)
    
    test_results = []
    classifier = None
    
    # Test 1: Model Loading
    success, classifier = test_model_loading()
    test_results.append(("Model Loading", success))
    
    if not success:
        print("\n❌ Cannot proceed with further tests due to model loading failure")
        return False
    
    # Test 2: Prediction Functionality
    success = test_prediction_functionality(classifier)
    test_results.append(("Prediction Functionality", success))
    
    # Test 3: Batch Processing
    success = test_batch_processing(classifier)
    test_results.append(("Batch Processing", success))
    
    # Test 4: Synthetic Data Generation
    success = test_synthetic_data_generation(classifier)
    test_results.append(("Synthetic Data Generation", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed_tests += 1
    
    print("-" * 60)
    print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)