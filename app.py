#!/usr/bin/env python3
"""
Clinical Trial Screening AI - Gradio Web Interface
Interactive demo for patient eligibility assessment.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from clinical_trial_classifier import ClinicalTrialClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier = None

def initialize_classifier():
    """Initialize the clinical trial classifier."""
    global classifier
    try:
        print("🔄 Initializing Clinical Trial Classifier...")
        classifier = ClinicalTrialClassifier()
        classifier.load_model()
        print("✅ Classifier initialized successfully!")
        logger.info("Classifier initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        logger.error(f"Failed to initialize classifier: {e}")
        return False

def screen_patient_eligibility(patient_text: str, criteria_text: str):
    """
    Screen patient eligibility for clinical trial.
    
    Args:
        patient_text: Patient clinical notes
        criteria_text: Trial eligibility criteria
        
    Returns:
        Tuple of (eligibility_label, confidence_score, reasoning_text, attention_plot)
    """
    global classifier
    
    if not classifier:
        return (
            {"Error": 1.0},
            0.0,
            "❌ Error: System not initialized. Please refresh the page.",
            None
        )
    
    if not patient_text.strip() or not criteria_text.strip():
        return (
            {"Missing input": 1.0},
            0.0,
            "⚠️ Please provide both patient information and trial criteria.",
            None
        )
    
    try:
        # Get prediction
        print(f"🔍 Analyzing patient eligibility...")
        result = classifier.predict_eligibility(patient_text, criteria_text)
        
        # Format eligibility result
        eligibility_status = "✅ ELIGIBLE" if result["eligible"] else "❌ NOT ELIGIBLE"
        eligibility_label = {
            "✅ ELIGIBLE" if result["eligible"] else "❌ NOT ELIGIBLE": result["confidence"]
        }
        
        # Generate reasoning text
        reasoning = generate_reasoning_text(result, patient_text, criteria_text)
        
        # Create confidence visualization
        confidence_plot = create_confidence_plot(result)
        
        print(f"✅ Analysis complete: {eligibility_status} ({result['confidence']:.1%} confidence)")
        
        return (
            eligibility_label,
            result["confidence"],
            reasoning,
            confidence_plot
        )
        
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        return (
            {"Error": 1.0},
            0.0,
            f"❌ {error_msg}",
            None
        )

def generate_reasoning_text(result: dict, patient_text: str, criteria_text: str) -> str:
    """
    Generate human-readable reasoning for the eligibility decision.
    
    Args:
        result: Prediction result dictionary
        patient_text: Patient clinical notes
        criteria_text: Trial criteria
        
    Returns:
        Formatted reasoning text
    """
    reasoning_parts = []
    
    # Decision summary
    decision = "ELIGIBLE" if result["eligible"] else "NOT ELIGIBLE"
    confidence = result["confidence"]
    
    reasoning_parts.append(f"🎯 **DECISION**: {decision}")
    reasoning_parts.append(f"📊 **CONFIDENCE**: {confidence:.1%}")
    reasoning_parts.append(f"🔍 **RISK ASSESSMENT**: {result['risk_assessment'].replace('_', ' ').title()}")
    
    reasoning_parts.append("\n**ANALYSIS BREAKDOWN**:")
    reasoning_parts.append(f"• Probability of Eligibility: {result['probability_eligible']:.1%}")
    reasoning_parts.append(f"• Probability of Ineligibility: {result['probability_not_eligible']:.1%}")
    
    # Risk-based recommendations
    reasoning_parts.append("\n**CLINICAL INTERPRETATION**:")
    
    if result["risk_assessment"].startswith("high_confidence"):
        reasoning_parts.append("• High confidence prediction - strong alignment with criteria")
        reasoning_parts.append("• Recommended action: Proceed with protocol decision")
    elif result["risk_assessment"].startswith("moderate_confidence"):
        reasoning_parts.append("• Moderate confidence prediction - some uncertainty present")
        reasoning_parts.append("• Recommended action: Review decision with clinical team")
    else:
        reasoning_parts.append("• Low confidence prediction - significant uncertainty")
        reasoning_parts.append("• Recommended action: Manual expert review required")
    
    # Extract key clinical factors
    reasoning_parts.append("\n**KEY CLINICAL FACTORS DETECTED**:")
    
    # Simple keyword extraction for demonstration
    patient_lower = patient_text.lower()
    if "diabetes" in patient_lower:
        reasoning_parts.append("• Diabetes diagnosis identified")
    if "hba1c" in patient_lower or "hemoglobin" in patient_lower:
        reasoning_parts.append("• HbA1c levels mentioned")
    if "bmi" in patient_lower or "body mass" in patient_lower:
        reasoning_parts.append("• BMI information provided")
    if "cardiovascular" in patient_lower or "heart" in patient_lower:
        reasoning_parts.append("• Cardiovascular history noted")
    if "insulin" in patient_lower:
        reasoning_parts.append("• Insulin therapy mentioned")
    
    # Safety disclaimers
    reasoning_parts.append("\n**⚠️ IMPORTANT DISCLAIMERS**:")
    reasoning_parts.append("• This is a research demonstration using synthetic data")
    reasoning_parts.append("• Not intended for actual clinical decision-making")
    reasoning_parts.append("• Requires validation by qualified medical professionals")
    reasoning_parts.append("• All patient data shown is artificially generated")
    
    return "\n".join(reasoning_parts)

def create_confidence_plot(result: dict):
    """
    Create a visualization of prediction confidence.
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Plotly figure showing confidence metrics
    """
    try:
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Eligibility Probabilities", "Confidence Assessment"],
            specs=[[{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Probability bar chart
        probabilities = [result["probability_eligible"], result["probability_not_eligible"]]
        labels = ["Eligible", "Not Eligible"]
        colors = ["#2E8B57", "#DC143C"]  # Sea green and crimson
        
        fig.add_trace(
            go.Bar(
                x=labels,
                y=probabilities,
                marker_color=colors,
                name="Probabilities",
                text=[f"{p:.1%}" for p in probabilities],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        # Confidence gauge
        confidence_color = "#2E8B57" if result["confidence"] >= 0.8 else "#FF8C00" if result["confidence"] >= 0.6 else "#DC143C"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=result["confidence"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence Score"},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": confidence_color},
                    "steps": [
                        {"range": [0, 0.6], "color": "lightgray"},
                        {"range": [0.6, 0.8], "color": "yellow"},
                        {"range": [0.8, 1], "color": "lightgreen"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.9
                    }
                }
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Clinical Trial Screening - Prediction Analysis",
            height=400,
            showlegend=False,
            font=dict(size=12)
        )
        
        fig.update_xaxes(title_text="Decision", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1, range=[0, 1])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating confidence plot: {e}")
        return None

def load_demo_examples():
    """Load demonstration examples."""
    global classifier
    if classifier:
        return classifier.create_synthetic_demo_data()
    return []

def load_example_1():
    """Load first example."""
    return (
        """45-year-old female with Type 2 diabetes diagnosed 3 years ago. 
Current medications: Metformin 1000mg twice daily, Lisinopril 10mg daily. 
HbA1c: 7.2% (last measured 2 months ago). Blood pressure: 135/85 mmHg. 
No history of cardiovascular events. BMI: 28.5 kg/m². Patient reports good 
medication adherence and follows diabetic diet.""",
        
        """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
pregnancy or nursing."""
    )

def load_example_2():
    """Load second example."""
    return (
        """52-year-old male with Type 1 diabetes since childhood. 
Currently on intensive insulin therapy with insulin pump. HbA1c: 8.1%. 
BMI: 24.2 kg/m². No cardiovascular history. Blood pressure well controlled 
at 120/80 mmHg.""",
        
        """Inclusion Criteria: Adults aged 18-65 years, Type 2 diabetes 
mellitus diagnosis, HbA1c between 7.0% and 10.0%, BMI 25-35 kg/m². 
Exclusion Criteria: History of cardiovascular events, current insulin therapy, 
pregnancy or nursing."""
    )

def create_interface():
    """Create the Gradio interface for clinical trial screening."""
    
    # Custom CSS for healthcare theme
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .healthcare-header {
        background: linear-gradient(90deg, #2E8B57, #20B2AA);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFE69C;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Clinical Trial Screening AI") as demo:
        
        # Header
        gr.HTML("""
        <div class="healthcare-header">
            <h1>🏥 Clinical Trial Screening AI</h1>
            <h3>Advanced Healthcare AI System for Patient Eligibility Assessment</h3>
        </div>
        """)
        
        # Warning disclaimer
        gr.HTML("""
        <div class="warning-box">
            <h4>⚠️ IMPORTANT DISCLAIMER</h4>
            <p><strong>This is a research demonstration using synthetic data only.</strong> 
            This system is not intended for actual clinical use and has not been validated 
            by medical professionals or regulatory authorities. All patient data shown is 
            artificially generated for demonstration purposes.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Information")
                
                patient_input = gr.Textbox(
                    label="Patient Clinical Notes",
                    placeholder="Enter patient medical history, current medications, lab results, etc...",
                    lines=8,
                    value=""
                )
                
                criteria_input = gr.Textbox(
                    label="Clinical Trial Eligibility Criteria", 
                    placeholder="Enter trial inclusion and exclusion criteria...",
                    lines=6,
                    value=""
                )
                
                with gr.Row():
                    screen_btn = gr.Button("🔍 Screen Patient", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### 📋 Example Cases")
                with gr.Row():
                    example1_btn = gr.Button("Load Example 1: Eligible Patient", variant="secondary")
                    example2_btn = gr.Button("Load Example 2: Ineligible Patient", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Screening Results")
                
                eligibility_output = gr.Label(label="Eligibility Decision", scale=2)
                
                with gr.Row():
                    confidence_output = gr.Number(
                        label="Confidence Score", 
                        precision=3,
                        interactive=False
                    )
                
                reasoning_output = gr.Textbox(
                    label="Clinical Analysis & Reasoning", 
                    lines=10,
                    interactive=False
                )
                
                attention_plot = gr.Plot(label="Confidence Visualization")
        
        # Event handlers
        screen_btn.click(
            screen_patient_eligibility,
            inputs=[patient_input, criteria_input],
            outputs=[eligibility_output, confidence_output, reasoning_output, attention_plot]
        )
        
        clear_btn.click(
            lambda: ("", "", None, 0.0, "", None),
            outputs=[patient_input, criteria_input, eligibility_output, confidence_output, reasoning_output, attention_plot]
        )
        
        example1_btn.click(
            load_example_1,
            outputs=[patient_input, criteria_input]
        )
        
        example2_btn.click(
            load_example_2,
            outputs=[patient_input, criteria_input]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #eee;">
            <h4>🔬 Advanced Healthcare AI Portfolio Project</h4>
            <p>Demonstrating sophisticated NLP capabilities for clinical applications with ethical AI principles.</p>
            <p><em>Built with PyTorch, Transformers, and Gradio for healthcare AI research.</em></p>
        </div>
        """)
    
    return demo

def main():
    """Main function to launch the application."""
    
    print("=" * 60)
    print("🏥 CLINICAL TRIAL SCREENING AI")
    print("=" * 60)
    
    # Initialize the classifier
    print("🔄 Initializing Clinical Trial Screening AI...")
    
    if not initialize_classifier():
        print("❌ Failed to initialize classifier. Exiting...")
        return
    
    print("✅ Classifier initialized successfully!")
    
    # Create and launch interface
    print("🚀 Launching web interface...")
    print("📡 Server will be available at: http://localhost:7860")
    print("🔧 Use Ctrl+C to stop the server")
    
    demo = create_interface()
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
