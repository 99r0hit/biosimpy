import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Import simulation modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from microfluidics.flow_simulator import MicrofluidicSimulator
    from sensors.biosensor import Biosensor
    from ai_module.calibration import SensorCalibrator, BiomarkerClassifier
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure all module files are in the correct directories")

# Set page config
st.set_page_config(
    page_title="BioSimPy - BioMEMS Simulator",
    page_icon="🧬",
    layout="wide"
)

def main():
    st.title("🧬 BioSimPy: BioMEMS Simulation Platform")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Module",
        ["🏠 Home", "💧 Microfluidics", "🔬 Biosensor", "🤖 AI Calibration"]
    )
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "💧 Microfluidics":
        show_microfluidics()
    elif app_mode == "🔬 Biosensor":
        show_biosensor()
    elif app_mode == "🤖 AI Calibration":
        show_ai_calibration()

def show_home():
    st.header("Welcome to BioSimPy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌍 Overview
        BioSimPy is an open-source platform that bridges:
        - **Biology** - Biosensors and biomarkers
        - **MEMS** - Micro-electro-mechanical systems  
        - **AI** - Machine learning for calibration
        
        ### 🎯 Features
        - 💧 Microfluidic flow simulation
        - 🔬 Realistic biosensor modeling
        - 🤖 AI-powered signal calibration
        - 📊 Interactive visualization
        """)
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #ff4b4b;'>Quick Start</h3>
        <ol>
        <li>Go to <b>Biosensor</b> tab</li>
        <li>Generate sensor data</li>
        <li>Apply AI calibration</li>
        <li>View results!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Microfluidic Simulations", "50+", "12")
    with col2:
        st.metric("Sensor Types", "3", "Glucose, Oxygen, Lactate")
    with col3:
        st.metric("AI Models", "2", "Random Forest, Linear")

def show_microfluidics():
    st.header("💧 Microfluidic Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        width = st.slider("Channel Width", 50, 200, 100)
        height = st.slider("Channel Height", 50, 200, 100)
        flow_rate = st.slider("Flow Rate", 0.1, 5.0, 1.0)
        viscosity = st.slider("Viscosity", 0.001, 0.1, 0.01)
        
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Simulating microfluidic flow..."):
                try:
                    simulator = MicrofluidicSimulator(width, height)
                    vx, vy, conc, mask = simulator.simulate_y_channel(flow_rate, viscosity)
                    
                    # Store results
                    st.session_state.flow_results = {
                        'vx': vx, 'vy': vy, 'conc': conc, 'mask': mask,
                        'simulator': simulator
                    }
                    st.success("Simulation completed!")
                except Exception as e:
                    st.error(f"Simulation error: {e}")
    
    with col2:
        st.subheader("Flow Visualization")
        
        if 'flow_results' in st.session_state:
            results = st.session_state.flow_results
            try:
                fig = results['simulator'].visualize_flow(
                    results['vx'], results['vy'], results['conc'], results['mask']
                )
                st.pyplot(fig)
                
                # Show statistics
                avg_velocity = np.mean(np.sqrt(results['vx']**2 + results['vy']**2))
                max_conc = np.max(results['conc'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Velocity", f"{avg_velocity:.4f} m/s")
                with col2:
                    st.metric("Max Concentration", f"{max_conc:.3f}")
            except Exception as e:
                st.error(f"Visualization error: {e}")
        else:
            st.info("Click 'Run Simulation' to see results")
            st.image("https://via.placeholder.com/600x400/2E86AB/FFFFFF?text=Microfluidic+Flow+Simulation", 
                    use_column_width=True)

def show_biosensor():
    st.header("🔬 Biosensor Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sensor Configuration")
        
        sensor_type = st.selectbox("Sensor Type", ["glucose", "oxygen", "lactate"])
        noise_level = st.slider("Noise Level", 0.0, 0.1, 0.02)
        time_duration = st.slider("Duration (hours)", 1, 24, 6)
        
        if st.button("Generate Sensor Data", type="primary"):
            with st.spinner("Generating sensor response..."):
                try:
                    # Create biosensor
                    sensor = Biosensor(sensor_type)
                    
                    # Generate sample concentration data
                    true_conc = sensor.create_sample_data(500)
                    
                    # Generate sensor response
                    results = sensor.generate_response(true_conc, time_duration, noise_level)
                    
                    # Store in session state
                    st.session_state.sensor_data = results
                    st.session_state.sensor_type = sensor_type
                    st.success("Sensor data generated!")
                except Exception as e:
                    st.error(f"Error generating sensor data: {e}")
    
    with col2:
        st.subheader("Sensor Response")
        
        if 'sensor_data' in st.session_state:
            data = st.session_state.sensor_data
            
            # Create interactive plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['time'], y=data['true_concentration'],
                mode='lines', name='True Concentration',
                line=dict(color='green', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['time'], y=data['sensor_output'],
                mode='lines', name='Sensor Output',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"{st.session_state.sensor_type.title()} Sensor Response",
                xaxis_title="Time (hours)",
                yaxis_title="Concentration",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate statistics
            error = np.mean(np.abs(data['sensor_output'] - data['true_concentration']))
            max_error = np.max(np.abs(data['sensor_output'] - data['true_concentration']))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{error:.3f}")
            with col2:
                st.metric("Maximum Error", f"{max_error:.3f}")
        else:
            st.info("Configure sensor and generate data to see results")

def show_ai_calibration():
    st.header("🤖 AI Calibration")
    
    if 'sensor_data' not in st.session_state:
        st.warning("Please generate sensor data first in the Biosensor tab")
        return
    
    sensor_data = st.session_state.sensor_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("AI Model Settings")
        
        model_type = st.selectbox("Model Type", ["random_forest", "linear"])
        window_size = st.slider("Window Size", 5, 50, 10)
        
        if st.button("Train AI Model", type="primary"):
            with st.spinner("Training calibration model..."):
                try:
                    calibrator = SensorCalibrator()
                    
                    # Train model
                    training_results = calibrator.train_model(
                        sensor_data['sensor_output'],
                        sensor_data['true_concentration'],
                        model_type
                    )
                    
                    # Apply calibration
                    calibrated = calibrator.calibrate_readings(sensor_data['sensor_output'])
                    
                    # Classify levels
                    classifier = BiomarkerClassifier()
                    classifications = classifier.classify(calibrated, st.session_state.sensor_type)
                    
                    # Store results
                    st.session_state.ai_results = {
                        'calibrated': calibrated,
                        'classifications': classifications,
                        'training_results': training_results
                    }
                    st.success("AI model trained successfully!")
                except Exception as e:
                    st.error(f"Error training AI model: {e}")
    
    with col2:
        st.subheader("Calibration Results")
        
        if 'ai_results' in st.session_state:
            ai_data = st.session_state.ai_results
            
            # Plot comparison
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sensor_data['time'], y=sensor_data['true_concentration'],
                mode='lines', name='True Concentration',
                line=dict(color='green', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=sensor_data['time'], y=sensor_data['sensor_output'],
                mode='lines', name='Raw Sensor',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=sensor_data['time'], y=ai_data['calibrated'],
                mode='lines', name='AI Calibrated',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title="AI Calibration Results",
                xaxis_title="Time (hours)",
                yaxis_title="Concentration",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate improvements
            raw_error = np.mean(np.abs(sensor_data['sensor_output'] - sensor_data['true_concentration']))
            ai_error = np.mean(np.abs(ai_data['calibrated'] - sensor_data['true_concentration']))
            improvement = ((raw_error - ai_error) / raw_error) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Error", f"{raw_error:.3f}")
            with col2:
                st.metric("AI Error", f"{ai_error:.3f}")
            with col3:
                st.metric("Improvement", f"{improvement:.1f}%")
            
            # Show classification
            st.subheader("Biomarker Classification")
            normal_count = ai_data['classifications'].count('Normal')
            high_count = ai_data['classifications'].count('High')
            critical_count = ai_data['classifications'].count('Critical')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Normal", normal_count)
            with col2:
                st.metric("High", high_count)
            with col3:
                st.metric("Critical", critical_count)
        else:
            st.info("Train AI model to see calibration results")

if __name__ == "__main__":
    main()
