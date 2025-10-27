"""
BioSimPy - Improved Streamlit App (Single-file)
Optimized + documented version combining:
 - Microfluidic simulation (laminar/parabolic flow option)
 - Biosensor response generator
 - AI calibration pipeline (cached)
 - Streamlit dashboard

Notes / References:
 - Poiseuille flow (parabolic velocity profile) for pressure-driven laminar flow in channels:
   u(y) = umax * (1 - (2y/H - 1)^2), approximated here in 2D across channel height.
 - Simple advection-diffusion approximations: we use shift + gaussian smoothing to mimic advective transport + diffusion.
 - Calibration uses supervised regression (RandomForest/Linear) on (sensor_reading) -> true_concentration.

This is a research-demo. For rigorous CFD/transport modeling, use COMSOL, OpenFOAM, or solve full Navier-Stokes + advection-diffusion PDEs.
"""

from functools import partial
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------
# App-level settings & meta
# --------------------------
st.set_page_config(page_title="BioSimPy - BioMEMS Simulator", page_icon="🧬", layout="wide")
np.set_printoptions(precision=4, suppress=True)




# --------------------------
# Utility / config
# --------------------------
RNG = np.random.default_rng  # factory
DEFAULT_SEED = 42

# --------------------------
# Microfluidic Simulator
# --------------------------
class MicrofluidicSimulator:
    """
    Lightweight microfluidic simulator with two modes:
     - 'random' : randomized fields (quick demo)
     - 'laminar': parabolic velocity profile across channel height + advective concentration decay across width
    This is intentionally simplified to be fast and visually realistic for demo/education.
    """

    def __init__(self, width=100, height=100, seed=DEFAULT_SEED):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.rng = RNG(self.seed)

    # @st.cache_data(show_spinner=False)
    def simulate_y_channel(self, flow_rate=1.0, viscosity=0.01, steps=1, mode='laminar'):
        """
        Simulate a microchannel cross-section (2D grid).
        Args:
          flow_rate: scalar controlling maximum velocity scaling
          viscosity: currently only used for label/metadata (not in physics simplified model)
          steps: future placeholder for time-stepping (kept for API compatibility)
          mode: 'laminar' or 'random'

        Returns:
          vx, vy, concentration, metadata
        """
        W, H = self.width, self.height
        rng = RNG(self.seed)

        if mode == 'laminar':
            # Create parabolic profile across height (Poiseuille-like)
            y = np.linspace(0, 1, H)
            # normalized parabola: 1 at center, 0 at walls
            parabola = 1.0 - ((2 * y - 1.0) ** 2)
            # Broadcast across width
            vx = np.tile(parabola[:, None], (1, W))
            # Scale by flow_rate
            vx = vx * flow_rate
            # Small lateral velocity to simulate weak transversal mixing
            vy = 0.05 * (rng.random((H, W)) - 0.5) * flow_rate

            # Concentration: inlet on the left, advected to right with diffusion smoothing
            # Start with high concentration at left half, low on right
            conc = np.zeros((H, W))
            inlet_strength = 1.0
            conc[:, : int(W * 0.2)] = inlet_strength  # inlet band
            # Advect (simulate by shifting) proportional to local mean vx
            avg_vx = np.mean(vx, axis=0)
            # build a simple advected concentration: smear to the right based on avg_vx
            advected = np.zeros_like(conc)
            for col in range(W):
                shift = int(np.clip(avg_vx[col] * 2.0, 0, W // 4))  # heuristic shift
                if shift == 0:
                    advected[:, col] = conc[:, col]
                else:
                    src_col = max(0, col - shift)
                    advected[:, col] = conc[:, src_col]
            # apply diffusion smoothing
            conc = gaussian_filter(advected, sigma=(1.2, 1.2))
            # add mild random fluctuations (measurement-level)
            conc += 0.02 * rng.standard_normal((H, W))
            conc = np.clip(conc, 0.0, None)
            mask = np.ones_like(conc, dtype=bool)
        else:
            # Random quick demo for very fast runs (kept as fallback)
            vx = rng.random((H, W)) * flow_rate
            vy = rng.random((H, W)) * flow_rate * 0.1
            conc = rng.random((H, W))
            mask = np.ones((H, W), dtype=bool)

        metadata = dict(width=W, height=H, flow_rate=float(flow_rate), viscosity=float(viscosity), mode=mode)
        return vx, vy, conc, mask, metadata

    def visualize_flow_figure(self, vx, vy, conc, mask, title_prefix="Flow Visualization"):
        """
        Returns a Plotly Figure with two subplots: velocity magnitude heatmap and concentration heatmap.
        Using Plotly keeps the interactivity and looks crisp in Streamlit.
        """
        H, W = vx.shape
        vel = np.sqrt(vx ** 2 + vy ** 2)

        fig = make_two_heatmap_figure(vel, conc, title_prefix=title_prefix)
        return fig

# --------------------------
# Biosensor module
# --------------------------
class Biosensor:
    """
    Simple biosensor model that takes a "true concentration" time-series and produces a noisy sensor output.
    The user can adjust noise level and scaling to simulate different sensor characteristics.
    """

    def __init__(self, sensor_type="glucose", seed=DEFAULT_SEED):
        self.sensor_type = sensor_type
        self.seed = int(seed)
        self.rng = RNG(self.seed)

    def create_sample_concentration(self, time_points=500, time_hours=6):
        """
        Create synthetic 'true' concentration time series.
        Structure: baseline + circadian-like oscillation + two gaussian 'meal' peaks + small stochastic component.
        """
        t = np.linspace(0, time_hours, time_points)
        baseline = 5.0
        circadian = 2.0 * np.sin(2 * np.pi * t / 3.0)
        meals = 3.0 * np.exp(-((t - 1.5) / 0.5) ** 2) + 2.5 * np.exp(-((t - 4.0) / 0.5) ** 2)
        noise = 0.25 * self.rng.normal(0, 1, time_points)  # smaller noise in true concentration
        conc = baseline + circadian + meals + noise
        conc = np.clip(conc, 0.0, None)
        return t, conc

    def generate_response(self, true_conc, noise_level=0.02, drift_strength=0.0):
        """
        Generate noisy sensor output from true concentration.
        drift_strength: linear drift over time (fraction)
        """
        n = len(true_conc)
        rng = RNG(self.seed)
        noise = rng.normal(0.0, noise_level, size=n)
        # Add simple linear drift if required
        drift = drift_strength * np.linspace(0, 1, n) * np.mean(true_conc)
        sensor_out = true_conc + noise + drift
        sensor_out = np.clip(sensor_out, 0.0, None)
        return sensor_out

# --------------------------
# Calibration / ML module
# --------------------------
class SensorCalibrator:
    """
    ML calibration wrapper using sklearn Pipeline (StandardScaler + Model).
    Training is cached via Streamlit caching to avoid re-training unless inputs change.
    """

    def __init__(self, model_type="random_forest", random_state=DEFAULT_SEED):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None

    # @st.cache_data(show_spinner=False)
    def train(self, sensor_readings, true_concentrations, model_type="random_forest"):
        """
        Train and return a persisted sklearn pipeline and metrics.
        Caching is keyed by function arguments (hashable arrays). Works well for demos.
        """
        X = sensor_readings.reshape(-1, 1)
        y = true_concentrations.reshape(-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        if model_type == "linear":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)

        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipeline.fit(X_train, y_train)

        # Predictions and metrics
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        metrics = {
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "model_type": model_type
        }
        # Return pipeline and metrics
        return pipeline, metrics

    def calibrate(self, pipeline, sensor_readings):
        """Apply a trained pipeline to sensor readings."""
        X = sensor_readings.reshape(-1, 1)
        calibrated = pipeline.predict(X)
        return calibrated

# --------------------------
# Classifier (simple thresholds)
# --------------------------
class BiomarkerClassifier:
    """
    Threshold-based classification. Thresholds here are illustrative—adjust to real clinical ranges per biomarker.
    """

    def __init__(self):
        self.thresholds = {
            "glucose": {"Normal": (70, 100), "High": (100, 125), "Critical": (125, 1000)},
            "oxygen": {"Normal": (95, 100), "Low": (90, 95), "Critical": (0, 90)}
        }

    def classify(self, concentrations, biomarker_type="glucose"):
        thr = self.thresholds.get(biomarker_type, self.thresholds["glucose"])
        cats = []
        for c in concentrations:
            # we expect concentrations in same units; these thresholds are placeholders
            if thr["Normal"][0] <= c <= thr["Normal"][1]:
                cats.append("Normal")
            elif thr["High"][0] <= c <= thr["High"][1]:
                cats.append("High")
            else:
                cats.append("Critical")
        return cats

# --------------------------
# Plot helpers
# --------------------------
def make_two_heatmap_figure(vel_field, conc_field, title_prefix="Flow Visualization"):
    """Return a Plotly figure containing two side-by-side heatmaps with colorbars."""
    H, W = vel_field.shape
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Velocity Magnitude", "Concentration Distribution"),
                        horizontal_spacing=0.08)

    fig.add_trace(go.Heatmap(z=vel_field, colorbar=dict(title="|v|"), showscale=True), row=1, col=1)
    fig.add_trace(go.Heatmap(z=conc_field, colorbar=dict(title="Conc"), showscale=True), row=1, col=2)

    fig.update_layout(title=f"{title_prefix}", height=450)
    return fig

# import here to avoid circular early import
from plotly.subplots import make_subplots

# --------------------------
# Streamlit App UI
# --------------------------
def main():
    st.title("🧬 BioSimPy: BioMEMS Simulation Platform")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Module", ["Home", "Learn_BioMEMS", "Microfluidics", "Biosensor", "AI Calibration"])

    if mode == "Home":
        show_home()
    elif mode == "Microfluidics":
        show_microfluidics()
    elif mode == "Biosensor":
        show_biosensor()
    elif mode == "Learn_BioMEMS":
        show_learn_biomems()
    elif mode == "AI Calibration":
        show_ai_calibration()


# --------------------------
# Pages
# --------------------------
def show_home():
    st.header("Welcome — quick overview")
    st.markdown(
        """
        **BioSimPy** is an educational/research demo for BioMEMS:
        - Microfluidic flow simulation (laminar & advective-diffusive approx)
        - Biosensor signal modeling (noise + drift)
        - AI calibration (sklearn pipelines)
        \n
        *This demo is intended for concept-prototyping and visualization. It is not a substitute for full CFD or lab experiments.*  
        """
    )
def show_learn_biomems():
    st.header("🧬 Learn BioMEMS — Understanding the Science Behind the Simulation")

    st.markdown("""
    ---
    ### 🌱 **What Are BioMEMS?**
    **BioMEMS** stands for *Biological Micro-Electro-Mechanical Systems* — tiny devices that merge **electronics, mechanics, and biology** on a single chip.  
    Think of it as a mini-lab that fits on your fingertip 🧫✨  

    These chips can sense, analyze, and even manipulate biological samples like blood, DNA, or proteins at microscopic scales.

    **They’re used in:**  
    - Rapid COVID & glucose testing strips  
    - DNA microarrays  
    - Lab-on-a-chip systems  
    - Organ-on-chip drug testing devices  

    📖 *In simple words:* BioMEMS are how we shrink biology into chips that **think, sense, and react.**
    """)

    st.markdown("""
    ---
    ### 💧 **Microfluidics — The Art of Moving Tiny Droplets**
    The “MEMS” part becomes *BioMEMS* when you add **microfluidics** — the science of controlling fluids at micrometer scales.  

    When fluids move through channels thinner than a hair, they behave differently:  
    no turbulence, no splashing — just **laminar flow**, smooth and predictable.

    Your simulator models this behavior through:
    - 💨 **Flow rate** — how fast the fluid moves  
    - 🧴 **Viscosity** — how thick the fluid is  
    - 🧩 **Channel geometry** — the shape of the microchannel  

    📊 The color maps and flow patterns in your simulation represent how real microchannels move and mix fluids in diagnostic devices.
    """)

    st.markdown("""
    ---
    ### 🔬 **Biosensors — Where Chemistry Meets Electronics**
    In BioMEMS, **biosensors** are where biological reactions meet microelectronics.  
    They detect substances like glucose, viruses, or toxins and convert them into electrical signals.

    But — these signals often suffer from **noise and drift**.  
    That’s where your simulator’s **AI calibration module** comes in 🤖💪 — to clean up and restore signal accuracy.
    """)

    st.markdown("""
    ---
    ### 🤖 **AI + BioMEMS — The New Frontier**
    Machine learning is transforming BioMEMS research by:  
    - Correcting **sensor drift** automatically  
    - Predicting **chemical interactions**  
    - Optimizing **microchannel designs** for efficient mixing  

    You’re not just simulating BioMEMS — you’re showcasing the *future of smart, AI-powered biochips.*

    🧩 *AI in BioMEMS = Smarter sensing, faster diagnosis, fewer errors.*
    """)

    st.markdown("""
    ---
    ### ⚙️ **Your Project in Action — BioSimPy**
    **BioSimPy** isn’t just a simulator.  
    It’s a **learning and research platform** combining:
    - ⚙️ Physics (fluid flow)  
    - 🧬 Biology & Chemistry (biosensing)  
    - 🤖 Artificial Intelligence (calibration and optimization)  

    When users play with parameters and visualize results, they’re literally running experiments in a *virtual micro-lab.*
    """)

    st.markdown("""
    ---
    ### 🌍 **Why It Matters**
    BioMEMS are changing the game in:  
    - 🩺 **Healthcare:** portable diagnostic chips  
    - 🌱 **Environment:** smart pollution monitors  
    - 💊 **Pharma:** drug testing & delivery  
    - 🍞 **Food safety:** real-time contamination sensors  

    Your simulator helps engineers and students test and learn safely — before moving to expensive lab setups.
    """)

    st.markdown("""
    ---
    ### 📚 **References & Further Reading**
    1. Madou, M. J. (2018). *Fundamentals of Microfabrication and Nanotechnology*. CRC Press.  
    2. Nguyen, N.-T., & Wereley, S. T. (2006). *Fundamentals and Applications of Microfluidics*. Artech House.  
    3. Kovacs, G. T. A. (1998). *Micromachined Transducers Sourcebook*. McGraw-Hill.  
    4. Kwon, J., et al. (2021). “Machine Learning-Driven Biosensor Calibration.” *Sensors and Actuators B: Chemical*, 341.  
    5. Manz, A., et al. (1990). “Miniaturized Total Chemical Analysis Systems.” *Sensors and Actuators B: Chemical*, 1(1–6), 244–248.
    """)

    st.info("💡 **Fun Fact:** Ten microchannels side-by-side are still thinner than a single human hair — yet they can move and mix fluids with microliter precision!")

def show_microfluidics():
    st.header("💧 Microfluidic Simulation")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Simulation Params")
        width = st.number_input("Grid Width (px)", value=100, min_value=20, max_value=400, step=10)
        height = st.number_input("Grid Height (px)", value=100, min_value=20, max_value=400, step=10)
        flow_rate = st.slider("Flow Rate (scale)", 0.1, 5.0, 1.0)
        viscosity = st.slider("Viscosity (approx)", 0.001, 0.1, 0.01)
        seed = st.number_input("Random Seed", value=DEFAULT_SEED)
        mode = st.radio("Flow Model", options=["laminar", "random"], index=0)
        if st.button("Run Simulation"):
            sim = MicrofluidicSimulator(width=width, height=height, seed=seed)
            vx, vy, conc, mask, metadata = sim.simulate_y_channel(flow_rate=flow_rate, viscosity=viscosity, mode=mode)
            # store
            st.session_state["flow"] = dict(vx=vx, vy=vy, conc=conc, mask=mask, metadata=metadata)

    with col_right:
        st.subheader("Visualization")
        if "flow" in st.session_state:
            f = st.session_state["flow"]
            sim = MicrofluidicSimulator(width=f["vx"].shape[1], height=f["vx"].shape[0], seed=DEFAULT_SEED)
            fig = sim.visualize_flow_figure(f["vx"], f["vy"], f["conc"], f["mask"])
            st.plotly_chart(fig, use_container_width=True)

            avg_vel = float(np.mean(np.sqrt(f["vx"] ** 2 + f["vy"] ** 2)))
            max_conc = float(np.max(f["conc"]))
            c1, c2 = st.columns(2)
            c1.metric("Average Velocity", f"{avg_vel:.4f} (a.u.)")
            c2.metric("Max Concentration", f"{max_conc:.4f}")
        else:
            st.info("Run a simulation to visualize flow.")
            st.image("https://via.placeholder.com/700x350.png?text=Run+Microfluidic+Simulation")

def show_biosensor():
    st.header("🔬 Biosensor Simulation")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        sensor_type = st.selectbox("Sensor Type", ["glucose", "oxygen", "lactate"])
        noise_level = st.slider("Noise Level (std dev)", 0.0, 0.2, 0.02)
        drift = st.slider("Linear Drift (fraction of mean over duration)", 0.0, 0.5, 0.0)
        time_points = st.slider("Time points", 100, 2000, 500)
        sim_seed = st.number_input("Seed", value=DEFAULT_SEED)

        if st.button("Generate Sensor Data"):
            sensor = Biosensor(sensor_type=sensor_type, seed=int(sim_seed))
            t, true_conc = sensor.create_sample_concentration(time_points=time_points, time_hours=6)
            sensor_out = sensor.generate_response(true_conc, noise_level=noise_level, drift_strength=drift)

            st.session_state["sensor_data"] = dict(time=t, true=true_conc, raw=sensor_out, sensor_type=sensor_type)
            st.success("Sensor data generated.")

    with col_right:
        st.subheader("Response")
        if "sensor_data" in st.session_state:
            data = st.session_state["sensor_data"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["time"], y=data["true"], name="True Concentration", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=data["time"], y=data["raw"], name="Sensor Output", line=dict(color="red")))
            fig.update_layout(height=450, xaxis_title="Time (hours)", yaxis_title="Concentration")
            st.plotly_chart(fig, use_container_width=True)

            mae = mean_absolute_error(data["true"], data["raw"])
            max_err = float(np.max(np.abs(data["true"] - data["raw"])))
            c1, c2 = st.columns(2)
            c1.metric("Mean Absolute Error", f"{mae:.4f}")
            c2.metric("Maximum Error", f"{max_err:.4f}")
        else:
            st.info("Generate sensor data to visualize response.")

def show_ai_calibration():
    st.header("🤖 AI Calibration")
    if "sensor_data" not in st.session_state:
        st.warning("Generate sensor data in the Biosensor tab first.")
        return

    data = st.session_state["sensor_data"]
    col_left, col_right = st.columns([1, 2])
    with col_left:
        model_type = st.selectbox("Model Type", ["random_forest", "linear"])
        test_size = st.slider("Test fraction", 5, 50, 20)
        train_seed = st.number_input("Training Seed", value=DEFAULT_SEED)

        if st.button("Train Calibration Model"):
            calibrator = SensorCalibrator()
            pipeline, metrics = calibrator.train(data["raw"], data["true"], model_type=model_type)
            calibrated = calibrator.calibrate(pipeline, data["raw"])
            st.session_state["ai"] = dict(pipeline=pipeline, metrics=metrics, calibrated=calibrated)
            st.success("Calibration model trained and applied.")

    with col_right:
        if "ai" in st.session_state:
            ai = st.session_state["ai"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["time"], y=data["true"], name="True", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=data["time"], y=data["raw"], name="Raw Sensor", line=dict(color="red", width=2), opacity=0.6))
            fig.add_trace(go.Scatter(x=data["time"], y=ai["calibrated"], name="Calibrated (AI)", line=dict(color="blue")))
            fig.update_layout(height=450, xaxis_title="Time (hours)", yaxis_title="Concentration")
            st.plotly_chart(fig, use_container_width=True)

            raw_mae = mean_absolute_error(data["true"], data["raw"])
            ai_mae = mean_absolute_error(data["true"], ai["calibrated"])
            improvement = (raw_mae - ai_mae) / raw_mae * 100 if raw_mae != 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Raw MAE", f"{raw_mae:.4f}")
            c2.metric("AI MAE", f"{ai_mae:.4f}")
            c3.metric("Improvement", f"{improvement:.1f}%")
            st.write("Training metrics (cached):")
            st.json(ai["metrics"])
            # Classification (optional demonstration)
            classifier = BiomarkerClassifier()
            classes = classifier.classify(ai["calibrated"], data["sensor_type"])
            st.subheader("Classification summary (post-calibration)")
            st.write(pd.Series(classes).value_counts())
        else:
            st.info("Train the calibration model to see results.")

if __name__ == "__main__":
    main()
