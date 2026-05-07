import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="APS + ETC AI Simulation Platform",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Assisted APS + ETC Launcher Simulation Platform")
st.caption(
    "System-level academic demonstrator: threat classification, multi-threat prioritization, "
    "DMA launcher selection, ETC launcher performance, interception timing, Pk estimation, "
    "APS damage analysis, and CFD-style curves/contours."
)

st.warning(
    "Academic simplified model only. Use this for thesis visualization, trend discussion, and system-level demonstration. "
    "It is not a validated weapon-design or operational fire-control solver."
)


# ============================================================
# THREAT DATABASE
# ============================================================

THREAT_DB = {
    "RPG": {
        "velocity": 120,
        "default_distance": 180,
        "default_angle": 20,
        "lethality": 0.55,
        "description": "Low-to-medium velocity rocket-propelled threat."
    },
    "HEAT / HESH": {
        "velocity": 200,
        "default_distance": 180,
        "default_angle": 25,
        "lethality": 0.70,
        "description": "Chemical energy type threat with moderate velocity."
    },
    "ATGM": {
        "velocity": 280,
        "default_distance": 250,
        "default_angle": 35,
        "lethality": 0.85,
        "description": "Guided missile threat with high lethality and engagement urgency."
    },
    "Top-Attack Munition": {
        "velocity": 300,
        "default_distance": 220,
        "default_angle": 70,
        "lethality": 0.90,
        "description": "High-angle attack profile; difficult engagement geometry."
    },
    "FSAPDS / KE": {
        "velocity": 700,
        "default_distance": 300,
        "default_angle": 10,
        "lethality": 0.95,
        "description": "High-velocity kinetic-energy penetrator type threat."
    }
}


# ============================================================
# AI CLASSIFICATION MODEL
# ============================================================

def rule_classify_threat(velocity, distance, angle, lethality):
    tti = distance / max(velocity, 1e-6)

    score = 0
    score += min(velocity / 700, 1.5) * 0.35
    score += min(1 / max(tti, 0.01), 20) / 20 * 0.30
    score += min(angle / 75, 1.2) * 0.20
    score += lethality * 0.15

    if score >= 0.72:
        return 2
    elif score >= 0.42:
        return 1
    return 0


@st.cache_resource
def train_ai_model():
    data = []

    threat_names = list(THREAT_DB.keys())

    for _ in range(6000):
        threat = np.random.choice(threat_names)
        base = THREAT_DB[threat]

        velocity = np.random.uniform(max(60, base["velocity"] * 0.65), base["velocity"] * 1.35)
        distance = np.random.uniform(30, 500)
        angle = np.random.uniform(0, 80)
        lethality = base["lethality"]

        label = rule_classify_threat(velocity, distance, angle, lethality)
        data.append([velocity, distance, angle, lethality, label])

    df = pd.DataFrame(data, columns=["Velocity", "Distance", "Angle", "Lethality", "Threat_Class"])

    X = df[["Velocity", "Distance", "Angle", "Lethality"]]
    y = df["Threat_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(40, 20),
        activation="relu",
        solver="adam",
        max_iter=1200,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler, df


model, scaler, training_df = train_ai_model()

THREAT_LABELS = {
    0: "LOW THREAT",
    1: "MEDIUM THREAT",
    2: "HIGH THREAT"
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Global APS + ETC Inputs")

mode = st.sidebar.radio(
    "Engagement Mode",
    ["Single Threat", "Multi-Threat"]
)

selected_threat = st.sidebar.selectbox("Primary Threat Type", list(THREAT_DB.keys()), index=1)
base_threat = THREAT_DB[selected_threat]

velocity = st.sidebar.slider(
    "Threat Velocity (m/s)",
    80,
    900,
    int(base_threat["velocity"]),
    10
)

distance = st.sidebar.slider(
    "Initial Distance (m)",
    20,
    500,
    int(base_threat["default_distance"]),
    10
)

angle = st.sidebar.slider(
    "Incoming Angle (deg)",
    0,
    80,
    int(base_threat["default_angle"]),
    1
)

detection_range = st.sidebar.slider("Detection Range (m)", 50, 800, 350, 10)
interception_range = st.sidebar.slider("Interception Range (m)", 20, 250, 120, 5)

st.sidebar.divider()
st.sidebar.header("Reaction Time Model")

sensor_delay_ms = st.sidebar.slider("Sensor Detection Delay (ms)", 1, 20, 4)
radar_delay_ms = st.sidebar.slider("Radar Tracking Delay (ms)", 1, 20, 3)
ai_decision_delay_ms = st.sidebar.slider("AI Decision Delay (ms)", 1, 20, 2)
launcher_delay_ms = st.sidebar.slider("Launcher Activation Delay (ms)", 1, 30, 5)

st.sidebar.divider()
st.sidebar.header("ETC Launcher Inputs")

propellant_mass_g = st.sidebar.slider("Propellant Mass (g)", 20, 300, 120, 5)
electric_pulse_kj = st.sidebar.slider("Electric Pulse Energy (kJ)", 5, 120, 35, 5)
plasma_efficiency = st.sidebar.slider("Plasma Coupling Efficiency", 0.05, 0.60, 0.25, 0.01)
chamber_length_mm = st.sidebar.slider("Chamber Length (mm)", 40, 250, 120, 5)
chamber_diameter_mm = st.sidebar.slider("Chamber Diameter (mm)", 20, 80, 50, 2)
barrel_length_mm = st.sidebar.slider("Launcher / Barrel Length (mm)", 150, 1500, 500, 25)
interceptor_mass_g = st.sidebar.slider("Interceptor Mass (g)", 20, 300, 80, 5)
interceptor_diameter_mm = st.sidebar.slider("Interceptor Diameter (mm)", 10, 50, 25, 1)

st.sidebar.divider()
st.sidebar.header("APS Health Inputs")

radar_health = st.sidebar.slider("Radar Health (%)", 0, 100, 90)
launcher_health = st.sidebar.slider("Launcher Health (%)", 0, 100, 85)
sensor_health = st.sidebar.slider("Sensor Health (%)", 0, 100, 90)
available_launchers = st.sidebar.slider("Available DMA Launchers", 1, 8, 6)
remaining_countermeasures = st.sidebar.slider("Remaining Countermeasures", 0, 8, 4)
fragment_count = st.sidebar.slider("Fragment Count for Damage Model", 5, 40, 20)
critical_component_area = st.sidebar.slider("Critical Component Area (ft²)", 0.05, 5.0, 1.0, 0.05)


# ============================================================
# BASIC COMPUTATION FUNCTIONS
# ============================================================

def safe_clip(value, low, high):
    return max(low, min(value, high))


def calculate_etc_outputs(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm
):
    mp = propellant_mass_g / 1000.0
    Ei = electric_pulse_kj * 1000.0
    mi = interceptor_mass_g / 1000.0

    Lc = chamber_length_mm / 1000.0
    Dc = chamber_diameter_mm / 1000.0
    Lb = barrel_length_mm / 1000.0
    Di = interceptor_diameter_mm / 1000.0

    chamber_volume = np.pi * (Dc / 2) ** 2 * Lc
    bore_area = np.pi * (Di / 2) ** 2

    propellant_energy_density = 3.0e6
    ballistic_efficiency = 0.18
    gas_fraction = 0.85
    gas_R = 300.0
    gamma = 1.22

    chemical_energy = mp * propellant_energy_density
    coupled_electric_energy = Ei * plasma_efficiency
    total_effective_heat = chemical_energy + coupled_electric_energy

    gas_mass = max(mp * gas_fraction, 1e-6)

    cv = gas_R / (gamma - 1.0)
    temperature_K = 300.0 + total_effective_heat / (gas_mass * cv)
    temperature_K = safe_clip(temperature_K, 500.0, 6500.0)

    pressure_pa = (gas_mass * gas_R * temperature_K) / max(chamber_volume, 1e-9)
    pressure_mpa = safe_clip(pressure_pa / 1e6, 5.0, 450.0)

    chamber_to_barrel_volume = bore_area * Lb
    useful_work = min(
        total_effective_heat * ballistic_efficiency,
        pressure_pa * chamber_to_barrel_volume * 0.55
    )

    muzzle_velocity = np.sqrt(max(2.0 * useful_work / max(mi, 1e-6), 0.0))
    muzzle_velocity = safe_clip(muzzle_velocity, 50.0, 950.0)

    avg_acceleration = muzzle_velocity ** 2 / (2 * max(Lb, 1e-6))
    launch_acceleration_time = muzzle_velocity / max(avg_acceleration, 1e-6)

    return {
        "chemical_energy": chemical_energy,
        "coupled_electric_energy": coupled_electric_energy,
        "total_effective_heat": total_effective_heat,
        "chamber_volume": chamber_volume,
        "bore_area": bore_area,
        "temperature_K": temperature_K,
        "pressure_mpa": pressure_mpa,
        "muzzle_velocity": muzzle_velocity,
        "avg_acceleration": avg_acceleration,
        "launch_acceleration_time": launch_acceleration_time
    }


def predict_ai_threat(velocity, distance, angle, lethality):
    X = pd.DataFrame([[velocity, distance, angle, lethality]],
                     columns=["Velocity", "Distance", "Angle", "Lethality"])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0]
    return pred, proba


def calculate_reaction_time(sensor_ms, radar_ms, ai_ms, launcher_ms):
    return (sensor_ms + radar_ms + ai_ms + launcher_ms) / 1000.0


def calculate_pk(time_margin, ai_class, radar_health, launcher_health, sensor_health, remaining_countermeasures):
    health_factor = (radar_health / 100) * 0.35 + (launcher_health / 100) * 0.35 + (sensor_health / 100) * 0.20
    ammo_factor = min(remaining_countermeasures / 4, 1.0) * 0.10

    timing_factor = 1 / (1 + np.exp(-12 * time_margin))
    threat_factor = {0: 0.98, 1: 0.86, 2: 0.68}[ai_class]

    pk = (0.65 * timing_factor + health_factor + ammo_factor) * threat_factor
    return safe_clip(pk, 0.0, 0.98)


def markov_failure(fragment_count, component_area_ft2, system_health, total_area_ft2=80):
    survival = (1 - component_area_ft2 / total_area_ft2) ** fragment_count
    survival *= system_health
    failure = 1 - survival
    expected_encounters = 1 / max(failure, 1e-6)
    return survival, failure, expected_encounters


def select_dma_launcher(angle_deg, available_launchers):
    sectors = [
        "Front DMA",
        "Front-Left DMA",
        "Left DMA",
        "Rear-Left DMA",
        "Rear DMA",
        "Rear-Right DMA",
        "Right DMA",
        "Front-Right DMA"
    ]

    index = int((angle_deg / 360) * 8) % 8
    index = min(index, max(available_launchers - 1, 0))
    return sectors[index], index + 1


etc = calculate_etc_outputs(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm
)

ai_class, ai_proba = predict_ai_threat(
    velocity,
    distance,
    angle,
    base_threat["lethality"]
)

reaction_time = calculate_reaction_time(
    sensor_delay_ms,
    radar_delay_ms,
    ai_decision_delay_ms,
    launcher_delay_ms
)

threat_time_to_vehicle = distance / velocity
threat_time_to_interception_zone = max((distance - interception_range) / velocity, 0)
interceptor_flight_time = interception_range / max(etc["muzzle_velocity"], 1e-6)
total_interception_time = reaction_time + etc["launch_acceleration_time"] + interceptor_flight_time
time_margin = threat_time_to_vehicle - total_interception_time

pk = calculate_pk(
    time_margin,
    ai_class,
    radar_health,
    launcher_health,
    sensor_health,
    remaining_countermeasures
)

system_health_factor = (radar_health + launcher_health + sensor_health) / 300
surv_prob, fail_prob, expected_encounters = markov_failure(
    fragment_count,
    critical_component_area,
    system_health_factor
)

selected_dma, selected_dma_id = select_dma_launcher(angle, available_launchers)


# ============================================================
# MAIN DASHBOARD
# ============================================================

main_tabs = st.tabs([
    "1. Mission Dashboard",
    "2. 3D APS Engagement",
    "3. Multi-Threat AI",
    "4. ETC CFD-Style Results",
    "5. APS Damage & Health",
    "6. System Architecture"
])


# ============================================================
# TAB 1 — MISSION DASHBOARD
# ============================================================

with main_tabs[0]:
    st.markdown("## Mission-Level Decision Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AI Threat Level", THREAT_LABELS[ai_class])
    k2.metric("Selected DMA Launcher", f"{selected_dma_id}: {selected_dma}")
    k3.metric("Probability of Kill (Pk)", f"{pk * 100:.1f}%")
    k4.metric("Time Margin", f"{time_margin:.4f} s")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Muzzle Velocity", f"{etc['muzzle_velocity']:.1f} m/s")
    k6.metric("Peak Pressure", f"{etc['pressure_mpa']:.1f} MPa")
    k7.metric("Chamber Temp.", f"{etc['temperature_K']:.0f} K")
    k8.metric("Total Reaction Time", f"{reaction_time * 1000:.1f} ms")

    if pk >= 0.80 and time_margin > 0:
        st.success("APS Decision: ENGAGE — High probability of successful interception.")
    elif pk >= 0.55 and time_margin > 0:
        st.warning("APS Decision: ENGAGE — Moderate probability, tight engagement window.")
    else:
        st.error("APS Decision: HIGH RISK — Interception window insufficient or system health degraded.")

    st.markdown("### Final Output Table")

    final_df = pd.DataFrame({
        "Parameter": [
            "Threat Type",
            "Threat Velocity",
            "Initial Distance",
            "Incoming Angle",
            "Detection Range",
            "Interception Range",
            "Estimated Time-to-Impact",
            "Sensor + Radar + AI + Launcher Delay",
            "Interceptor Flight Time",
            "Total Interception Time",
            "AI Classified Threat Level",
            "Selected DMA Launcher",
            "Probability of Kill",
            "APS Failure Probability",
            "Expected Functional Encounters"
        ],
        "Value": [
            selected_threat,
            f"{velocity} m/s",
            f"{distance} m",
            f"{angle} degrees",
            f"{detection_range} m",
            f"{interception_range} m",
            f"{threat_time_to_vehicle:.4f} s",
            f"{reaction_time:.4f} s",
            f"{interceptor_flight_time:.4f} s",
            f"{total_interception_time:.4f} s",
            THREAT_LABELS[ai_class],
            f"{selected_dma_id}: {selected_dma}",
            f"{pk * 100:.2f}%",
            f"{fail_prob * 100:.2f}%",
            f"{expected_encounters:.2f}"
        ]
    })

    st.table(final_df)

    conf_df = pd.DataFrame({
        "Threat Class": ["LOW", "MEDIUM", "HIGH"],
        "AI Confidence (%)": [ai_proba[0] * 100, ai_proba[1] * 100, ai_proba[2] * 100]
    })

    st.markdown("### AI Confidence")
    st.bar_chart(conf_df.set_index("Threat Class"))


# ============================================================
# 3D HELPERS
# ============================================================

def cuboid(x0, x1, y0, y1, z0, z1, color="olive", opacity=1.0):
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 6]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 7, 2]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 1, 0, 2, 3]

    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity,
        flatshading=True,
        hoverinfo="skip"
    )


def cylinder_x(x0, x1, y, z, radius=0.12, color="darkolivegreen", n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    x, yy, zz = [], [], []
    for xi in [x0, x1]:
        x.extend([xi] * n)
        yy.extend(y + radius * np.cos(theta))
        zz.extend(z + radius * np.sin(theta))
    ii, jj, kk = [], [], []
    for a in range(n - 1):
        ii += [a, a + 1]
        jj += [a + 1, a + n + 1]
        kk += [a + n, a + n]
    return go.Mesh3d(x=x, y=yy, z=zz, i=ii, j=jj, k=kk, color=color, hoverinfo="skip")


def cylinder_y(x, y0, y1, z, radius=0.12, color="darkolivegreen", n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    xx, y, zz = [], [], []
    for yi in [y0, y1]:
        xx.extend(x + radius * np.cos(theta))
        y.extend([yi] * n)
        zz.extend(z + radius * np.sin(theta))
    ii, jj, kk = [], [], []
    for a in range(n - 1):
        ii += [a, a + 1]
        jj += [a + 1, a + n + 1]
        kk += [a + n, a + n]
    return go.Mesh3d(x=xx, y=y, z=zz, i=ii, j=jj, k=kk, color=color, hoverinfo="skip")


def range_ring(radius, z, color, width=6):
    t = np.linspace(0, 2 * np.pi, 220)
    return go.Scatter3d(
        x=radius * np.cos(t),
        y=radius * np.sin(t),
        z=np.full_like(t, z),
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip"
    )


def create_mbt():
    parts = []
    parts.append(cuboid(-4.5, 4.5, -2.0, 2.0, 0.35, 1.20, "darkolivegreen"))
    parts.append(cuboid(-5.2, -4.2, -1.7, 1.7, 0.35, 0.95, "olivedrab"))
    parts.append(cuboid(2.7, 4.7, -1.7, 1.7, 1.20, 1.50, "darkkhaki"))
    parts.append(cuboid(-1.8, 1.8, -1.25, 1.25, 1.20, 2.05, "olive"))
    parts.append(cuboid(-1.0, 1.0, -0.8, 0.8, 2.05, 2.35, "darkolivegreen"))
    parts.append(cuboid(-2.3, -1.65, -0.5, 0.5, 1.50, 2.00, "darkkhaki"))
    parts.append(cylinder_x(-2.3, -8.4, 0, 1.75, 0.14, "darkolivegreen"))
    parts.append(cuboid(-4.8, 4.7, -2.55, -2.05, 0.0, 0.65, "dimgray"))
    parts.append(cuboid(-4.8, 4.7, 2.05, 2.55, 0.0, 0.65, "dimgray"))

    for xw in np.linspace(-3.8, 3.8, 7):
        parts.append(cylinder_y(xw, -2.65, -2.58, 0.32, 0.28, "gray"))
        parts.append(cylinder_y(xw, 2.58, 2.65, 0.32, 0.28, "gray"))

    dma = [
        (-4.4, -2.35, 1.45),
        (-4.4, 2.35, 1.45),
        (0.0, -2.35, 1.55),
        (0.0, 2.35, 1.55),
        (4.2, -2.35, 1.45),
        (4.2, 2.35, 1.45),
        (0.1, 0.0, 2.58),
        (-1.3, 0.0, 2.45)
    ]

    for idx, (lx, ly, lz) in enumerate(dma[:available_launchers]):
        color = "yellow" if idx + 1 == selected_dma_id else "gold"
        parts.append(cylinder_y(lx, ly - 0.20, ly + 0.20, lz, 0.16, color))

    return parts, np.array(dma[:available_launchers])


def missile_marker(x, y, z, color="red"):
    return go.Cone(
        x=[x], y=[y], z=[z],
        u=[0.8], v=[0.0], w=[0.0],
        sizemode="absolute", sizeref=0.55,
        anchor="tip",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo="skip"
    )


def interceptor_marker(x, y, z):
    return go.Cone(
        x=[x], y=[y], z=[z],
        u=[-0.7], v=[0.25], w=[0.05],
        sizemode="absolute", sizeref=0.45,
        anchor="tip",
        colorscale=[[0, "lime"], [1, "lime"]],
        showscale=False,
        hoverinfo="skip"
    )


# ============================================================
# TAB 2 — 3D APS ENGAGEMENT
# ============================================================

with main_tabs[1]:
    st.markdown("## 3D APS Engagement with DMA Selection")

    scale = 25.0
    det_r = detection_range / scale
    int_r = interception_range / scale

    theta = np.radians(angle)
    start = np.array([-distance / scale, -np.tan(theta) * (distance / scale) * 0.45, 2.2 + (angle / 80) * 3.0])
    tank_center = np.array([0.0, 0.0, 1.45])
    direction = tank_center - start
    direction = direction / np.linalg.norm(direction)
    intercept_point = tank_center - direction * int_r

    frames_n = 45
    missile_path = np.linspace(start, tank_center, frames_n)

    tank_parts, dma_positions = create_mbt()
    launcher_point = dma_positions[min(selected_dma_id - 1, len(dma_positions) - 1)]
    interceptor_path = np.linspace(launcher_point, intercept_point, frames_n)

    intercept_idx = int(np.argmin(np.linalg.norm(missile_path - intercept_point, axis=1)))

    ground_x, ground_y = np.meshgrid(np.linspace(-18, 10, 2), np.linspace(-10, 10, 2))
    ground_z = np.zeros_like(ground_x) - 0.04

    ground = go.Surface(
        x=ground_x, y=ground_y, z=ground_z,
        colorscale=[[0, "rgb(190,175,145)"], [1, "rgb(190,175,145)"]],
        opacity=0.45, showscale=False, hoverinfo="skip"
    )

    threat_line = go.Scatter3d(
        x=missile_path[:, 0], y=missile_path[:, 1], z=missile_path[:, 2],
        mode="lines", line=dict(color="red", width=6), name="Threat Path"
    )

    interceptor_line = go.Scatter3d(
        x=[launcher_point[0], intercept_point[0]],
        y=[launcher_point[1], intercept_point[1]],
        z=[launcher_point[2], intercept_point[2]],
        mode="lines", line=dict(color="lime", width=7, dash="dash"), name="Interceptor Path"
    )

    intercept_label = go.Scatter3d(
        x=[intercept_point[0]], y=[intercept_point[1]], z=[intercept_point[2]],
        mode="markers+text",
        marker=dict(size=7, color="lime"),
        text=["Neutralization Point"],
        textposition="top center",
        hoverinfo="skip"
    )

    base_data = [
        ground,
        range_ring(det_r, 0.04, "cyan", 6),
        range_ring(int_r, 0.08, "lime", 7),
    ] + tank_parts + [
        threat_line, interceptor_line, intercept_label,
        missile_marker(missile_path[0, 0], missile_path[0, 1], missile_path[0, 2]),
        interceptor_marker(launcher_point[0], launcher_point[1], launcher_point[2])
    ]

    frames = []
    for i in range(frames_n):
        idx = min(i, intercept_idx)
        mpos = missile_path[idx]
        ipos = interceptor_path[min(i, frames_n - 1)]
        if i >= intercept_idx:
            mpos = intercept_point
            ipos = intercept_point
            mcolor = "orange"
        else:
            mcolor = "red"

        frames.append(
            go.Frame(
                data=[missile_marker(mpos[0], mpos[1], mpos[2], mcolor), interceptor_marker(ipos[0], ipos[1], ipos[2])],
                traces=[len(base_data)-2, len(base_data)-1],
                name=str(i)
            )
        )

    fig = go.Figure(data=base_data, frames=frames)
    fig.update_layout(
        title="Threat Detection → DMA Launcher Selection → Interception",
        height=760,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="Forward Direction", range=[-18, 10], showgrid=False, showbackground=False),
            yaxis=dict(title="Lateral Direction", range=[-10, 10], showgrid=False, showbackground=False),
            zaxis=dict(title="Height", range=[0, 9], showgrid=False, showbackground=False),
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.0, z=0.55),
            camera=dict(eye=dict(x=-1.85, y=-1.35, z=0.85), center=dict(x=-0.15, y=0.0, z=-0.15))
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.98,
                buttons=[
                    dict(
                        label="▶ Play Animation",
                        method="animate",
                        args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                    )
                ]
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Cyan ring = detection range, green ring = interception range, gold/yellow modules = DMA launcher architecture."
    )


# ============================================================
# TAB 3 — MULTI-THREAT AI
# ============================================================

with main_tabs[2]:
    st.markdown("## Multi-Threat Engagement and AI Prioritization")

    if mode == "Single Threat":
        st.info("Switch Engagement Mode to Multi-Threat in the sidebar to activate multi-threat scenario.")
    else:
        np.random.seed(7)

        threat_list = []
        names = list(THREAT_DB.keys())

        for i in range(3):
            name = np.random.choice(names)
            base = THREAT_DB[name]
            v = safe_clip(base["velocity"] * np.random.uniform(0.85, 1.20), 80, 900)
            d = np.random.uniform(80, 420)
            a = np.random.uniform(0, 80)
            leth = base["lethality"]
            cls, proba = predict_ai_threat(v, d, a, leth)
            tti = d / v

            priority_score = (v / max(d, 1)) * 0.45 + (1 / max(tti, 0.01)) * 0.35 + leth * 0.20
            threat_list.append([f"Threat-{i+1}", name, v, d, a, tti, THREAT_LABELS[cls], priority_score])

        multi_df = pd.DataFrame(
            threat_list,
            columns=["ID", "Type", "Velocity", "Distance", "Angle", "TTI", "AI Class", "Priority Score"]
        )

        multi_df = multi_df.sort_values("Priority Score", ascending=False).reset_index(drop=True)
        st.table(multi_df)

        st.success(f"AI Priority Decision: Engage {multi_df.loc[0, 'ID']} first ({multi_df.loc[0, 'Type']}).")


# ============================================================
# TAB 4 — ETC CFD STYLE RESULTS
# ============================================================

with main_tabs[3]:
    st.markdown("## ETC CFD-Style Curves and Contours")

    t = np.linspace(0, 5.0, 160)
    peak_t = 1.0 + 0.4 * (1 - plasma_efficiency)
    width = 0.55 + 0.25 * (propellant_mass_g / 300)

    pressure_curve = etc["pressure_mpa"] * np.exp(-((t - peak_t) / width) ** 2)
    temperature_curve = 300 + (etc["temperature_K"] - 300) * np.exp(-((t - peak_t) / (width * 1.15)) ** 2)
    velocity_curve = etc["muzzle_velocity"] * (1 - np.exp(-t / 1.25))
    velocity_curve = np.minimum(velocity_curve, etc["muzzle_velocity"])

    curve_tabs = st.tabs(["Pressure Curve", "Temperature Curve", "Velocity Curve"])

    for tab, title, y, ylabel in [
        (curve_tabs[0], "Pressure vs Time", pressure_curve, "Pressure (MPa)"),
        (curve_tabs[1], "Temperature vs Time", temperature_curve, "Temperature (K)"),
        (curve_tabs[2], "Velocity vs Time", velocity_curve, "Velocity (m/s)")
    ]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(width=4)))
            fig.update_layout(height=370, title=title, xaxis_title="Time (ms)", yaxis_title=ylabel, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Contour Results")

    Lc_mm = chamber_length_mm
    Lb_mm = barrel_length_mm
    total_len = Lc_mm + Lb_mm
    chamber_r = chamber_diameter_mm / 2
    barrel_r = interceptor_diameter_mm / 2

    nx, ny = 220, 90
    x = np.linspace(0, total_len, nx)
    y = np.linspace(0, chamber_r, ny)
    X, Y = np.meshgrid(x, y)

    transition_start = Lc_mm * 0.82
    transition_end = Lc_mm

    R = np.where(
        X < transition_start,
        chamber_r,
        np.where(
            X < transition_end,
            chamber_r - (chamber_r - barrel_r) * ((X - transition_start) / max(transition_end - transition_start, 1e-6)),
            barrel_r
        )
    )

    inside = Y <= R
    xn = X / max(total_len, 1e-6)

    radial_factor = np.exp(-(Y / np.maximum(R, 1e-6)) ** 2 * 0.8)

    P_field = etc["pressure_mpa"] * np.exp(-2.7 * xn) * radial_factor
    T_field = 300 + (etc["temperature_K"] - 300) * np.exp(-3.3 * xn) * np.exp(-(Y / (0.72 * np.maximum(R, 1e-6))) ** 2)
    V_field = etc["muzzle_velocity"] * (1 - np.exp(-3.2 * xn)) * (0.65 + 0.35 * (1 - (Y / np.maximum(R, 1e-6)) ** 2))

    P_field = np.where(inside, P_field, np.nan)
    T_field = np.where(inside, T_field, np.nan)
    V_field = np.where(inside, V_field, np.nan)

    contour_tabs = st.tabs(["Pressure Contour", "Temperature Contour", "Velocity Contour"])

    for tab, title, field, unit, scale in [
        (contour_tabs[0], "Pressure Contour", P_field, "MPa", "Jet"),
        (contour_tabs[1], "Temperature Contour", T_field, "K", "Hot"),
        (contour_tabs[2], "Velocity Contour", V_field, "m/s", "Viridis")
    ]:
        with tab:
            figc = go.Figure()
            figc.add_trace(go.Contour(x=x, y=y, z=field, colorscale=scale, colorbar=dict(title=unit)))
            figc.update_layout(
                height=460,
                title=title,
                xaxis_title="Axial Length (mm)",
                yaxis_title="Radius (mm)",
                template="plotly_white"
            )
            st.plotly_chart(figc, use_container_width=True)

    st.markdown("### ETC Internal Ballistic Output Table")
    st.table(pd.DataFrame({
        "Parameter": [
            "Propellant Mass",
            "Electric Pulse Energy",
            "Plasma Coupling Efficiency",
            "Peak Pressure",
            "Peak Temperature",
            "Muzzle Velocity",
            "Average Acceleration",
            "Launcher Acceleration Time"
        ],
        "Value": [
            f"{propellant_mass_g} g",
            f"{electric_pulse_kj} kJ",
            f"{plasma_efficiency:.2f}",
            f"{etc['pressure_mpa']:.2f} MPa",
            f"{etc['temperature_K']:.0f} K",
            f"{etc['muzzle_velocity']:.2f} m/s",
            f"{etc['avg_acceleration']:.2f} m/s²",
            f"{etc['launch_acceleration_time']:.5f} s"
        ]
    }))


# ============================================================
# TAB 5 — APS DAMAGE & HEALTH
# ============================================================

with main_tabs[4]:
    st.markdown("## APS Damage, Failure and System Health Dashboard")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Radar Health", f"{radar_health}%")
    h2.metric("Launcher Health", f"{launcher_health}%")
    h3.metric("Sensor Health", f"{sensor_health}%")
    h4.metric("Countermeasures", remaining_countermeasures)

    st.markdown("### Markov-Based Failure Prediction")

    st.table(pd.DataFrame({
        "Parameter": [
            "Fragment Count",
            "Critical Component Area",
            "System Health Factor",
            "Survival Probability",
            "Failure Probability",
            "Expected Functional Encounters"
        ],
        "Value": [
            fragment_count,
            f"{critical_component_area:.2f} ft²",
            f"{system_health_factor:.2f}",
            f"{surv_prob:.4f}",
            f"{fail_prob:.4f}",
            f"{expected_encounters:.2f}"
        ]
    }))

    if fail_prob > 0.6:
        st.error("Battle Damage Assessment: HIGH FAILURE RISK")
    elif fail_prob > 0.3:
        st.warning("Battle Damage Assessment: MODERATE FAILURE RISK")
    else:
        st.success("Battle Damage Assessment: ACCEPTABLE SURVIVABILITY")


# ============================================================
# TAB 6 — ARCHITECTURE
# ============================================================

with main_tabs[5]:
    st.markdown("## System Architecture Page")

    st.code(
        """
Incoming Threat
      ↓
Sensor Detection
      ↓
Radar Tracking
      ↓
AI Threat Classification
      ↓
Trajectory Prediction
      ↓
DMA Launcher Selection
      ↓
ETC Launcher Firing
      ↓
Interceptor Flight
      ↓
Threat Neutralization
      ↓
Battle Damage Assessment
        """,
        language="text"
    )

    st.markdown("### Thesis-Level Description")

    st.write("""
This platform integrates simplified ETC internal ballistic modelling with AI-assisted APS engagement logic.
The ETC module estimates chamber pressure, temperature, muzzle velocity and launcher acceleration time.
The AI module classifies threats, prioritizes multi-threat scenarios and selects a suitable DMA launcher module.
The engagement module estimates reaction time, interceptor flight time, probability of kill and final time margin.
The survivability module evaluates APS damage probability using a Markov-based failure model.

Together, the framework demonstrates a system-level simulation environment for next-generation combat vehicle
survivability studies, combining CFD-inspired internal ballistics, AI decision support, DMA architecture and
battle damage assessment in a single interactive dashboard.
    """)

    st.markdown("### Key Real-World Applications")

    st.write("""
1. Conceptual evaluation of APS reaction-time requirements.
2. Comparison of threat types based on time-to-impact and severity.
3. Preliminary study of ETC launcher performance trends.
4. Visualization of DMA launcher architecture for 360-degree coverage.
5. AI-assisted threat prioritization for multi-threat conditions.
6. Survivability assessment after fragment exposure.
    """)
