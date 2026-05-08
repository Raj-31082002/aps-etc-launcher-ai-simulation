import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="APS ETC Digital Twin",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# DARK UI STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07110d 0%, #101c16 45%, #050707 100%);
        color: #e8f5e9;
    }
    .block-container {
        padding-top: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(20, 40, 30, 0.82);
        border: 1px solid rgba(80, 255, 160, 0.25);
        border-radius: 16px;
        padding: 14px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(18, 35, 28, 0.9);
        border-radius: 12px;
        padding: 10px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ AI-Assisted APS–ETC Digital Twin with 3D Interception Visualization")
st.caption(
    "Clean 3D APS engagement animation with radar sweep, missile/interceptor differentiation, "
    "neutralization burst, reaction time, slew-rate comparison, contours, and downloadable reports."
)

st.warning(
    "Academic simplified model for thesis demonstration. Not a validated weapon-design or operational fire-control solver."
)

# ============================================================
# THREAT DATABASE
# ============================================================

THREAT_DB = {
    "RPG": {"velocity": 120, "distance": 180, "angle": 20, "lethality": 0.55},
    "HEAT / HESH": {"velocity": 200, "distance": 180, "angle": 25, "lethality": 0.70},
    "ATGM": {"velocity": 280, "distance": 250, "angle": 35, "lethality": 0.85},
    "Top-Attack Munition": {"velocity": 300, "distance": 220, "angle": 70, "lethality": 0.90},
    "FSAPDS / KE": {"velocity": 700, "distance": 300, "angle": 10, "lethality": 0.95},
}

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Threat Inputs")

threat_type = st.sidebar.selectbox("Threat Type", list(THREAT_DB.keys()), index=1)
base = THREAT_DB[threat_type]

velocity = st.sidebar.slider("Threat Velocity (m/s)", 80, 900, int(base["velocity"]), 10)
distance = st.sidebar.slider("Initial Threat Distance (m)", 20, 500, int(base["distance"]), 10)
angle = st.sidebar.slider("Incoming Angle / Slew Angle (deg)", 0, 90, int(base["angle"]), 1)

detection_range = st.sidebar.slider("Detection Range (m)", 50, 800, 350, 10)
interception_range = st.sidebar.slider("Interception Range (m)", 20, 250, 120, 5)

st.sidebar.header("ETC Launcher Inputs")

propellant_mass_g = st.sidebar.slider("Propellant Mass (g)", 20, 300, 120, 5)
electric_pulse_kj = st.sidebar.slider("Electric Pulse Energy (kJ)", 5, 120, 35, 5)
plasma_efficiency = st.sidebar.slider("Plasma Coupling Efficiency", 0.05, 0.60, 0.25, 0.01)
chamber_length_mm = st.sidebar.slider("Chamber Length (mm)", 40, 250, 120, 5)
chamber_diameter_mm = st.sidebar.slider("Chamber Diameter (mm)", 20, 80, 50, 2)
barrel_length_mm = st.sidebar.slider("Launcher / Barrel Length (mm)", 150, 1500, 500, 25)
interceptor_mass_g = st.sidebar.slider("Interceptor Mass (g)", 20, 300, 80, 5)
interceptor_diameter_mm = st.sidebar.slider("Interceptor Diameter (mm)", 10, 50, 25, 1)

st.sidebar.header("Reaction + Slew Inputs")

sensor_delay_ms = st.sidebar.slider("Sensor Detection Delay (ms)", 1, 30, 4)
radar_delay_ms = st.sidebar.slider("Radar Tracking Delay (ms)", 1, 30, 3)
ai_delay_ms = st.sidebar.slider("AI Decision Delay (ms)", 1, 30, 2)
etc_activation_delay_ms = st.sidebar.slider("ETC Puck Activation Delay (ms)", 1, 30, 5)
conventional_ignition_delay_ms = st.sidebar.slider("Conventional Ignition Delay (ms)", 5, 80, 25)

conventional_slew_rate_dps = st.sidebar.slider("Conventional Slew Rate (deg/s)", 20, 300, 90, 10)
etc_dma_slew_rate_dps = st.sidebar.slider("ETC-DMA Puck Slew Rate (deg/s)", 300, 2000, 1000, 50)
etc_effective_slew_angle_deg = st.sidebar.slider("ETC-DMA Effective Micro-Slew Angle (deg)", 0, 10, 2, 1)

st.sidebar.header("System Health")

radar_health = st.sidebar.slider("Radar Health (%)", 0, 100, 90)
sensor_health = st.sidebar.slider("Sensor Health (%)", 0, 100, 90)
launcher_health = st.sidebar.slider("Launcher Health (%)", 0, 100, 85)
thermal_load = st.sidebar.slider("Thermal Load (%)", 0, 100, 45)
available_launchers = st.sidebar.slider("Available DMA Launchers", 1, 8, 6)
remaining_countermeasures = st.sidebar.slider("Remaining Countermeasures", 0, 8, 4)

st.sidebar.header("Graphics Controls")

camera_view = st.sidebar.selectbox(
    "Camera View",
    ["Isometric", "Side View", "Top View", "Interceptor View"],
    index=0
)

show_detection = st.sidebar.checkbox("Show Detection Range", True)
show_interception = st.sidebar.checkbox("Show Interception Range", True)
show_radar_sweep = st.sidebar.checkbox("Show Radar Sweep", True)
show_trajectories = st.sidebar.checkbox("Show Trajectories", True)
show_labels = st.sidebar.checkbox("Show Labels", True)
show_exhaust = st.sidebar.checkbox("Show Exhaust Trail", True)

animation_frames = st.sidebar.slider("Animation Smoothness", 35, 90, 55, 5)

# ============================================================
# CORE MODELS
# ============================================================

def clip(v, lo, hi):
    return max(lo, min(v, hi))

def threat_ai_score(v, d, a, lethality):
    tti = d / max(v, 1e-6)
    score = 0
    score += min(v / 700, 1.5) * 0.35
    score += min(1 / max(tti, 0.01), 20) / 20 * 0.30
    score += min(a / 75, 1.2) * 0.20
    score += lethality * 0.15

    if score >= 0.72:
        return 2, score
    elif score >= 0.42:
        return 1, score
    return 0, score

def class_name(c):
    return ["LOW THREAT", "MEDIUM THREAT", "HIGH THREAT"][c]

def calculate_etc(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm
):
    mp = propellant_mass_g / 1000
    Ei = electric_pulse_kj * 1000
    mi = interceptor_mass_g / 1000

    Lc = chamber_length_mm / 1000
    Dc = chamber_diameter_mm / 1000
    Lb = barrel_length_mm / 1000
    Di = interceptor_diameter_mm / 1000

    chamber_volume = np.pi * (Dc / 2) ** 2 * Lc
    bore_area = np.pi * (Di / 2) ** 2

    propellant_energy_density = 3.0e6
    ballistic_efficiency = 0.18
    gas_fraction = 0.85
    gas_R = 300
    gamma = 1.22

    chemical_energy = mp * propellant_energy_density
    electric_energy_coupled = Ei * plasma_efficiency
    total_energy = chemical_energy + electric_energy_coupled

    gas_mass = max(mp * gas_fraction, 1e-6)
    cv = gas_R / (gamma - 1)

    temp_K = 300 + total_energy / (gas_mass * cv)
    temp_K = clip(temp_K, 500, 6500)

    pressure_pa = (gas_mass * gas_R * temp_K) / max(chamber_volume, 1e-9)
    pressure_mpa = clip(pressure_pa / 1e6, 5, 450)

    barrel_volume = bore_area * Lb
    useful_work = min(total_energy * ballistic_efficiency, pressure_pa * barrel_volume * 0.55)

    muzzle_velocity = np.sqrt(max(2 * useful_work / max(mi, 1e-6), 0))
    muzzle_velocity = clip(muzzle_velocity, 50, 950)

    avg_acceleration = muzzle_velocity ** 2 / (2 * max(Lb, 1e-6))
    launch_accel_time = muzzle_velocity / max(avg_acceleration, 1e-6)

    return {
        "chemical_energy": chemical_energy,
        "electric_energy_coupled": electric_energy_coupled,
        "total_energy": total_energy,
        "pressure_mpa": pressure_mpa,
        "temp_K": temp_K,
        "muzzle_velocity": muzzle_velocity,
        "avg_acceleration": avg_acceleration,
        "launch_accel_time": launch_accel_time,
        "chamber_volume": chamber_volume,
        "bore_area": bore_area
    }

def reaction_time_s(sensor_ms, radar_ms, ai_ms, launcher_ms):
    return (sensor_ms + radar_ms + ai_ms + launcher_ms) / 1000

def slew_time_s(required_angle_deg, slew_rate_deg_s):
    return required_angle_deg / max(slew_rate_deg_s, 1e-6)

def pk_model(time_margin, threat_class, radar_h, sensor_h, launcher_h, ammo, thermal):
    timing_factor = 1 / (1 + np.exp(-12 * time_margin))
    health_factor = (radar_h/100)*0.25 + (sensor_h/100)*0.20 + (launcher_h/100)*0.30
    ammo_factor = min(ammo / 4, 1) * 0.15
    thermal_penalty = (thermal / 100) * 0.20
    threat_factor = [0.98, 0.86, 0.65][threat_class]
    pk = (0.70 * timing_factor + health_factor + ammo_factor - thermal_penalty) * threat_factor
    return clip(pk, 0, 0.98)

def selected_dma(angle_deg, available):
    sectors = ["Front", "Front-Left", "Left", "Rear-Left", "Rear", "Rear-Right", "Right", "Front-Right"]
    idx = int((angle_deg / 360) * 8) % 8
    idx = min(idx, available - 1)
    return idx + 1, sectors[idx]

def compare_pyro_etc(etc_out):
    pyro_velocity = etc_out["muzzle_velocity"] * 0.62
    pyro_pressure = etc_out["pressure_mpa"] * 0.55
    pyro_temp = etc_out["temp_K"] * 0.70
    return pyro_velocity, pyro_pressure, pyro_temp

# ============================================================
# MAIN CALCULATIONS
# ============================================================

etc = calculate_etc(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm
)

pyro_velocity, pyro_pressure, pyro_temp = compare_pyro_etc(etc)

threat_class, score = threat_ai_score(velocity, distance, angle, base["lethality"])

tti = distance / velocity

etc_base_reaction = reaction_time_s(sensor_delay_ms, radar_delay_ms, ai_delay_ms, etc_activation_delay_ms)
conventional_base_reaction = reaction_time_s(sensor_delay_ms, radar_delay_ms, ai_delay_ms, conventional_ignition_delay_ms)

etc_slew_time = slew_time_s(etc_effective_slew_angle_deg, etc_dma_slew_rate_dps)
conventional_slew_time = slew_time_s(angle, conventional_slew_rate_dps)

etc_interceptor_flight = interception_range / etc["muzzle_velocity"]
conventional_interceptor_flight = interception_range / max(pyro_velocity, 1e-6)

etc_total_reaction = etc_base_reaction + etc_slew_time
conventional_total_reaction = conventional_base_reaction + conventional_slew_time

etc_total_interception = etc_total_reaction + etc["launch_accel_time"] + etc_interceptor_flight
conventional_total_interception = conventional_total_reaction + (etc["launch_accel_time"] * 1.6) + conventional_interceptor_flight

etc_time_margin = tti - etc_total_interception
conventional_time_margin = tti - conventional_total_interception

etc_pk = pk_model(etc_time_margin, threat_class, radar_health, sensor_health, launcher_health, remaining_countermeasures, thermal_load)
conventional_pk = pk_model(conventional_time_margin, threat_class, radar_health, sensor_health, launcher_health, remaining_countermeasures, thermal_load + 8)

dma_id, dma_sector = selected_dma(angle, available_launchers)

# ============================================================
# HEADER METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)
m1.metric("Threat Level", class_name(threat_class))
m2.metric("ETC Pk", f"{etc_pk*100:.1f}%")
m3.metric("Selected DMA", f"{dma_id}: {dma_sector}")
m4.metric("ETC Reaction Time", f"{etc_total_reaction*1000:.1f} ms")

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Mission Dashboard",
    "Clean 3D Simulation",
    "Reaction + Slew Comparison",
    "Tactical Top View",
    "ETC Curves + Contours",
    "Download Report"
])

# ============================================================
# TAB 1
# ============================================================

with tabs[0]:
    st.subheader("Mission Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time-to-Impact", f"{tti:.4f} s")
    c2.metric("ETC Interception Time", f"{etc_total_interception:.4f} s")
    c3.metric("ETC Time Margin", f"{etc_time_margin:.4f} s")
    c4.metric("Muzzle Velocity", f"{etc['muzzle_velocity']:.1f} m/s")

    if etc_pk >= 0.80 and etc_time_margin > 0:
        st.success("ETC-DMA Decision: ENGAGE — High success probability.")
    elif etc_pk >= 0.55 and etc_time_margin > 0:
        st.warning("ETC-DMA Decision: ENGAGE — Moderate probability, tight window.")
    else:
        st.error("ETC-DMA Decision: HIGH RISK — Timing or system condition is weak.")

# ============================================================
# 3D HELPERS
# ============================================================

def cuboid(x0, x1, y0, y1, z0, z1, color="olive", opacity=1):
    x = [x0,x1,x1,x0,x0,x1,x1,x0]
    y = [y0,y0,y1,y1,y0,y0,y1,y1]
    z = [z0,z0,z0,z0,z1,z1,z1,z1]
    i = [0,0,0,1,1,2,4,4,5,5,6,6]
    j = [1,2,4,2,5,3,5,6,6,1,7,2]
    k = [2,3,5,5,6,7,6,7,1,0,2,3]
    return go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,color=color,opacity=opacity,flatshading=True,hoverinfo="skip")

def cylinder_x(x0, x1, y, z, radius=0.12, color="darkolivegreen", n=20):
    theta = np.linspace(0, 2*np.pi, n)
    x, yy, zz = [], [], []
    for xi in [x0, x1]:
        x.extend([xi]*n)
        yy.extend(y + radius*np.cos(theta))
        zz.extend(z + radius*np.sin(theta))
    ii, jj, kk = [], [], []
    for a in range(n-1):
        ii += [a, a+1]
        jj += [a+1, a+n+1]
        kk += [a+n, a+n]
    return go.Mesh3d(x=x,y=yy,z=zz,i=ii,j=jj,k=kk,color=color,hoverinfo="skip")

def ring(radius, z, color, width=6):
    t = np.linspace(0, 2*np.pi, 250)
    return go.Scatter3d(
        x=radius*np.cos(t),
        y=radius*np.sin(t),
        z=np.full_like(t,z),
        mode="lines",
        line=dict(color=color,width=width),
        hoverinfo="skip"
    )

def create_tank():
    parts = []
    parts.append(cuboid(-4.5,4.5,-2,2,0.35,1.2,"darkolivegreen"))
    parts.append(cuboid(-1.8,1.8,-1.25,1.25,1.2,2.05,"olive"))
    parts.append(cuboid(-2.3,-1.65,-0.5,0.5,1.5,2.0,"darkkhaki"))
    parts.append(cylinder_x(-8.4,-2.3,0,1.75,0.12,"darkolivegreen"))
    parts.append(cuboid(-4.8,4.7,-2.55,-2.05,0,0.65,"dimgray"))
    parts.append(cuboid(-4.8,4.7,2.05,2.55,0,0.65,"dimgray"))

    dma = [
        (-4.4,-2.35,1.45),
        (-4.4,2.35,1.45),
        (0,-2.35,1.55),
        (0,2.35,1.55),
        (4.2,-2.35,1.45),
        (4.2,2.35,1.45),
        (0.1,0,2.58),
        (-1.3,0,2.45)
    ]

    for idx, p in enumerate(dma[:available_launchers]):
        col = "yellow" if idx+1 == dma_id else "gold"
        parts.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers",
            marker=dict(size=7,color=col),
            hoverinfo="skip"
        ))

    return parts, np.array(dma[:available_launchers])

def missile_body(x, y, z, color="red"):
    traces = []
    traces.append(go.Cone(
        x=[x], y=[y], z=[z],
        u=[0.8], v=[0], w=[0],
        sizemode="absolute", sizeref=0.48,
        colorscale=[[0,color],[1,color]],
        showscale=False,
        hoverinfo="skip"
    ))
    traces.append(cuboid(x-0.9, x-0.15, y-0.08, y+0.08, z-0.08, z+0.08, color, 1))
    traces.append(cuboid(x-1.0, x-0.75, y-0.30, y-0.20, z-0.04, z+0.04, color, 1))
    traces.append(cuboid(x-1.0, x-0.75, y+0.20, y+0.30, z-0.04, z+0.04, color, 1))
    traces.append(cuboid(x-1.0, x-0.75, y-0.04, y+0.04, z+0.20, z+0.30, color, 1))
    return traces

def interceptor_body(x, y, z):
    return [
        go.Cone(
            x=[x], y=[y], z=[z],
            u=[-0.7], v=[0.25], w=[0.05],
            sizemode="absolute", sizeref=0.40,
            colorscale=[[0,"lime"],[1,"lime"]],
            showscale=False,
            hoverinfo="skip"
        ),
        cuboid(x+0.12, x+0.55, y-0.06, y+0.06, z-0.06, z+0.06, "lime", 1)
    ]

def burst_cloud(x, y, z):
    pts = []
    for r, col, size in [(0.15, "yellow", 5), (0.35, "orange", 4), (0.55, "red", 3)]:
        theta = np.linspace(0, 2*np.pi, 18)
        phi = np.linspace(0, np.pi, 10)
        theta, phi = np.meshgrid(theta, phi)
        xx = x + r*np.sin(phi)*np.cos(theta)
        yy = y + r*np.sin(phi)*np.sin(theta)
        zz = z + r*np.cos(phi)
        pts.append(go.Scatter3d(
            x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
            mode="markers",
            marker=dict(size=size, color=col, opacity=0.7),
            hoverinfo="skip"
        ))
    return pts

def exhaust_trail(path, upto, color):
    if upto <= 2:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines")
    start = max(0, upto-10)
    return go.Scatter3d(
        x=path[start:upto,0],
        y=path[start:upto,1],
        z=path[start:upto,2],
        mode="lines",
        line=dict(color=color, width=5),
        opacity=0.55,
        hoverinfo="skip"
    )

def radar_sweep_line(radius, frame_idx):
    ang = np.radians((frame_idx * 10) % 360)
    return go.Scatter3d(
        x=[0, radius*np.cos(ang)],
        y=[0, radius*np.sin(ang)],
        z=[0.12, 0.12],
        mode="lines",
        line=dict(color="cyan", width=5),
        hoverinfo="skip"
    )

def camera_dict(view):
    if view == "Top View":
        return dict(eye=dict(x=0, y=0, z=2.3), center=dict(x=0, y=0, z=0))
    if view == "Side View":
        return dict(eye=dict(x=-2.2, y=0.1, z=0.7), center=dict(x=-0.15, y=0, z=-0.1))
    if view == "Interceptor View":
        return dict(eye=dict(x=-1.2, y=-2.0, z=0.8), center=dict(x=-0.2, y=0, z=-0.1))
    return dict(eye=dict(x=-1.85, y=-1.35, z=0.85), center=dict(x=-0.15, y=0, z=-0.15))

def build_3d_figure():
    scale = 25
    det_r = detection_range / scale
    int_r = interception_range / scale

    theta = np.radians(angle)
    start = np.array([
        -distance/scale,
        -np.tan(theta)*(distance/scale)*0.45,
        2.2 + (angle/90)*3
    ])

    center = np.array([0,0,1.45])
    direction = (center-start) / np.linalg.norm(center-start)
    intercept = center - direction * int_r

    n = animation_frames
    missile_path = np.linspace(start, center, n)

    tank, dma_pos = create_tank()
    launch_pt = dma_pos[min(dma_id-1, len(dma_pos)-1)]
    interceptor_path = np.linspace(launch_pt, intercept, n)
    intercept_idx = int(np.argmin(np.linalg.norm(missile_path-intercept, axis=1)))

    gx, gy = np.meshgrid(np.linspace(-18,10,2), np.linspace(-10,10,2))
    ground = go.Surface(
        x=gx, y=gy, z=np.zeros_like(gx)-0.04,
        colorscale=[[0,"rgb(45,43,35)"],[1,"rgb(45,43,35)"]],
        opacity=0.75,
        showscale=False,
        hoverinfo="skip"
    )

    base_data = [ground] + tank

    if show_detection:
        base_data.append(ring(det_r,0.04,"cyan",5))
    if show_interception:
        base_data.append(ring(int_r,0.08,"lime",6))
    if show_trajectories:
        base_data.append(go.Scatter3d(
            x=missile_path[:,0], y=missile_path[:,1], z=missile_path[:,2],
            mode="lines", line=dict(color="red", width=4, dash="dot"),
            hoverinfo="skip"
        ))
        base_data.append(go.Scatter3d(
            x=[launch_pt[0],intercept[0]], y=[launch_pt[1],intercept[1]], z=[launch_pt[2],intercept[2]],
            mode="lines", line=dict(color="lime", width=5, dash="dash"),
            hoverinfo="skip"
        ))

    if show_labels:
        base_data.append(go.Scatter3d(
            x=[intercept[0]], y=[intercept[1]], z=[intercept[2]],
            mode="markers+text",
            marker=dict(size=7,color="yellow"),
            text=["Neutralization Point"],
            textposition="top center",
            hoverinfo="skip"
        ))
    else:
        base_data.append(go.Scatter3d(
            x=[intercept[0]], y=[intercept[1]], z=[intercept[2]],
            mode="markers",
            marker=dict(size=7,color="yellow"),
            hoverinfo="skip"
        ))

    if show_radar_sweep:
        base_data.append(radar_sweep_line(det_r, 0))

    base_data += missile_body(missile_path[0,0], missile_path[0,1], missile_path[0,2], "red")
    base_data += interceptor_body(launch_pt[0], launch_pt[1], launch_pt[2])

    missile_trace_count = 5
    interceptor_trace_count = 2
    dynamic_start = len(base_data) - missile_trace_count - interceptor_trace_count

    frames = []

    for i in range(n):
        idx = min(i, intercept_idx)

        if i < intercept_idx:
            mpos = missile_path[idx]
            ipos = interceptor_path[min(i,n-1)]
            mcolor = "red"
            dynamic = []
            if show_radar_sweep:
                dynamic.append(radar_sweep_line(det_r, i))
            dynamic += missile_body(mpos[0], mpos[1], mpos[2], mcolor)
            dynamic += interceptor_body(ipos[0], ipos[1], ipos[2])
            if show_exhaust:
                dynamic.append(exhaust_trail(missile_path, idx, "orange"))
                dynamic.append(exhaust_trail(interceptor_path, i, "lime"))
        else:
            dynamic = []
            if show_radar_sweep:
                dynamic.append(radar_sweep_line(det_r, i))
            dynamic += missile_body(intercept[0], intercept[1], intercept[2], "orange")
            dynamic += interceptor_body(intercept[0], intercept[1], intercept[2])
            dynamic += burst_cloud(intercept[0], intercept[1], intercept[2])

        # simple trace replacement: redraw dynamic traces after base
        frames.append(go.Frame(data=dynamic, name=str(i)))

    fig = go.Figure(data=base_data, frames=frames)

    fig.update_layout(
        height=760,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0,r=0,t=40,b=0),
        scene=dict(
            xaxis=dict(range=[-18,10],showgrid=False,showbackground=False,zeroline=False),
            yaxis=dict(range=[-10,10],showgrid=False,showbackground=False,zeroline=False),
            zaxis=dict(range=[0,9],showgrid=False,showbackground=False,zeroline=False),
            aspectmode="manual",
            aspectratio=dict(x=1.7,y=1,z=0.55),
            camera=camera_dict(camera_view)
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.98,
                buttons=[
                    dict(
                        label="▶ Play Simulation",
                        method="animate",
                        args=[None, dict(frame=dict(duration=90, redraw=True), transition=dict(duration=0), mode="immediate")]
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

    return fig

# ============================================================
# TAB 2
# ============================================================

with tabs[1]:
    st.subheader("Clean 3D APS Interception Simulation")
    fig3d = build_3d_figure()
    st.plotly_chart(fig3d, use_container_width=True)
    st.success("Red = incoming threat. Lime = interceptor. Yellow/orange burst = neutralization. Cyan = detection. Green = interception.")

# ============================================================
# TAB 3
# ============================================================

with tabs[2]:
    st.subheader("Reaction Time and Slew Rate Comparison")

    comparison_df = pd.DataFrame({
        "Parameter": [
            "Required Slew Angle",
            "Launcher / Puck Slew Rate",
            "Slew Time",
            "Sensor Delay",
            "Radar Tracking Delay",
            "AI Decision Delay",
            "Launcher Activation / Ignition Delay",
            "Total Reaction Time",
            "Projectile / Interceptor Velocity",
            "Interceptor Flight Time",
            "Total Interception Time",
            "Time Margin",
            "Probability of Kill"
        ],
        "ETC-DMA Puck Launcher": [
            f"{etc_effective_slew_angle_deg} deg",
            f"{etc_dma_slew_rate_dps} deg/s",
            f"{etc_slew_time*1000:.2f} ms",
            f"{sensor_delay_ms} ms",
            f"{radar_delay_ms} ms",
            f"{ai_delay_ms} ms",
            f"{etc_activation_delay_ms} ms",
            f"{etc_total_reaction*1000:.2f} ms",
            f"{etc['muzzle_velocity']:.2f} m/s",
            f"{etc_interceptor_flight:.4f} s",
            f"{etc_total_interception:.4f} s",
            f"{etc_time_margin:.4f} s",
            f"{etc_pk*100:.2f}%"
        ],
        "Conventional Slewed Launcher": [
            f"{angle} deg",
            f"{conventional_slew_rate_dps} deg/s",
            f"{conventional_slew_time*1000:.2f} ms",
            f"{sensor_delay_ms} ms",
            f"{radar_delay_ms} ms",
            f"{ai_delay_ms} ms",
            f"{conventional_ignition_delay_ms} ms",
            f"{conventional_total_reaction*1000:.2f} ms",
            f"{pyro_velocity:.2f} m/s",
            f"{conventional_interceptor_flight:.4f} s",
            f"{conventional_total_interception:.4f} s",
            f"{conventional_time_margin:.4f} s",
            f"{conventional_pk*100:.2f}%"
        ]
    })

    st.table(comparison_df)

    chart_df = pd.DataFrame({
        "Metric": ["Slew Time (ms)", "Reaction Time (ms)", "Interception Time (ms)", "Pk (%)"],
        "ETC-DMA": [etc_slew_time*1000, etc_total_reaction*1000, etc_total_interception*1000, etc_pk*100],
        "Conventional": [conventional_slew_time*1000, conventional_total_reaction*1000, conventional_total_interception*1000, conventional_pk*100]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df["Metric"], y=chart_df["ETC-DMA"], name="ETC-DMA"))
    fig.add_trace(go.Bar(x=chart_df["Metric"], y=chart_df["Conventional"], name="Conventional"))
    fig.update_layout(barmode="group", height=430, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4
# ============================================================

with tabs[3]:
    st.subheader("2D Tactical Top View")

    theta = np.radians(angle)
    missile_x = -distance/25
    missile_y = -np.tan(theta)*(distance/25)*0.45

    fig2d = go.Figure()

    t = np.linspace(0, 2*np.pi, 250)
    fig2d.add_trace(go.Scatter(x=(detection_range/25)*np.cos(t), y=(detection_range/25)*np.sin(t), mode="lines", name="Detection Range"))
    fig2d.add_trace(go.Scatter(x=(interception_range/25)*np.cos(t), y=(interception_range/25)*np.sin(t), mode="lines", name="Interception Range"))
    fig2d.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", marker=dict(size=20), text=["MBT"], textposition="top center"))
    fig2d.add_trace(go.Scatter(x=[missile_x,0], y=[missile_y,0], mode="lines+markers", name="Threat Path"))

    sectors = ["F","FL","L","RL","R","RR","Right","FR"]
    for i in range(8):
        a = np.radians(i*45+22.5)
        col = "red" if i+1 == dma_id else "gold"
        fig2d.add_trace(go.Scatter(x=[3*np.cos(a)], y=[3*np.sin(a)], mode="markers+text",
                                   marker=dict(size=14,color=col), text=[str(i+1)], textposition="top center", showlegend=False))

    fig2d.update_layout(height=560, template="plotly_dark", title="Tactical Top-View Engagement Map", xaxis=dict(scaleanchor="y"))
    st.plotly_chart(fig2d, use_container_width=True)

# ============================================================
# TAB 5
# ============================================================

with tabs[4]:
    st.subheader("ETC Curves and Contours")

    t_ms = np.linspace(0, 5, 160)
    peak_t = 1.0 + 0.4*(1-plasma_efficiency)
    width = 0.55 + 0.25*(propellant_mass_g/300)

    pressure = etc["pressure_mpa"] * np.exp(-((t_ms-peak_t)/width)**2)
    temp = 300 + (etc["temp_K"]-300)*np.exp(-((t_ms-peak_t)/(width*1.15))**2)
    vel = np.minimum(etc["muzzle_velocity"]*(1-np.exp(-t_ms/1.25)), etc["muzzle_velocity"])

    curve_tabs = st.tabs(["Pressure", "Temperature", "Velocity"])

    for tb, title, y, yl in [
        (curve_tabs[0],"Pressure vs Time",pressure,"Pressure (MPa)"),
        (curve_tabs[1],"Temperature vs Time",temp,"Temperature (K)"),
        (curve_tabs[2],"Velocity vs Time",vel,"Velocity (m/s)")
    ]:
        with tb:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_ms,y=y,mode="lines",line=dict(width=4)))
            fig.update_layout(height=360,title=title,xaxis_title="Time (ms)",yaxis_title=yl,template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)

    st.markdown("### CFD-Style Contours")

    Lc, Lb = chamber_length_mm, barrel_length_mm
    total_len = Lc + Lb
    chamber_r = chamber_diameter_mm/2
    barrel_r = interceptor_diameter_mm/2

    x = np.linspace(0,total_len,220)
    y = np.linspace(0,chamber_r,90)
    X,Y = np.meshgrid(x,y)

    transition_start = Lc*0.82

    R = np.where(
        X<transition_start,
        chamber_r,
        np.where(X<Lc, chamber_r-(chamber_r-barrel_r)*((X-transition_start)/max(Lc-transition_start,1e-6)), barrel_r)
    )

    inside = Y <= R
    xn = X/max(total_len,1e-6)

    P = etc["pressure_mpa"]*np.exp(-2.7*xn)*np.exp(-(Y/np.maximum(R,1e-6))**2*0.8)
    T = 300+(etc["temp_K"]-300)*np.exp(-3.3*xn)*np.exp(-(Y/(0.72*np.maximum(R,1e-6)))**2)
    V = etc["muzzle_velocity"]*(1-np.exp(-3.2*xn))*(0.65+0.35*(1-(Y/np.maximum(R,1e-6))**2))

    P,T,V = [np.where(inside,F,np.nan) for F in (P,T,V)]

    contour_tabs = st.tabs(["Pressure Contour", "Temperature Contour", "Velocity Contour"])

    for tb, title, field, unit, scale_name in [
        (contour_tabs[0],"Pressure Contour",P,"MPa","Jet"),
        (contour_tabs[1],"Temperature Contour",T,"K","Hot"),
        (contour_tabs[2],"Velocity Contour",V,"m/s","Viridis")
    ]:
        with tb:
            figc = go.Figure(go.Contour(x=x,y=y,z=field,colorscale=scale_name,colorbar=dict(title=unit)))
            figc.update_layout(height=450,title=title,xaxis_title="Axial Length (mm)",yaxis_title="Radius (mm)",template="plotly_dark")
            st.plotly_chart(figc,use_container_width=True)

# ============================================================
# TAB 6
# ============================================================

with tabs[5]:
    st.subheader("Download Simulation and Result Report")

    report_df = pd.DataFrame({
        "Parameter": [
            "Date",
            "Threat Type",
            "Velocity",
            "Distance",
            "Incoming Angle",
            "Threat Class",
            "Selected DMA",
            "ETC Peak Pressure",
            "ETC Temperature",
            "ETC Muzzle Velocity",
            "ETC Slew Time",
            "ETC Total Reaction Time",
            "ETC Total Interception Time",
            "ETC Pk",
            "Conventional Slew Time",
            "Conventional Total Reaction Time",
            "Conventional Total Interception Time",
            "Conventional Pk"
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            threat_type,
            f"{velocity} m/s",
            f"{distance} m",
            f"{angle} deg",
            class_name(threat_class),
            f"{dma_id}: {dma_sector}",
            f"{etc['pressure_mpa']:.2f} MPa",
            f"{etc['temp_K']:.0f} K",
            f"{etc['muzzle_velocity']:.2f} m/s",
            f"{etc_slew_time*1000:.2f} ms",
            f"{etc_total_reaction*1000:.2f} ms",
            f"{etc_total_interception:.4f} s",
            f"{etc_pk*100:.2f}%",
            f"{conventional_slew_time*1000:.2f} ms",
            f"{conventional_total_reaction*1000:.2f} ms",
            f"{conventional_total_interception:.4f} s",
            f"{conventional_pk*100:.2f}%"
        ]
    })

    st.table(report_df)

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Result Report",
        data=csv,
        file_name="aps_etc_simulation_result_report.csv",
        mime="text/csv"
    )

    fig_html = build_3d_figure().to_html(include_plotlyjs="cdn")

    html_report = f"""
    <html>
    <head>
        <title>APS ETC Simulation Report</title>
    </head>
    <body style="background:#07110d;color:#e8f5e9;font-family:Arial;">
        <h1>APS ETC Simulation Report</h1>
        <h2>Key Results</h2>
        {report_df.to_html(index=False)}
        <h2>3D Simulation View</h2>
        {fig_html}
    </body>
    </html>
    """

    st.download_button(
        label="🌐 Download Full HTML Simulation View",
        data=html_report.encode("utf-8"),
        file_name="aps_etc_3d_simulation_view.html",
        mime="text/html"
    )
