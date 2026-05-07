import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ETC Launcher Chamber Simulation",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electro-Thermal Chemical Launcher: Chamber Working Simulation")
st.caption(
    "Academic visualization of ETC chamber ignition, plasma energy deposition, pressure rise, "
    "projectile acceleration, and interceptor timing."
)

st.warning(
    "This app uses a simplified educational model for thesis visualization. "
    "It is not a validated weapon-design solver."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Input Parameters")

propellant_mass_g = st.sidebar.slider(
    "Propellant Mass (g)",
    min_value=20,
    max_value=300,
    value=120,
    step=5
)

electric_pulse_kj = st.sidebar.slider(
    "Electric Pulse Energy (kJ)",
    min_value=5,
    max_value=120,
    value=35,
    step=5
)

plasma_efficiency = st.sidebar.slider(
    "Plasma Coupling Efficiency",
    min_value=0.05,
    max_value=0.60,
    value=0.25,
    step=0.01
)

chamber_length_mm = st.sidebar.slider(
    "Chamber Length (mm)",
    min_value=40,
    max_value=250,
    value=120,
    step=5
)

chamber_diameter_mm = st.sidebar.slider(
    "Chamber Diameter (mm)",
    min_value=20,
    max_value=80,
    value=50,
    step=2
)

barrel_length_mm = st.sidebar.slider(
    "Launcher / Barrel Length (mm)",
    min_value=150,
    max_value=1500,
    value=500,
    step=25
)

interceptor_mass_g = st.sidebar.slider(
    "Interceptor Mass (g)",
    min_value=20,
    max_value=300,
    value=80,
    step=5
)

interceptor_diameter_mm = st.sidebar.slider(
    "Interceptor Diameter (mm)",
    min_value=10,
    max_value=50,
    value=25,
    step=1
)

threat_distance_m = st.sidebar.slider(
    "Threat Distance (m)",
    min_value=20,
    max_value=500,
    value=180,
    step=10
)

threat_velocity_ms = st.sidebar.slider(
    "Threat Velocity (m/s)",
    min_value=100,
    max_value=900,
    value=200,
    step=10
)

intercept_distance_m = st.sidebar.slider(
    "Desired Interception Distance (m)",
    min_value=20,
    max_value=200,
    value=80,
    step=5
)


# ============================================================
# SIMPLIFIED PHYSICS MODEL
# ============================================================

def safe_clip(value, low, high):
    return max(low, min(value, high))


def calculate_outputs(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm,
    threat_distance_m,
    threat_velocity_ms,
    intercept_distance_m
):
    # Unit conversion
    mp = propellant_mass_g / 1000.0
    Ei = electric_pulse_kj * 1000.0
    mi = interceptor_mass_g / 1000.0

    Lc = chamber_length_mm / 1000.0
    Dc = chamber_diameter_mm / 1000.0
    Lb = barrel_length_mm / 1000.0
    Di = interceptor_diameter_mm / 1000.0

    chamber_volume = np.pi * (Dc / 2) ** 2 * Lc
    bore_area = np.pi * (Di / 2) ** 2

    # Conservative academic constants
    propellant_energy_density = 3.0e6      # J/kg, simplified low-order estimate
    ballistic_efficiency = 0.18            # fraction converted to useful projectile work
    gas_fraction = 0.85                    # notional gas generation fraction
    gas_R = 300.0                          # J/kg-K, propellant gas approximation
    gamma = 1.22

    chemical_energy = mp * propellant_energy_density
    coupled_electric_energy = Ei * plasma_efficiency

    total_effective_heat = chemical_energy + coupled_electric_energy

    # Gas mass estimate
    gas_mass = max(mp * gas_fraction, 1e-6)

    # Temperature rise estimate
    cv = gas_R / (gamma - 1.0)
    temperature_K = 300.0 + total_effective_heat / (gas_mass * cv)
    temperature_K = safe_clip(temperature_K, 500.0, 6500.0)

    # Peak pressure estimate from ideal gas, capped for academic range
    pressure_pa = (gas_mass * gas_R * temperature_K) / max(chamber_volume, 1e-9)
    pressure_mpa = pressure_pa / 1e6
    pressure_mpa = safe_clip(pressure_mpa, 5.0, 450.0)

    # Work on interceptor: pressure work + efficiency correction
    chamber_to_barrel_volume = bore_area * Lb
    useful_work = min(
        total_effective_heat * ballistic_efficiency,
        pressure_pa * chamber_to_barrel_volume * 0.55
    )

    muzzle_velocity = np.sqrt(max(2.0 * useful_work / max(mi, 1e-6), 0.0))
    muzzle_velocity = safe_clip(muzzle_velocity, 50.0, 950.0)

    # Average acceleration and launch time
    avg_acceleration = muzzle_velocity ** 2 / (2 * max(Lb, 1e-6))
    launch_time = muzzle_velocity / max(avg_acceleration, 1e-6)

    # Interception timing
    threat_time_to_vehicle = threat_distance_m / threat_velocity_ms
    threat_time_to_intercept_zone = max(
        (threat_distance_m - intercept_distance_m) / threat_velocity_ms,
        0.0
    )

    interceptor_flight_time = intercept_distance_m / muzzle_velocity
    total_interception_time = launch_time + interceptor_flight_time
    time_margin = threat_time_to_vehicle - total_interception_time

    if time_margin > 0.10:
        engagement_status = "FEASIBLE WITH GOOD TIME MARGIN"
    elif time_margin > 0:
        engagement_status = "FEASIBLE BUT TIGHT WINDOW"
    else:
        engagement_status = "NOT FEASIBLE FOR SELECTED PARAMETERS"

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
        "launch_time": launch_time,
        "threat_time_to_vehicle": threat_time_to_vehicle,
        "threat_time_to_intercept_zone": threat_time_to_intercept_zone,
        "interceptor_flight_time": interceptor_flight_time,
        "total_interception_time": total_interception_time,
        "time_margin": time_margin,
        "engagement_status": engagement_status
    }


out = calculate_outputs(
    propellant_mass_g,
    electric_pulse_kj,
    plasma_efficiency,
    chamber_length_mm,
    chamber_diameter_mm,
    barrel_length_mm,
    interceptor_mass_g,
    interceptor_diameter_mm,
    threat_distance_m,
    threat_velocity_ms,
    intercept_distance_m
)


# ============================================================
# OUTPUT CARDS
# ============================================================

st.markdown("## Main Output Results")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Peak Chamber Pressure", f"{out['pressure_mpa']:.1f} MPa")

with c2:
    st.metric("Chamber Temperature", f"{out['temperature_K']:.0f} K")

with c3:
    st.metric("Muzzle Velocity", f"{out['muzzle_velocity']:.1f} m/s")

with c4:
    st.metric("Interception Time", f"{out['total_interception_time']:.4f} s")


if out["time_margin"] > 0:
    st.success(f"Engagement Status: {out['engagement_status']} | Time Margin: {out['time_margin']:.4f} s")
else:
    st.error(f"Engagement Status: {out['engagement_status']} | Time Deficit: {abs(out['time_margin']):.4f} s")


# ============================================================
# TIME HISTORY CURVES
# ============================================================

st.markdown("## Pressure, Temperature, and Velocity Time History")

t = np.linspace(0, 5.0, 120)  # ms

peak_t = 1.0 + 0.4 * (1 - plasma_efficiency)
width = 0.55 + 0.25 * (propellant_mass_g / 300)

pressure_curve = out["pressure_mpa"] * np.exp(-((t - peak_t) / width) ** 2)
temperature_curve = 300 + (out["temperature_K"] - 300) * np.exp(-((t - peak_t) / (width * 1.15)) ** 2)
velocity_curve = out["muzzle_velocity"] * (1 - np.exp(-t / 1.25))
velocity_curve = np.minimum(velocity_curve, out["muzzle_velocity"])

curve_df = pd.DataFrame({
    "Time (ms)": t,
    "Pressure (MPa)": pressure_curve,
    "Temperature (K)": temperature_curve,
    "Velocity (m/s)": velocity_curve
})

tab1, tab2, tab3 = st.tabs(["Pressure Curve", "Temperature Curve", "Velocity Curve"])

with tab1:
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=curve_df["Time (ms)"],
        y=curve_df["Pressure (MPa)"],
        mode="lines",
        line=dict(width=4),
        name="Pressure"
    ))
    fig_p.update_layout(
        height=360,
        title="Pressure vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Pressure (MPa)",
        template="plotly_white"
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=curve_df["Time (ms)"],
        y=curve_df["Temperature (K)"],
        mode="lines",
        line=dict(width=4),
        name="Temperature"
    ))
    fig_t.update_layout(
        height=360,
        title="Temperature vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Temperature (K)",
        template="plotly_white"
    )
    st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=curve_df["Time (ms)"],
        y=curve_df["Velocity (m/s)"],
        mode="lines",
        line=dict(width=4),
        name="Velocity"
    ))
    fig_v.update_layout(
        height=360,
        title="Velocity vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Velocity (m/s)",
        template="plotly_white"
    )
    st.plotly_chart(fig_v, use_container_width=True)


# ============================================================
# 3D ETC CHAMBER ANIMATION
# ============================================================

st.markdown("## 3D ETC Chamber Working Animation")


def cylinder_surface_x(x0, x1, radius, y0=0, z0=0, color="lightgray", opacity=0.35, name="Cylinder"):
    theta = np.linspace(0, 2 * np.pi, 50)
    x = np.linspace(x0, x1, 2)
    theta, x = np.meshgrid(theta, x)
    y = y0 + radius * np.cos(theta)
    z = z0 + radius * np.sin(theta)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hoverinfo="skip"
    )


def projectile_mesh(xc, length, radius):
    x0 = xc - length / 2
    x1 = xc + length / 2
    return cylinder_surface_x(x0, x1, radius, color="darkslategray", opacity=1.0, name="Interceptor")


def plasma_cloud(xc, strength):
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 15)
    u, v = np.meshgrid(u, v)

    r = 0.18 + 0.22 * strength
    x = xc + r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.30,
        colorscale=[[0, "orange"], [1, "red"]],
        showscale=False,
        name="Plasma Energy Zone",
        hoverinfo="skip"
    )


# Plot units
Lc = chamber_length_mm / 1000
Lb = barrel_length_mm / 1000
Dc = chamber_diameter_mm / 1000
Di = interceptor_diameter_mm / 1000

scale_len = 5.0 / max(Lc + Lb, 1e-6)

chamber_x0 = 0
chamber_x1 = Lc * scale_len
barrel_x0 = chamber_x1
barrel_x1 = (Lc + Lb) * scale_len

chamber_r = max(Dc * scale_len / 2, 0.30)
barrel_r = max(Di * scale_len / 2, 0.14)
proj_len = max((interceptor_diameter_mm / 1000) * scale_len * 1.2, 0.35)
proj_r = barrel_r * 0.85

frames = []
n_frames = 45

projectile_positions = np.linspace(barrel_x0 + proj_len / 2, barrel_x1 - proj_len / 2, n_frames)

base_chamber = cylinder_surface_x(chamber_x0, chamber_x1, chamber_r, color="lightsteelblue", opacity=0.30, name="Chamber")
base_barrel = cylinder_surface_x(barrel_x0, barrel_x1, barrel_r, color="gray", opacity=0.35, name="Barrel")

axis_line = go.Scatter3d(
    x=[chamber_x0, barrel_x1],
    y=[0, 0],
    z=[0, 0],
    mode="lines",
    line=dict(width=4, color="black"),
    name="Axis",
    hoverinfo="skip"
)

initial_projectile = projectile_mesh(projectile_positions[0], proj_len, proj_r)
initial_plasma = plasma_cloud(chamber_x0 + 0.18, 0.2)

pressure_label = go.Scatter3d(
    x=[chamber_x1 * 0.45],
    y=[-chamber_r * 1.4],
    z=[chamber_r * 1.2],
    mode="text",
    text=[f"P = {pressure_curve[0]:.1f} MPa<br>T = {temperature_curve[0]:.0f} K"],
    name="State Label",
    hoverinfo="skip"
)

base_data = [
    base_chamber,
    base_barrel,
    axis_line,
    initial_plasma,
    initial_projectile,
    pressure_label
]

for i in range(n_frames):
    frac = i / (n_frames - 1)
    idx = min(int(frac * (len(t) - 1)), len(t) - 1)

    plasma_strength = max(0.05, np.exp(-((frac - 0.15) / 0.22) ** 2))
    proj = projectile_mesh(projectile_positions[i], proj_len, proj_r)
    cloud = plasma_cloud(chamber_x0 + 0.18, plasma_strength)

    label = go.Scatter3d(
        x=[chamber_x1 * 0.45],
        y=[-chamber_r * 1.4],
        z=[chamber_r * 1.2],
        mode="text",
        text=[
            f"t = {t[idx]:.2f} ms<br>"
            f"P = {pressure_curve[idx]:.1f} MPa<br>"
            f"T = {temperature_curve[idx]:.0f} K<br>"
            f"V = {velocity_curve[idx]:.1f} m/s"
        ],
        hoverinfo="skip"
    )

    frames.append(
        go.Frame(
            data=[cloud, proj, label],
            traces=[3, 4, 5],
            name=str(i)
        )
    )

fig3d = go.Figure(data=base_data, frames=frames)

fig3d.update_layout(
    height=650,
    title="ETC Chamber Working: Electric Pulse → Plasma → Pressure Rise → Interceptor Motion",
    scene=dict(
        xaxis=dict(title="Launcher Axis", showgrid=False),
        yaxis=dict(title="Radial Direction", showgrid=False),
        zaxis=dict(title="Radial Direction", showgrid=False),
        aspectmode="manual",
        aspectratio=dict(x=2.5, y=0.8, z=0.8),
        camera=dict(eye=dict(x=1.6, y=-1.8, z=0.9))
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    showlegend=False,
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.03,
            y=1.05,
            buttons=[
                dict(
                    label="▶ Play ETC Working Animation",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=90, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False), mode="immediate")
                    ]
                )
            ]
        )
    ]
)

st.plotly_chart(fig3d, use_container_width=True)


# ============================================================
# FINAL OUTPUT TABLE
# ============================================================

st.markdown("## Final Technical Output Table")

output_table = pd.DataFrame({
    "Parameter": [
        "Propellant Mass",
        "Electric Pulse Energy",
        "Plasma Coupling Efficiency",
        "Chamber Length",
        "Chamber Diameter",
        "Barrel / Launcher Length",
        "Interceptor Mass",
        "Interceptor Diameter",
        "Peak Chamber Pressure",
        "Peak Chamber Temperature",
        "Estimated Muzzle Velocity",
        "Launcher Acceleration Time",
        "Threat Distance",
        "Threat Velocity",
        "Threat Time-to-Vehicle",
        "Desired Interception Distance",
        "Interceptor Flight Time",
        "Total Interception Time",
        "Time Margin",
        "Engagement Status"
    ],
    "Value": [
        f"{propellant_mass_g} g",
        f"{electric_pulse_kj} kJ",
        f"{plasma_efficiency:.2f}",
        f"{chamber_length_mm} mm",
        f"{chamber_diameter_mm} mm",
        f"{barrel_length_mm} mm",
        f"{interceptor_mass_g} g",
        f"{interceptor_diameter_mm} mm",
        f"{out['pressure_mpa']:.2f} MPa",
        f"{out['temperature_K']:.0f} K",
        f"{out['muzzle_velocity']:.2f} m/s",
        f"{out['launch_time']:.5f} s",
        f"{threat_distance_m} m",
        f"{threat_velocity_ms} m/s",
        f"{out['threat_time_to_vehicle']:.4f} s",
        f"{intercept_distance_m} m",
        f"{out['interceptor_flight_time']:.4f} s",
        f"{out['total_interception_time']:.4f} s",
        f"{out['time_margin']:.4f} s",
        out["engagement_status"]
    ]
})

st.table(output_table)


# ============================================================
# THESIS EXPLANATION
# ============================================================

st.markdown("## Thesis Explanation")

st.write("""
This ETC chamber simulation app demonstrates the internal sequence of an Electro-Thermal Chemical launcher
using a simplified academic model. The user inputs propellant mass, electric pulse energy, plasma coupling
efficiency, chamber dimensions, and interceptor dimensions. The model estimates peak chamber pressure,
temperature, muzzle velocity, and interception timing using energy balance, ideal gas approximation, and
basic projectile motion.

The 3D animation shows electric pulse energy being deposited near the breech region as a plasma cloud,
followed by pressure rise and interceptor motion along the launcher axis. The output is intended for
conceptual thesis demonstration and trend analysis. Detailed validation should be performed using ANSYS
Fluent transient CFD results and published ETC/internal ballistics literature.
""")
