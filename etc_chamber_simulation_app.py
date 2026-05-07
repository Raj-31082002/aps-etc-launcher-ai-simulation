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

st.title("⚡ ETC Launcher Chamber Simulation with Contour Results")
st.caption(
    "Simplified academic simulation of plasma-assisted ETC launcher chamber behavior: "
    "pressure, temperature, velocity, contour fields, and interception timing."
)

st.warning(
    "Academic simplified model only. Use this for thesis visualization/trend explanation, "
    "not as a validated weapon-design solver."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Input Parameters")

propellant_mass_g = st.sidebar.slider("Propellant Mass (g)", 20, 300, 120, 5)
electric_pulse_kj = st.sidebar.slider("Electric Pulse Energy (kJ)", 5, 120, 35, 5)
plasma_efficiency = st.sidebar.slider("Plasma Coupling Efficiency", 0.05, 0.60, 0.25, 0.01)

chamber_length_mm = st.sidebar.slider("Chamber Length (mm)", 40, 250, 120, 5)
chamber_diameter_mm = st.sidebar.slider("Chamber Diameter (mm)", 20, 80, 50, 2)
barrel_length_mm = st.sidebar.slider("Launcher / Barrel Length (mm)", 150, 1500, 500, 25)

interceptor_mass_g = st.sidebar.slider("Interceptor Mass (g)", 20, 300, 80, 5)
interceptor_diameter_mm = st.sidebar.slider("Interceptor Diameter (mm)", 10, 50, 25, 1)

threat_distance_m = st.sidebar.slider("Threat Distance (m)", 20, 500, 180, 10)
threat_velocity_ms = st.sidebar.slider("Threat Velocity (m/s)", 100, 900, 200, 10)
intercept_distance_m = st.sidebar.slider("Desired Interception Distance (m)", 20, 200, 80, 5)


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
    pressure_mpa = pressure_pa / 1e6
    pressure_mpa = safe_clip(pressure_mpa, 5.0, 450.0)

    chamber_to_barrel_volume = bore_area * Lb
    useful_work = min(
        total_effective_heat * ballistic_efficiency,
        pressure_pa * chamber_to_barrel_volume * 0.55
    )

    muzzle_velocity = np.sqrt(max(2.0 * useful_work / max(mi, 1e-6), 0.0))
    muzzle_velocity = safe_clip(muzzle_velocity, 50.0, 950.0)

    avg_acceleration = muzzle_velocity ** 2 / (2 * max(Lb, 1e-6))
    launch_time = muzzle_velocity / max(avg_acceleration, 1e-6)

    threat_time_to_vehicle = threat_distance_m / threat_velocity_ms
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
    st.metric("Peak Chamber Temperature", f"{out['temperature_K']:.0f} K")
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

t = np.linspace(0, 5.0, 160)

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

st.markdown("## Curves: Pressure, Temperature and Velocity")

tab1, tab2, tab3 = st.tabs(["Pressure Curve", "Temperature Curve", "Velocity Curve"])

with tab1:
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=t, y=pressure_curve, mode="lines", line=dict(width=4), name="Pressure"))
    fig_p.update_layout(
        height=380,
        title="Pressure vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Pressure (MPa)",
        template="plotly_white"
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=t, y=temperature_curve, mode="lines", line=dict(width=4), name="Temperature"))
    fig_t.update_layout(
        height=380,
        title="Temperature vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Temperature (K)",
        template="plotly_white"
    )
    st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t, y=velocity_curve, mode="lines", line=dict(width=4), name="Velocity"))
    fig_v.update_layout(
        height=380,
        title="Velocity vs Time",
        xaxis_title="Time (ms)",
        yaxis_title="Velocity (m/s)",
        template="plotly_white"
    )
    st.plotly_chart(fig_v, use_container_width=True)


# ============================================================
# 2D CONTOUR FIELD MODEL
# ============================================================

st.markdown("## Contour Results: Pressure, Temperature and Velocity Fields")

Lc_mm = chamber_length_mm
Lb_mm = barrel_length_mm
total_length_mm = Lc_mm + Lb_mm
chamber_radius_mm = chamber_diameter_mm / 2
barrel_radius_mm = interceptor_diameter_mm / 2

nx = 230
ny = 95

x = np.linspace(0, total_length_mm, nx)
y = np.linspace(0, chamber_radius_mm, ny)
X, Y = np.meshgrid(x, y)

# Radius profile: chamber -> smooth contraction -> barrel
transition_start = Lc_mm * 0.82
transition_end = Lc_mm
R = np.where(
    X < transition_start,
    chamber_radius_mm,
    np.where(
        X < transition_end,
        chamber_radius_mm - (chamber_radius_mm - barrel_radius_mm) * ((X - transition_start) / max(transition_end - transition_start, 1e-6)),
        barrel_radius_mm
    )
)

inside = Y <= R

# Normalized axial position
xn = X / max(total_length_mm, 1e-6)
radial_factor = np.exp(-(Y / np.maximum(R, 1e-6)) ** 2 * 0.8)

# Synthetic but physically shaped contour fields
P_field = out["pressure_mpa"] * np.exp(-2.7 * xn) * radial_factor
P_field += 0.18 * out["pressure_mpa"] * np.exp(-((X - 0.22 * Lc_mm) / (0.22 * Lc_mm + 1e-6)) ** 2) * np.exp(-(Y / (0.55 * chamber_radius_mm + 1e-6)) ** 2)
P_field = np.where(inside, P_field, np.nan)

T_field = 300 + (out["temperature_K"] - 300) * np.exp(-3.3 * xn) * np.exp(-(Y / (0.72 * np.maximum(R, 1e-6))) ** 2)
T_field += 0.25 * (out["temperature_K"] - 300) * np.exp(-((X - 0.15 * Lc_mm) / (0.16 * Lc_mm + 1e-6)) ** 2)
T_field = np.where(inside, T_field, np.nan)

V_field = out["muzzle_velocity"] * (1 - np.exp(-3.2 * xn)) * (0.65 + 0.35 * (1 - (Y / np.maximum(R, 1e-6)) ** 2))
V_field = np.where(inside, V_field, np.nan)

contour_tabs = st.tabs(["Pressure Contour", "Temperature Contour", "Velocity Contour"])

with contour_tabs[0]:
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Contour(
        x=x,
        y=y,
        z=P_field,
        colorscale="Jet",
        colorbar=dict(title="MPa"),
        contours=dict(showlabels=True)
    ))
    fig_pc.update_layout(
        height=470,
        title="Pressure Contour Inside ETC Chamber and Barrel",
        xaxis_title="Axial Length (mm)",
        yaxis_title="Radius (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig_pc, use_container_width=True)

with contour_tabs[1]:
    fig_tc = go.Figure()
    fig_tc.add_trace(go.Contour(
        x=x,
        y=y,
        z=T_field,
        colorscale="Hot",
        colorbar=dict(title="K"),
        contours=dict(showlabels=True)
    ))
    fig_tc.update_layout(
        height=470,
        title="Temperature Contour Inside ETC Chamber and Barrel",
        xaxis_title="Axial Length (mm)",
        yaxis_title="Radius (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig_tc, use_container_width=True)

with contour_tabs[2]:
    fig_vc = go.Figure()
    fig_vc.add_trace(go.Contour(
        x=x,
        y=y,
        z=V_field,
        colorscale="Viridis",
        colorbar=dict(title="m/s"),
        contours=dict(showlabels=True)
    ))
    fig_vc.update_layout(
        height=470,
        title="Velocity Contour Inside ETC Chamber and Barrel",
        xaxis_title="Axial Length (mm)",
        yaxis_title="Radius (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig_vc, use_container_width=True)


# ============================================================
# 3D ETC CHAMBER ANIMATION
# ============================================================

st.markdown("## 3D ETC Chamber Working Animation")

def cylinder_surface_x(x0, x1, radius, y0=0, z0=0, color="lightgray", opacity=0.35, name="Cylinder"):
    theta = np.linspace(0, 2 * np.pi, 50)
    xx = np.linspace(x0, x1, 2)
    theta, xx = np.meshgrid(theta, xx)
    yy = y0 + radius * np.cos(theta)
    zz = z0 + radius * np.sin(theta)

    return go.Surface(
        x=xx,
        y=yy,
        z=zz,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hoverinfo="skip"
    )

def projectile_mesh(xc, length, radius):
    return cylinder_surface_x(xc - length / 2, xc + length / 2, radius, color="darkslategray", opacity=1.0, name="Interceptor")

def plasma_cloud(xc, strength):
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 15)
    u, v = np.meshgrid(u, v)
    r = 0.18 + 0.22 * strength
    xx = xc + r * np.cos(u) * np.sin(v)
    yy = r * np.sin(u) * np.sin(v)
    zz = r * np.cos(v)

    return go.Surface(
        x=xx,
        y=yy,
        z=zz,
        opacity=0.30,
        colorscale=[[0, "orange"], [1, "red"]],
        showscale=False,
        name="Plasma Energy Zone",
        hoverinfo="skip"
    )

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

state_label = go.Scatter3d(
    x=[chamber_x1 * 0.45],
    y=[-chamber_r * 1.4],
    z=[chamber_r * 1.2],
    mode="text",
    text=[f"P = {pressure_curve[0]:.1f} MPa<br>T = {temperature_curve[0]:.0f} K"],
    name="State Label",
    hoverinfo="skip"
)

base_data = [base_chamber, base_barrel, axis_line, initial_plasma, initial_projectile, state_label]

frames = []
for i in range(n_frames):
    frac = i / (n_frames - 1)
    idx = min(int(frac * (len(t) - 1)), len(t) - 1)
    plasma_strength = max(0.05, np.exp(-((frac - 0.15) / 0.22) ** 2))

    cloud = plasma_cloud(chamber_x0 + 0.18, plasma_strength)
    proj = projectile_mesh(projectile_positions[i], proj_len, proj_r)

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

    frames.append(go.Frame(data=[cloud, proj, label], traces=[3, 4, 5], name=str(i)))

fig3d = go.Figure(data=base_data, frames=frames)

fig3d.update_layout(
    height=650,
    title="ETC Working: Electric Pulse → Plasma → Pressure Rise → Projectile Motion",
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

The pressure, temperature, and velocity contours show the expected spatial distribution inside the chamber
and barrel. The maximum pressure and temperature occur near the chamber/breech region due to plasma-assisted
energy deposition, while the velocity field increases along the barrel as the projectile accelerates toward
the muzzle. These results are intended for conceptual thesis demonstration and trend analysis. Detailed
validation should be performed using ANSYS Fluent transient CFD results and published ETC/internal ballistics
literature.
""")
