import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="Zoom-based Hexbin Map", layout="wide")
st.title("Zoom-Level Based Hexagon Merging")

# Sample data - Replace with your actual dataset
# -- Generate synthetic data (adjust as needed) --
NUM_POINTS = st.slider("Number of Points", min_value=1000, max_value=200000, value=100000, step=10000)

# Generate lat/lon around a center (San Francisco)
center_lat, center_lon = 37.76, -122.43
latitudes = np.random.normal(loc=center_lat, scale=0.01, size=NUM_POINTS)
longitudes = np.random.normal(loc=center_lon, scale=0.01, size=NUM_POINTS)

data = pd.DataFrame({'lat': latitudes, 'lon': longitudes})

# Let the user pick a zoom level via slider (simulated zoom control)
zoom_level = st.slider("Simulated Zoom Level", min_value=6, max_value=16, value=12)

# Function to compute hex radius based on zoom level
def zoom_to_radius(zoom):
    return int(20000 / (2 ** (zoom - 8)))

radius = zoom_to_radius(zoom_level)
st.write(f"🔍 Hex Radius: {radius} meters")

# Define a custom color range with 10 intensity levels
color_range = [
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
    [128, 0, 38],
    [77, 0, 75]
]

# -- Create HexagonLayer --
layer = pdk.Layer(
    'HexagonLayer',
    data,
    get_position='[lon, lat]',
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 1000],
    extruded=True,
    coverage=1,
    radius=radius,
    get_color_weight=1,
    color_aggregation='sum',
    color_range=color_range,
)

# --- View state ---
view_state = pdk.ViewState(
    latitude=data['lat'].mean(),
    longitude=data['lon'].mean(),
    zoom=zoom_level,
    pitch=40,
)

# --- Show map ---
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Count: {elevationValue}"}
))
