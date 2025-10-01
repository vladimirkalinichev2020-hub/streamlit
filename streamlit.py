import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

model = joblib.load("model.pkl")
df = pd.read_csv("weather_classification_data.csv")

cloud_map = {"ясно": 0, "облачно": 1, "пасмурно": 2, "малооблачно": 3}
season_map = {"осень": 0, "весна": 1, "лето": 2, "зима": 3}
location_map = {"побережье": 0, "внутренние": 1, "горы": 2}
weather_map = {0: "Облачно", 1: "Дождь", 2: "Снег", 3: "Солнце"}

df_numeric = df.copy()
df_numeric["Cloud Cover"] = df_numeric["Cloud Cover"].map(cloud_map)
df_numeric["Season"] = df_numeric["Season"].map(season_map)
df_numeric["Location"] = df_numeric["Location"].map(location_map)
df_numeric["Weather Type"] = df_numeric["Weather Type"].map(
    {"Cloudy": 0, "Rainy": 1, "Snowy": 2, "Sunny": 3}
)

templates = {
    "Снег": ("пасмурно", "зима", "горы", 2),
    "Дождь": ("облачно", "весна", "внутренние", 1),
    "Солнце": ("ясно", "лето", "побережье", 3),
    "Облачно": ("малооблачно", "осень", "внутренние", 0),
}

weather_templates = {}
for name, (cloud_val, season_val, loc_val, num) in templates.items():
    tmp = df_numeric[df_numeric["Weather Type"] == num].mean()
    weather_templates[name] = {
        "temperature": round(tmp["Temperature"]),
        "humidity": round(tmp["Humidity"]),
        "wind_speed": round(tmp["Wind Speed"]),
        "precipitation": round(tmp["Precipitation (%)"]),
        "cloud_cover": cloud_val,
        "atmospheric_pressure": round(tmp["Atmospheric Pressure"]),
        "uv_index": round(tmp["UV Index"]),
        "season": season_val,
        "visibility": round(tmp["Visibility (km)"]),
        "location": loc_val,
    }

default_values = {
    "temperature": 20.0,
    "humidity": 50.0,
    "wind_speed": 10.0,
    "precipitation": 0.0,
    "cloud_cover": "ясно",
    "atmospheric_pressure": 1013.0,
    "uv_index": 5.0,
    "season": "лето",
    "visibility": 10.0,
    "location": "внутренние",
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Прогноз типа погоды")

def set_weather(template_name):
    for key, val in weather_templates[template_name].items():
        st.session_state[key] = val

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
if col_btn1.button("Снег"):
    set_weather("Снег")
if col_btn2.button("Дождь"):
    set_weather("Дождь")
if col_btn3.button("Солнце"):
    set_weather("Солнце")
if col_btn4.button("Облачно"):
    set_weather("Облачно")

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Температура (°C)",
        -50.0,
        50.0,
        float(st.session_state.temperature),
        1.0,
        key="temperature",
    )
    st.number_input(
        "Влажность (%)", 0.0, 200.0, float(st.session_state.humidity), 1.0, key="humidity"
    )
    st.number_input(
        "Скорость ветра (км/ч)",
        0.0,
        150.0,
        float(st.session_state.wind_speed),
        1.0,
        key="wind_speed",
    )
    st.number_input(
        "Осадки (%)",
        0.0,
        150.0,
        float(st.session_state.precipitation),
        1.0,
        key="precipitation",
    )

with col2:
    st.selectbox(
        "Облачность",
        list(cloud_map.keys()),
        list(cloud_map.keys()).index(st.session_state.cloud_cover),
        key="cloud_cover",
    )
    st.number_input(
        "Атмосферное давление (гПа)",
        800.0,
        1100.0,
        float(st.session_state.atmospheric_pressure),
        1.0,
        key="atmospheric_pressure",
    )
    st.number_input(
        "УФ-индекс", 0.0, 15.0, float(st.session_state.uv_index), 1.0, key="uv_index"
    )
    st.selectbox(
        "Сезон",
        list(season_map.keys()),
        list(season_map.keys()).index(st.session_state.season),
        key="season",
    )

with col3:
    st.number_input(
        "Видимость (км)",
        0.0,
        50.0,
        float(st.session_state.visibility),
        1.0,
        key="visibility",
    )
    st.selectbox(
        "Локация",
        list(location_map.keys()),
        list(location_map.keys()).index(st.session_state.location),
        key="location",
    )

if st.button("Предсказать тип погоды"):
    input_data = pd.DataFrame(
        {
            "Temperature": [st.session_state.temperature],
            "Humidity": [st.session_state.humidity],
            "Wind Speed": [st.session_state.wind_speed],
            "Precipitation (%)": [st.session_state.precipitation],
            "Cloud Cover": [cloud_map[st.session_state.cloud_cover]],
            "Atmospheric Pressure": [st.session_state.atmospheric_pressure],
            "UV Index": [st.session_state.uv_index],
            "Season": [season_map[st.session_state.season]],
            "Visibility (km)": [st.session_state.visibility],
            "Location": [location_map[st.session_state.location]],
        }
    )
    prediction_num = model.predict(input_data)[0]
    prediction_word = weather_map[prediction_num]
    st.success(f"Предсказанный тип погоды: {prediction_word}")