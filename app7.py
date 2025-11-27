import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('random_forest_model.pkl')
        
        # Create label encoders with the actual unique values from training
        source_encoder = LabelEncoder()
        source_encoder.classes_ = np.array(['Source1', 'Source2', 'Source3'])
        
        state_encoder = LabelEncoder()
        state_encoder.classes_ = np.array(['OH', 'WV', 'CA', 'FL', 'GA', 'SC', 'NE', 'IA', 'IL', 'MO', 
                                           'WI', 'IN', 'MI', 'NJ', 'NY', 'CT', 'MA', 'RI', 'NH', 'PA', 
                                           'KY', 'MD', 'VA', 'DC', 'DE', 'TX', 'WA', 'OR', 'AL', 'NC', 
                                           'AZ', 'TN', 'LA', 'MN', 'CO', 'OK', 'NV', 'UT', 'KS', 'NM', 
                                           'AR', 'MS', 'ME', 'VT', 'WY', 'ID', 'ND', 'MT', 'SD'])
        
        timezone_encoder = LabelEncoder()
        timezone_encoder.classes_ = np.array(['US/Central', 'US/Eastern', 'US/Mountain', 'US/Pacific'])
        
        wind_direction_encoder = LabelEncoder()
        wind_direction_encoder.classes_ = np.array(['Calm', 'E', 'ENE', 'ESE', 'East', 'N', 'NE', 'NNE', 
                                                    'NNW', 'NW', 'North', 'S', 'SE', 'SSE', 'SSW', 'South', 
                                                    'SW', 'Variable', 'W', 'WNW', 'WSW', 'West'])
        
        # Top weather conditions from the dataset
        weather_encoder = LabelEncoder()
        weather_encoder.classes_ = np.array(['Blowing Dust', 'Blowing Snow', 'Clear', 'Cloudy', 'Drizzle', 
                                             'Fair', 'Fog', 'Hail', 'Haze', 'Heavy Rain', 'Heavy Snow', 
                                             'Heavy T-Storm', 'Light Drizzle', 'Light Freezing Rain', 
                                             'Light Rain', 'Light Snow', 'Mist', 'Mostly Cloudy', 
                                             'Overcast', 'Partly Cloudy', 'Rain', 'Scattered Clouds', 
                                             'Sleet', 'Smoke', 'Snow', 'T-Storm', 'Thunder', 
                                             'Thunderstorm', 'Tornado', 'Wintry Mix'])
        
        return model, source_encoder, state_encoder, timezone_encoder, wind_direction_encoder, weather_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

# Feature engineering function
def engineer_features(data, source_encoder, state_encoder, timezone_encoder, wind_direction_encoder, weather_encoder):
    """
    Apply the same feature engineering as during training
    """
    df = data.copy()
    
    # Encode categorical variables
    df['Source'] = source_encoder.transform(df['Source'])
    df['State'] = state_encoder.transform(df['State'])
    df['Timezone'] = timezone_encoder.transform(df['Timezone'])
    df['Wind_Direction'] = wind_direction_encoder.transform(df['Wind_Direction'])
    df['Weather_Condition'] = weather_encoder.transform(df['Weather_Condition'])
    
    # Parse datetime features
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])
    df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'])
    
    # Calculate duration in minutes
    df['Duration_in_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    # Extract time features from Start_Time
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Weekday'] = df['Start_Time'].dt.dayofweek
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_Year'] = df['Start_Time'].dt.year
    
    # Extract time features from Weather_Timestamp
    df['Weather_Hour'] = df['Weather_Timestamp'].dt.hour
    df['Weather_Weekday'] = df['Weather_Timestamp'].dt.dayofweek
    
    # Drop datetime columns
    df = df.drop(['Start_Time', 'End_Time', 'Weather_Timestamp'], axis=1)
    
    # Reorder columns to match the exact order from training (based on your features list)
    expected_columns = [
        'Source', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 'State', 'Timezone',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
        'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
        'Astronomical_Twilight', 'Duration_in_minutes', 'Start_Hour', 'Start_Weekday',
        'Start_Month', 'Start_Year', 'Weather_Hour', 'Weather_Weekday'
    ]
    
    # Ensure all columns exist and are in the correct order
    df = df[expected_columns]
    
    # Convert boolean columns to integers (True/False to 1/0)
    bool_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                    'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                    'Traffic_Calming', 'Traffic_Signal']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Ensure all numeric columns are float64
    for col in df.columns:
        if col not in bool_columns:
            df[col] = df[col].astype('float64')
    
    return df

# Load model
model, source_encoder, state_encoder, timezone_encoder, wind_direction_encoder, weather_encoder = load_model_and_encoders()

if model is None:
    st.stop()

# Title and description
st.title("🚗 Accident Severity Predictor")
st.markdown("### Predict traffic accident severity using machine learning")
st.markdown("---")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "ℹ️ About"])

with tab1:
    st.subheader("Enter Accident Details")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📍 Location Information")
        start_lat = st.number_input("Start Latitude", value=39.7392, format="%.6f")
        start_lng = st.number_input("Start Longitude", value=-104.9903, format="%.6f")
        distance = st.number_input("Distance (mi)", value=0.5, min_value=0.0, format="%.2f")
        
        # State selection (49 states from your data)
        state_list = ['OH', 'WV', 'CA', 'FL', 'GA', 'SC', 'NE', 'IA', 'IL', 'MO', 'WI', 'IN', 
                      'MI', 'NJ', 'NY', 'CT', 'MA', 'RI', 'NH', 'PA', 'KY', 'MD', 'VA', 'DC', 
                      'DE', 'TX', 'WA', 'OR', 'AL', 'NC', 'AZ', 'TN', 'LA', 'MN', 'CO', 'OK', 
                      'NV', 'UT', 'KS', 'NM', 'AR', 'MS', 'ME', 'VT', 'WY', 'ID', 'ND', 'MT', 'SD']
        state = st.selectbox("State", options=state_list, index=0)
        
        # Timezone options
        timezone_options = ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific']
        timezone = st.selectbox("Timezone", options=timezone_options, index=0)
        
        # Source options
        source = st.selectbox("Data Source", options=['Source1', 'Source2', 'Source3'], index=0)
    
    with col2:
        st.markdown("##### ⏰ Time Information")
        start_date = st.date_input("Start Date", value=datetime.now())
        start_time_input = st.time_input("Start Time", value=time(12, 0))
        end_time_input = st.time_input("End Time", value=time(12, 30))
        weather_time = st.time_input("Weather Observation Time", value=time(12, 0))
        
        # Combine date and time
        start_datetime = datetime.combine(start_date, start_time_input)
        end_datetime = datetime.combine(start_date, end_time_input)
        weather_datetime = datetime.combine(start_date, weather_time)
        
        st.markdown("##### 🌅 Daylight Conditions")
        sunrise_sunset = st.radio("Sunrise/Sunset", ["Day", "Night"], horizontal=True)
        sunrise_sunset_code = 0 if sunrise_sunset == "Day" else 1
        
        civil_twilight = st.radio("Civil Twilight", ["Day", "Night"], horizontal=True)
        civil_twilight_code = 0 if civil_twilight == "Day" else 1
        
        nautical_twilight = st.radio("Nautical Twilight", ["Day", "Night"], horizontal=True)
        nautical_twilight_code = 0 if nautical_twilight == "Day" else 1
        
        astronomical_twilight = st.radio("Astronomical Twilight", ["Day", "Night"], horizontal=True)
        astronomical_twilight_code = 0 if astronomical_twilight == "Day" else 1
    
    with col3:
        st.markdown("##### 🌤️ Weather Conditions")
        temperature = st.slider("Temperature (°F)", -20.0, 120.0, 70.0, 0.1)
        wind_chill = st.slider("Wind Chill (°F)", -30.0, 100.0, 65.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 0.1)
        pressure = st.slider("Pressure (in)", 28.0, 31.0, 29.92, 0.01)
        visibility = st.slider("Visibility (mi)", 0.0, 20.0, 10.0, 0.1)
        wind_speed = st.slider("Wind Speed (mph)", 0.0, 100.0, 10.0, 0.1)
        precipitation = st.slider("Precipitation (in)", 0.0, 5.0, 0.0, 0.01)
        
        wind_direction_list = ['Calm', 'SW', 'SSW', 'WSW', 'WNW', 'NW', 'West', 'NNW', 'NNE', 
                               'South', 'North', 'Variable', 'SE', 'SSE', 'ESE', 'East', 'NE', 
                               'ENE', 'E', 'W', 'S', 'VAR', 'CALM', 'N']
        wind_direction = st.selectbox("Wind Direction", options=wind_direction_list, index=0)
        
        # Weather condition selection (top 20 most common conditions)
        weather_condition_list = ['Blowing Dust', 'Blowing Snow', 'Clear', 'Cloudy', 'Drizzle', 
                                             'Fair', 'Fog', 'Hail', 'Haze', 'Heavy Rain', 'Heavy Snow', 
                                             'Heavy T-Storm', 'Light Drizzle', 'Light Freezing Rain', 
                                             'Light Rain', 'Light Snow', 'Mist', 'Mostly Cloudy', 
                                             'Overcast', 'Partly Cloudy', 'Rain', 'Scattered Clouds', 
                                             'Sleet', 'Smoke', 'Snow', 'T-Storm', 'Thunder', 
                                             'Thunderstorm', 'Tornado', 'Wintry Mix']
        weather_condition = st.selectbox("Weather Condition", options=weather_condition_list, index=0)
    
    # Road features section
    st.markdown("---")
    st.markdown("##### 🛣️ Road Features")
    
    col_road1, col_road2, col_road3, col_road4 = st.columns(4)
    
    with col_road1:
        amenity = st.checkbox("Amenity")
        bump = st.checkbox("Bump")
        crossing = st.checkbox("Crossing")
    
    with col_road2:
        give_way = st.checkbox("Give Way")
        junction = st.checkbox("Junction")
        no_exit = st.checkbox("No Exit")
    
    with col_road3:
        railway = st.checkbox("Railway")
        roundabout = st.checkbox("Roundabout")
        station = st.checkbox("Station")
    
    with col_road4:
        stop = st.checkbox("Stop Sign")
        traffic_calming = st.checkbox("Traffic Calming")
        traffic_signal = st.checkbox("Traffic Signal")
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Severity", type="primary", use_container_width=True):
        # Create input dataframe with original features
        input_data = pd.DataFrame({
            'Start_Time': [start_datetime],
            'End_Time': [end_datetime],
            'Weather_Timestamp': [weather_datetime],
            'Start_Lat': [start_lat],
            'Start_Lng': [start_lng],
            'Distance(mi)': [distance],
            'Temperature(F)': [temperature],
            'Wind_Chill(F)': [wind_chill],
            'Humidity(%)': [humidity],
            'Pressure(in)': [pressure],
            'Visibility(mi)': [visibility],
            'Wind_Speed(mph)': [wind_speed],
            'Precipitation(in)': [precipitation],
            'Wind_Direction': [wind_direction],
            'Weather_Condition': [weather_condition],
            'Sunrise_Sunset': [sunrise_sunset_code],
            'Civil_Twilight': [civil_twilight_code],
            'Nautical_Twilight': [nautical_twilight_code],
            'Astronomical_Twilight': [astronomical_twilight_code],
            'Amenity': [amenity],
            'Bump': [bump],
            'Crossing': [crossing],
            'Give_Way': [give_way],
            'Junction': [junction],
            'No_Exit': [no_exit],
            'Railway': [railway],
            'Roundabout': [roundabout],
            'Station': [station],
            'Stop': [stop],
            'Traffic_Calming': [traffic_calming],
            'Traffic_Signal': [traffic_signal],
            'Source': [source],
            'State': [state],
            'Timezone': [timezone]
        })
        
        # Apply feature engineering
        try:
            engineered_data = engineer_features(input_data, source_encoder, state_encoder, timezone_encoder, 
                                               wind_direction_encoder, weather_encoder)
            
            # Debug: Show the feature order
            with st.expander("🔍 Debug Info - Feature Order"):
                st.write("Engineered features shape:", engineered_data.shape)
                st.write("Feature names:", list(engineered_data.columns))
                st.dataframe(engineered_data)
            
            # Make prediction
            prediction = model.predict(engineered_data)[0]
            prediction_proba = model.predict_proba(engineered_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                severity_colors = {
                    1: ("🟢", "Low", "success"),
                    2: ("🟡", "Moderate", "warning"),
                    3: ("🟠", "High", "warning"),
                    4: ("🔴", "Very High", "error")
                }
                
                icon, severity_text, color_type = severity_colors.get(prediction, ("⚪", "Unknown", "info"))
                
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; color: black'>
                        <h1 style='font-size: 4em; margin: 0;'>{icon}</h1>
                        <h2 style='margin: 10px 0;'>Severity Level: {prediction}</h2>
                        <h3 style='color: #666;'>{severity_text} Impact</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("##### Confidence Scores")
                for i, prob in enumerate(prediction_proba, 1):
                    st.progress(float(prob), text=f"Level {i}: {prob*100:.1f}%")
            
            # Display interpretation
            st.markdown("---")
            st.markdown("### 📝 Interpretation")
            interpretations = {
                1: "Minor accident with minimal traffic impact. May cause slight delays.",
                2: "Moderate accident with noticeable traffic delays. Emergency response may be required.",
                3: "Serious accident with significant traffic disruption. Multiple lanes may be affected.",
                4: "Severe accident with major traffic impact. Extensive road closure likely."
            }
            st.info(interpretations.get(prediction, "Unknown severity level"))
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Input data shape:", engineered_data.shape)
            st.write("Expected features:", model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown")

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file with multiple accident records for batch prediction.")
    
    # Download sample template
    st.markdown("##### 📥 Download Sample Template")
    sample_data = pd.DataFrame({
        'Start_Time': ['2024-01-01 12:00:00'],
        'End_Time': ['2024-01-01 12:30:00'],
        'Weather_Timestamp': ['2024-01-01 12:00:00'],
        'Start_Lat': [39.7392],
        'Start_Lng': [-104.9903],
        'Distance(mi)': [0.5],
        'Temperature(F)': [70.0],
        'Wind_Chill(F)': [65.0],
        'Humidity(%)': [50.0],
        'Pressure(in)': [29.92],
        'Visibility(mi)': [10.0],
        'Wind_Speed(mph)': [10.0],
        'Precipitation(in)': [0.0],
        'Wind_Direction': ['Calm'],
        'Weather_Condition': ['Clear'],
        'Sunrise_Sunset': [0],
        'Civil_Twilight': [0],
        'Nautical_Twilight': [0],
        'Astronomical_Twilight': [0],
        'Amenity': [False],
        'Bump': [False],
        'Crossing': [True],
        'Give_Way': [False],
        'Junction': [True],
        'No_Exit': [False],
        'Railway': [False],
        'Roundabout': [False],
        'Station': [False],
        'Stop': [False],
        'Traffic_Calming': [False],
        'Traffic_Signal': [True],
        'Source': ['Source1'],
        'State': ['CA'],
        'Timezone': ['US/Pacific']
    })
    
    csv_template = sample_data.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="accident_prediction_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_data)} records")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(batch_data.head())
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    # Apply feature engineering
                    engineered_batch = engineer_features(batch_data, source_encoder, state_encoder, timezone_encoder,
                                                        wind_direction_encoder, weather_encoder)
                    
                    # Make predictions
                    predictions = model.predict(engineered_batch)
                    prediction_probas = model.predict_proba(engineered_batch)
                    
                    # Add predictions to original data
                    batch_data['Predicted_Severity'] = predictions
                    for i in range(prediction_probas.shape[1]):
                        batch_data[f'Severity_{i+1}_Probability'] = prediction_probas[:, i]
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(batch_data)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    severity_counts = pd.Series(predictions).value_counts().sort_index()
                    
                    with col1:
                        st.metric("Severity 1 (Low)", severity_counts.get(1, 0))
                    with col2:
                        st.metric("Severity 2 (Moderate)", severity_counts.get(2, 0))
                    with col3:
                        st.metric("Severity 3 (High)", severity_counts.get(3, 0))
                    with col4:
                        st.metric("Severity 4 (Very High)", severity_counts.get(4, 0))
                    
                    # Download results
                    csv_results = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_results,
                        file_name="accident_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.markdown("### About This Application")
    
    st.markdown("""
    This application uses a **Random Forest machine learning model** to predict the severity of traffic accidents 
    based on various factors including location, weather conditions, road features, and time of day.
    
    #### 🎯 Severity Levels
    - **Level 1 (Low)**: Minor impact, slight delays
    - **Level 2 (Moderate)**: Noticeable delays, emergency response may be needed
    - **Level 3 (High)**: Significant disruption, multiple lanes affected
    - **Level 4 (Very High)**: Major impact, extensive road closure likely
    
    #### 📊 Model Features
    The model considers 39 engineered features including:
    - **Location**: Latitude, longitude, distance, state, timezone
    - **Weather**: Temperature, wind chill, humidity, pressure, visibility, precipitation
    - **Time**: Hour, day of week, month, year, daylight conditions
    - **Road**: Traffic signals, crossings, junctions, and other road features
    
    #### 🔧 Feature Engineering
    The application automatically performs the following transformations:
    - Extracts temporal features (hour, weekday, month, year)
    - Calculates accident duration from start and end times
    - Processes categorical variables (state, timezone, weather conditions)
    - Converts boolean road features to numeric format
    
    #### 💡 Usage Tips
    - Ensure accurate GPS coordinates for better predictions
    - Weather conditions at the time of the accident are crucial
    - Road features (signals, crossings, etc.) significantly impact severity
    - Time of day affects traffic patterns and accident severity
    
    #### 📝 Data Source
    Model trained on the US Accidents dataset with comprehensive feature engineering.
    """)
    
    st.markdown("---")
    st.markdown("**Model Performance Metrics** ")
    st.code("""
    # Example metrics (replace with your actual model metrics)
    Accuracy: 86.40%
    """)

# Footer
st.markdown("---")
