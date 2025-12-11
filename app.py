import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import os

# Page configuration - Minimal interface
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide all the default Streamlit UI elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the dataset (hardcoded as requested)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('for_llm.csv', index_col=0)
        
        # Clean the data
        df_clean = df.copy()
        
        # Fix data types
        numeric_cols = ['mileage', 'price', 'co2', 'warranty', 'year']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Clean text columns
        text_cols = ['transmission', 'fuel', 'brand', 'model', 'brand_model']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

if df.empty:
    st.error("Dataset could not be loaded. Please check if 'for_llm.csv' exists in the same directory.")
    st.stop()

# Title
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #1E3A8A;">🚗 Car Price Predictor</h1>
    <p style="color: #666; margin-bottom: 2rem;">Predict car price using LLM based on similar vehicles</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for input form
col1, col2 = st.columns(2)

with col1:
    # Brand selection
    available_brands = ['peugeot', 'kia', 'volkswagen', 'audi', 'ford', 'mercedes', 'volvo', 'bmw']
    brand = st.selectbox("Brand", available_brands, key="brand")
    
    # Year selection (2014-2025)
    year = st.slider("Year", 2014, 2025, 2020, key="year")
    
    # Mileage selection
    mileage = st.slider("Mileage (km)", 0, 200000, 50000, step=1000, key="mileage")
    
    # Transmission selection
    transmission = st.radio("Transmission", ["manual", "automatic"], horizontal=True, key="transmission")

with col2:
    # Model selection based on brand
    if brand:
        # Filter models for the selected brand
        brand_models = sorted(df[df['brand'] == brand]['brand_model'].unique())
        # Clean up the model names for display
        display_models = [model.replace(f"{brand} ", "").upper() for model in brand_models]
        model_map = {display: original for display, original in zip(display_models, brand_models)}
        
        if display_models:
            selected_display = st.selectbox("Model", display_models, key="model_display")
            brand_model = model_map[selected_display]
        else:
            st.warning(f"No models found for {brand}")
            brand_model = ""
    else:
        brand_model = ""
    
    # CO2 input
    co2 = st.number_input("CO2 Emissions (g/km)", min_value=0, max_value=400, value=120, key="co2")
    
    # Emission Class input
    emission_classes = ['Euro 6', 'Euro 6d', 'Euro 6b', 'Euro 5', 'Euro 6d-TEMP', 'Euro 6e', 'Euro 6c']
    emission_class = st.selectbox("Emission Class", emission_classes, index=0, key="emission_class")

    # Fuel input
    fuel_options = ['petrol', 'hybrid - petrol', 'electric', 'diesel', 'hybrid', 'petrol super', 'hybrid - diesel']
    fuel = st.selectbox("Fuel", fuel_options, index=0, key="fuel")
    
    # Warranty input
    warranty = st.number_input("Warranty (months)", min_value=0, max_value=60, value=12, key="warranty")
    
    # Predict button
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# Function to filter similar cars
def filter_similar_cars(df, brand, brand_model, year, mileage, transmission, co2, warranty, emission_class, fuel):
    """
    Filter cars ensuring minimum 10 samples if available.
    Strictly enforces Brand and Model match.
    Relaxes other criteria if fewer than 10 cars are found.
    """
    
    try:
        year = int(year)
        mileage = float(mileage)
        co2 = float(co2)
        warranty = float(warranty)
    except:
        return pd.DataFrame()
    
    # 1. BASE MASK: STRICTLY enforce Brand and Brand_Model
    # We never deviate from this.
    base_mask = (
        (df['brand'] == brand.lower()) &
        (df['brand_model'] == brand_model.lower())
    )
    base_df = df[base_mask].copy()

    # If the database simply doesn't have 10 cars of this specific model total,
    # we return whatever we have.
    if len(base_df) < 10:
        if not base_df.empty:
            base_df['year_diff'] = abs(base_df['year'] - year)
            base_df['mileage_diff'] = abs(base_df['mileage'] - mileage)
            return base_df.sort_values(['year_diff', 'mileage_diff'])
        return pd.DataFrame()

    # 2. ATTEMPT 1: Strict Match (Original Logic)
    mask_1 = (
        (base_df['year'] >= year - 1) & (base_df['year'] <= year + 1) &
        (base_df['mileage'] <= mileage) &
        (base_df['mileage'] >= mileage*0.7) &
        (base_df['transmission'] == transmission.lower()) &
        (base_df['co2'] >= co2 * 0.8) & (base_df['co2'] <= co2 * 1.2) &
        (base_df["fuel"] == fuel.lower()) &
        (base_df['emission_class'] == emission_class)
    )
    
    df_1 = base_df[mask_1]
    if len(df_1) >= 10:
        df_1 = df_1.copy()
        df_1['year_diff'] = abs(df_1['year'] - year)
        df_1['mileage_diff'] = abs(df_1['mileage'] - mileage)
        return df_1.sort_values(['year_diff', 'mileage_diff']).head(20)

    # 3. ATTEMPT 2: Moderate Relaxation
    # Relax: Year (+/- 2), Mileage (allow up to 20% more), Drop CO2, Warranty, Emission constraints
    # Keep: Transmission, Fuel
    mask_2 = (
        (base_df['year'] >= year - 2) & (base_df['year'] <= year + 2) &
        (base_df['mileage'] <= mileage ) &
        (base_df['transmission'] == transmission.lower()) &
        (base_df['fuel'] == fuel.lower())
    )
    
    df_2 = base_df[mask_2]
    if len(df_2) >= 10:
        df_2 = df_2.copy()
        df_2['year_diff'] = abs(df_2['year'] - year)
        df_2['mileage_diff'] = abs(df_2['mileage'] - mileage)
        return df_2.sort_values(['year_diff', 'mileage_diff']).head(20)

    # 4. ATTEMPT 3: High Relaxation
    # Relax: Year (+/- 4), Mileage (allow up to 100% more / double)
    # Relax: Drop Transmission and Fuel constraints (to find ENOUGH cars)
    mask_3 = (
        # (base_df['year'] >= year - 4) & (base_df['year'] <= year + 4) &
        (base_df['mileage'] <= mileage * 2.0)
    )
    
    df_3 = base_df[mask_3]
    if len(df_3) >= 10:
        df_3 = df_3.copy()
        df_3['year_diff'] = abs(df_3['year'] - year)
        df_3['mileage_diff'] = abs(df_3['mileage'] - mileage)
        # We prefer matches with correct transmission/fuel even in this loose set
        df_3['same_trans'] = (df_3['transmission'] == transmission.lower()).astype(int)
        df_3['same_fuel'] = (df_3['fuel'] == fuel.lower()).astype(int)
        
        # Sort by: Same Trans (desc), Same Fuel (desc), Year Diff (asc), Mileage Diff (asc)
        return df_3.sort_values(
            ['same_trans', 'same_fuel', 'year_diff', 'mileage_diff'], 
            ascending=[False, False, True, True]
        ).head(20)

    # 5. ATTEMPT 4: Ultimate Fallback
    # Just return the closest matches from the base_df (Fixed Brand/Model)
    df_final = base_df.copy()
    df_final['year_diff'] = abs(df_final['year'] - year)
    df_final['mileage_diff'] = abs(df_final['mileage'] - mileage)
    
    return df_final.sort_values(['year_diff', 'mileage_diff']).head(20)

# Function to call GPT-4 for price prediction
def predict_price_with_gpt(filtered_cars, brand, model, year, mileage, transmission, co2, warranty):
    """Use GPT-4 to predict price based on similar cars"""
    
    # Check if we have similar cars
    if filtered_cars.empty:
        return "Insufficient data for prediction. Please try different parameters."
    
    # Prepare the data for the prompt
    similar_cars_info = []
    for _, car in filtered_cars.iterrows():
        similar_cars_info.append(
            f"- {car['brand_model'].title()} ({car['year']}): {car['mileage']:,.0f} km, "
            f"{car['transmission']}, €{car['price']:,.0f}, CO2: {car['co2']}g/km, "
            f"Warranty: {car['warranty']} months"
        )
    
    similar_cars_text = "\n".join(similar_cars_info)
    
    # Prepare the prompt
    prompt = f"""
    I need to predict the price of a car based on similar vehicles in the market.
    
    **Target Car Specifications:**
    - Brand: {brand.title()}
    - Model: {model.title()}
    - Year: {year}
    - Mileage: {mileage:,.0f} km
    - Transmission: {transmission}
    - CO2 Emissions: {co2} g/km
    - Warranty: {warranty} months
    
    **Similar Cars in Market (for reference):**
    {similar_cars_text}
    
    **Task:**
    Based ONLY on the similar cars above, predict a fair market price for the target car.
    Consider:
    1. Age depreciation (newer cars cost more)
    2. Mileage impact (higher mileage = lower price)
    3. Transmission type value
    4. Warranty value
    5. Overall condition based on specifications
    
    **Important Instructions:**
    - Return ONLY the predicted price in EUROS as a single number (e.g., 24500)
    - Do NOT include any text, explanations, or currency symbols
    - Do NOT use commas in the number
    - Base your prediction SOLELY on the similar cars data provided
    - If similar cars data is insufficient, return "0"
    
    **Predicted Price (€):**
    """
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
        
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a car pricing expert. You predict car prices based on market data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Zero temperature for deterministic output
            max_tokens=10
        )
        
        # Extract the price
        predicted_price = response.choices[0].message.content.strip()
        
        # Clean the response - remove any non-numeric characters except decimal point
        predicted_price = ''.join(c for c in predicted_price if c.isdigit() or c == '.')
        
        # Try to convert to float
        try:
            price_float = float(predicted_price)
            if price_float > 0:
                return f"€{price_float:,.0f}"
            else:
                return "€0 (Insufficient data)"
        except:
            return "€0 (Prediction error)"
            
    except Exception as e:
        # Fallback to statistical prediction if API fails
        return calculate_statistical_price(filtered_cars, year, mileage)

# Function for statistical fallback price prediction
def calculate_statistical_price(filtered_cars, year, mileage):
    """Calculate price statistically if GPT fails"""
    if filtered_cars.empty:
        return "€0"
    
    # Calculate base price from similar cars
    avg_price = filtered_cars['price'].mean()
    median_price = filtered_cars['price'].median()
    
    # Adjust for year (depreciation ~15% per year)
    avg_year = filtered_cars['year'].mean()
    year_diff = year - avg_year
    year_adjustment = avg_price * (0.85 ** abs(year_diff))
    
    # Adjust for mileage (depreciation ~€0.10 per km from average)
    avg_mileage = filtered_cars['mileage'].mean()
    mileage_diff = mileage - avg_mileage
    mileage_adjustment = mileage_diff * 0.10
    
    # Calculate final price
    if year_diff > 0:  # Target car is newer
        base_price = max(avg_price, median_price, year_adjustment)
    else:  # Target car is older
        base_price = min(avg_price, median_price, year_adjustment)
    
    final_price = base_price - mileage_adjustment
    
    # Ensure price is reasonable
    min_price = filtered_cars['price'].min() * 0.7
    max_price = filtered_cars['price'].max() * 1.3
    
    final_price = max(min_price, min(final_price, max_price))
    
    return f"€{final_price:,.0f}"

# Main prediction logic
if predict_button:
    if not brand_model:
        st.error("Please select a model")
    else:
        # Show loading spinner
        with st.spinner("🔍 Finding similar cars and predicting price..."):
            # Filter similar cars
            similar_cars = filter_similar_cars(
                df, brand, brand_model, year, mileage, 
                transmission, co2, warranty, emission_class, fuel
            )

            # Display how many similar cars found
            st.info(f"Found {len(similar_cars)} similar vehicles for reference")
            
            if len(similar_cars) < 5:
                st.warning("⚠️ High uncertainty: Very few similar cars found even with relaxed criteria.")

            # Predict price
            predicted_price_str = predict_price_with_gpt(
                similar_cars, brand, brand_model, year, 
                mileage, transmission, co2, warranty
            )
            
            # --- CALCULATE RANGE ---
            # Extract number from string (e.g., "€24,500" -> 24500.0)
            try:
                clean_price = ''.join(c for c in predicted_price_str if c.isdigit() or c == '.')
                price_val = float(clean_price)
                
                if price_val > 0:
                    min_range = price_val * 0.90
                    max_range = price_val * 1.10
                    range_display = f"Estimated Range: €{min_range:,.0f} - €{max_range:,.0f}"
                else:
                    range_display = "Range unavailable"
            except:
                range_display = "Range unavailable"

            # Display the predicted price prominently
            st.markdown("---")
            
            # Create a nice price display with Range
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h3 style="margin: 0; font-size: 1.2rem; opacity: 0.9;">PREDICTED PRICE</h3>
                <h1 style="margin: 1rem 0; font-size: 3.5rem; font-weight: bold;">{predicted_price_str}</h1>
                <h4 style="margin: 0; font-size: 1.2rem; font-weight: normal; color: #E2E8F0;">{range_display}</h4>
                <p style="margin-top: 1rem; opacity: 0.8; font-size: 0.9rem;">
                    Based on {len(similar_cars)} similar vehicles in market
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Optionally show similar cars (collapsed by default)
            with st.expander("📊 View Similar Cars Used for Prediction"):
                if not similar_cars.empty:
                    # Format the display
                    display_df = similar_cars[[
                        'brand_model', 'year', 'mileage', 'transmission', 
                        'price', 'co2', 'warranty', 'fuel'
                    ]].copy()
                    
                    display_df['brand_model'] = display_df['brand_model'].str.title()
                    display_df['mileage'] = display_df['mileage'].apply(lambda x: f"{x:,.0f} km")
                    display_df['transmission'] = display_df['transmission'].str.title()
                    display_df['price'] = display_df['price'].apply(lambda x: f"€{x:,.0f}")
                    display_df['co2'] = display_df['co2'].apply(lambda x: f"{x:.0f} g/km")
                    display_df['warranty'] = display_df['warranty'].apply(lambda x: f"{x:.0f} months")
                    display_df['fuel'] = display_df['fuel'].str.title()
                    
                    display_df = display_df.rename(columns={
                        'brand_model': 'Model',
                        'year': 'Year',
                        'mileage': 'Mileage',
                        'transmission': 'Transmission',
                        'price': 'Price',
                        'co2': 'CO2',
                        'warranty': 'Warranty',
                        'fuel': 'Fuel'
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write("No similar cars found matching all criteria.")

            # --- DOWNLOAD BUTTON ---
            if not similar_cars.empty:
                st.markdown("###") # Spacer
                csv = similar_cars.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv,
                    file_name=f'filtered_cars_{brand}_{brand_model}.csv',
                    mime='text/csv',
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "Car Price Predictor using LLM | Based on similar vehicle market data"
    "</div>",
    unsafe_allow_html=True
)