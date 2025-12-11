import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
.stCheckbox {padding-top: 5px;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    try:
        if not os.path.exists('for_llm.csv'):
            return pd.DataFrame()

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

# Helper function to get most frequent value (Mode)
def get_most_frequent_value(df, brand_model, column_name):
    """Returns the most frequent value for a column given a car model."""
    # Try to find mode for specific model
    subset = df[df['brand_model'] == brand_model]
    
    # If model has no data, fall back to brand
    if subset.empty:
        subset = df[df['brand'] == df[df['brand_model'] == brand_model]['brand'].iloc[0]]
    
    # If still empty (shouldn't happen), take global mode
    if subset.empty:
        subset = df
        
    try:
        return subset[column_name].mode().iloc[0]
    except:
        return df[column_name].mode().iloc[0] # Ultimate fallback

# Load data
df = load_data()

if df.empty:
    st.error("Dataset could not be loaded. Please check if 'for_llm.csv' exists in the same directory.")
    st.stop()

# Title
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #1E3A8A;">🚗 Car Price Predictor</h1>
    <p style="color: #666; margin-bottom: 2rem;">Compare Statistical vs. AI-Powered Valuations</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for input form
col1, col2 = st.columns(2)

# --- COLUMN 1 INPUTS ---
with col1:
    # Brand selection
    available_brands = ['peugeot', 'kia', 'volkswagen', 'audi', 'ford', 'mercedes', 'volvo', 'bmw']
    brand = st.selectbox("Brand", available_brands, key="brand")
    
    # Year selection
    st.write("Year")
    year_unknown = st.checkbox("Don't know / Auto-fill", key="year_unk")
    if year_unknown:
        year_input = None
        st.info("Will use most common Year")
    else:
        year_input = st.slider("Select Year", 2014, 2025, 2020, label_visibility="collapsed")
    
    # Mileage selection
    st.write("Mileage (km)")
    mileage_unknown = st.checkbox("Don't know / Auto-fill", key="mile_unk")
    if mileage_unknown:
        mileage_input = None
        st.info("Will use most common Mileage")
    else:
        mileage_input = st.slider("Select Mileage", 0, 200000, 50000, step=1000, label_visibility="collapsed")
    
    # Transmission selection
    trans_options = ["manual", "automatic", "Unknown"]
    transmission_input = st.radio("Transmission", trans_options, horizontal=True, index=1)

# --- COLUMN 2 INPUTS ---
with col2:
    # Model selection based on brand
    if brand:
        brand_models = sorted(df[df['brand'] == brand]['brand_model'].unique())
        display_models = [model.replace(f"{brand} ", "").upper() for model in brand_models]
        model_map = {display: original for display, original in zip(display_models, brand_models)}
        
        if display_models:
            selected_display = st.selectbox("Model", display_models, key="model_display")
            brand_model = model_map[selected_display]
        else:
            brand_model = ""
    else:
        brand_model = ""
    
    # CO2 input
    st.write("CO2 Emissions (g/km)")
    co2_unknown = st.checkbox("Don't know / Auto-fill", key="co2_unk")
    if co2_unknown:
        co2_input = None
        st.info("Will use average CO2")
    else:
        co2_input = st.number_input("Enter CO2", min_value=0, max_value=400, value=120, label_visibility="collapsed")
    
    # Emission Class input
    emission_classes = ['Euro 6', 'Euro 6d', 'Euro 6b', 'Euro 5', 'Euro 6d-TEMP', 'Euro 6e', 'Euro 6c', 'Unknown']
    emission_class_input = st.selectbox("Emission Class", emission_classes, index=0)

    # Fuel input
    fuel_options = ['petrol', 'hybrid - petrol', 'electric', 'diesel', 'hybrid', 'petrol super', 'hybrid - diesel', 'Unknown']
    fuel_input = st.selectbox("Fuel", fuel_options, index=0)
    
    # Warranty input
    st.write("Warranty (months)")
    warranty_unknown = st.checkbox("Don't know / Auto-fill", key="war_unk")
    if warranty_unknown:
        warranty_input = None
        st.info("Will use standard warranty")
    else:
        warranty_input = st.number_input("Enter Warranty", min_value=0, max_value=60, value=12, label_visibility="collapsed")
    
    # Predict button
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# --- LOGIC TO HANDLE "UNKNOWN" VALUES ---
def prepare_search_values(df, brand_model):
    """
    Checks inputs. If 'Unknown' or None, calculates the mode/mean 
    from the dataset for that specific car model.
    """
    values = {}
    
    # Year
    if year_unknown or year_input is None:
        values['year'] = int(get_most_frequent_value(df, brand_model, 'year'))
    else:
        values['year'] = int(year_input)
        
    # Mileage
    if mileage_unknown or mileage_input is None:
        values['mileage'] = float(get_most_frequent_value(df, brand_model, 'mileage'))
    else:
        values['mileage'] = float(mileage_input)
        
    # Transmission
    if transmission_input == "Unknown":
        values['transmission'] = get_most_frequent_value(df, brand_model, 'transmission')
    else:
        values['transmission'] = transmission_input.lower()
        
    # CO2
    if co2_unknown or co2_input is None:
        values['co2'] = float(df[df['brand_model'] == brand_model]['co2'].mean()) # Use mean for CO2
        if pd.isna(values['co2']): values['co2'] = 120.0
    else:
        values['co2'] = float(co2_input)
        
    # Emission
    if emission_class_input == "Unknown":
        values['emission_class'] = get_most_frequent_value(df, brand_model, 'emission_class')
    else:
        values['emission_class'] = emission_class_input
        
    # Fuel
    if fuel_input == "Unknown":
        values['fuel'] = get_most_frequent_value(df, brand_model, 'fuel')
    else:
        values['fuel'] = fuel_input.lower()
        
    # Warranty
    if warranty_unknown or warranty_input is None:
        values['warranty'] = float(get_most_frequent_value(df, brand_model, 'warranty'))
    else:
        values['warranty'] = float(warranty_input)
        
    return values

# Function to filter similar cars (UPDATED LOGIC)
def filter_similar_cars(df, brand, brand_model, vals):
    """
    Filter Logic:
    1. Strict Match.
    2. If Count >= 5: STOP and Return Top 10.
    3. Else: Relax Criteria.
    4. If Count >= 5: STOP and Return Top 10.
    5. Else: Super Relaxed.
    """
    
    # 1. BASE MASK: STRICTLY enforce Brand and Brand_Model
    base_mask = (
        (df['brand'] == brand.lower()) &
        (df['brand_model'] == brand_model.lower())
    )
    base_df = df[base_mask].copy()

    # Pre-calculate diffs for sorting later
    if not base_df.empty:
        base_df['year_diff'] = abs(base_df['year'] - vals['year'])
        base_df['mileage_diff'] = abs(base_df['mileage'] - vals['mileage'])

    # --- LEVEL 1: STRICT MATCH ---
    mask_1 = (
        (base_df['year'] >= vals['year'] - 1) & (base_df['year'] <= vals['year'] + 1) &
        (base_df['mileage'] <= vals['mileage'] * 1.1) &
        (base_df['mileage'] >= vals['mileage'] * 0.1) &
        (base_df['transmission'] == vals['transmission']) &
        (base_df['co2'] >= vals['co2'] * 0.8) & (base_df['co2'] <= vals['co2'] * 1.2) &
        (base_df["fuel"] == vals['fuel']) &
        (base_df['emission_class'] == vals['emission_class'])
    )
    
    df_1 = base_df[mask_1]
    
    # LOGIC: If 5 exist, then OK. Max 10 needed.
    if len(df_1) >= 5:
        return df_1.sort_values(['year_diff', 'mileage_diff']).head(10)

    # --- LEVEL 2: MODERATE RELAXATION ---
    # Relax Year, Mileage, CO2. Keep Trans/Fuel strict if possible.
    mask_2 = (
        (base_df['transmission'] == vals['transmission']) &
        (base_df['fuel'] == vals['fuel'])
    )
    
    df_2 = base_df[mask_2]
    
    if len(df_2) >= 5:
        return df_2.sort_values(['year_diff', 'mileage_diff']).head(10)

    # --- LEVEL 3: HIGH RELAXATION ---
    # Just filter by mileage cap, keep Model fixed.
    mask_3 = (
        (base_df['mileage'] <= vals['mileage'] * 2.0)
    )
    
    df_3 = base_df[mask_3]
    
    if len(df_3) >= 5:
        # Sort so that if we have same transmission/fuel matches, they come first
        df_3['same_trans'] = (df_3['transmission'] == vals['transmission']).astype(int)
        df_3['same_fuel'] = (df_3['fuel'] == vals['fuel']).astype(int)
        
        return df_3.sort_values(
            ['same_trans', 'same_fuel', 'year_diff', 'mileage_diff'], 
            ascending=[False, False, True, True]
        ).head(10)

    # --- LEVEL 4: FALLBACK ---
    # Return whatever we have for this Model
    return base_df.sort_values(['year_diff', 'mileage_diff']).head(10)

# Function to call GPT-4
def predict_price_with_gpt(filtered_cars, brand, model, vals):
    if filtered_cars.empty:
        return "Insufficient data"
    
    similar_cars_info = []
    for _, car in filtered_cars.iterrows():
        similar_cars_info.append(
            f"- {car['brand_model'].title()} ({car['year']}): {car['mileage']:,.0f} km, "
            f"{car['transmission']}, €{car['price']:,.0f}, CO2: {car['co2']}g/km"
        )
    
    similar_cars_text = "\n".join(similar_cars_info)
    
    prompt = f"""
    Predict price for:
    {brand.title()} {model.title()}, Year: {vals['year']}, Mileage: {vals['mileage']:,.0f} km, 
    Trans: {vals['transmission']}, Fuel: {vals['fuel']}, CO2: {vals['co2']}
    
    Based ONLY on these market examples:
    {similar_cars_text}
    
    Return ONLY the price as a number (e.g. 25000). Return 0 if data insufficient.
    """
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: return "Error: API Key Missing"

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=10
        )
        
        pred = response.choices[0].message.content.strip()
        pred = ''.join(c for c in pred if c.isdigit() or c == '.')
        price = float(pred)
        return f"€{price:,.0f}" if price > 0 else "€0"
            
    except:
        return "API Error"

# Statistical Calculation
def calculate_statistical_price(filtered_cars, vals):
    if filtered_cars.empty: return "€0"
    
    avg_price = filtered_cars['price'].mean()
    median_price = filtered_cars['price'].median()
    
    # Depreciation adjustments
    year_diff = vals['year'] - filtered_cars['year'].mean()
    mileage_diff = vals['mileage'] - filtered_cars['mileage'].mean()
    
    base_price = median_price if year_diff < 0 else avg_price
    
    # Simple adjustment: 5% per year, 0.05 per km
    final_price = base_price * (1.05 ** year_diff) - (mileage_diff * 0.05)
    
    # Safety bounds
    min_p = filtered_cars['price'].min() * 0.8
    max_p = filtered_cars['price'].max() * 1.2
    final_price = max(min_p, min(final_price, max_p))
    
    return f"€{final_price:,.0f}"

def get_price_range(price_str):
    try:
        val = float(''.join(c for c in price_str if c.isdigit() or c == '.'))
        if val > 0: return f"€{val*0.9:,.0f} - €{val*1.1:,.0f}"
    except: pass
    return "N/A"

# Main prediction logic
if predict_button:
    if not brand_model:
        st.error("Please select a model")
    else:
        with st.spinner("🔍 Imputing missing data & analyzing market..."):
            
            # 1. Prepare Data (Handle Unknowns)
            search_vals = prepare_search_values(df, brand_model)
            
            # 2. Get Data
            similar_cars = filter_similar_cars(df, brand, brand_model, search_vals)

            st.info(f"Found {len(similar_cars)} similar vehicles for reference")
            
            # 3. Run Models
            stats_price_str = calculate_statistical_price(similar_cars, search_vals)
            llm_price_str = predict_price_with_gpt(similar_cars, brand, brand_model, search_vals)

            # 4. Display
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background-color: #f0f2f6; border-radius: 10px; border: 1px solid #d1d5db;">
                    <h4 style="margin: 0; color: #4b5563; font-size: 1rem;">STATISTICAL</h4>
                    <h2 style="margin: 0.5rem 0; color: #1f2937; font-size: 2.2rem;">{stats_price_str}</h2>
                    <p style="color: #6b7280; font-size: 0.8rem;">Range: {get_price_range(stats_price_str)}</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                    <h4 style="margin: 0; opacity: 0.9; font-size: 1rem;">AI MODEL</h4>
                    <h2 style="margin: 0.5rem 0; font-size: 2.2rem; font-weight: bold;">{llm_price_str}</h2>
                    <p style="opacity: 0.8; font-size: 0.8rem;">Range: {get_price_range(llm_price_str)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📊 View Data Used"):
                if not similar_cars.empty:
                    st.dataframe(similar_cars[['brand_model', 'year', 'mileage', 'transmission', 'price', 'fuel',"emission_class"]], use_container_width=True, hide_index=True)
                else:
                    st.write("No matching data.")

            if not similar_cars.empty:
                st.markdown("###") 
                csv = similar_cars.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "filtered_cars.csv", "text/csv", use_container_width=True)