import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load model
# -------------------------
with open('../linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="📊 Advertising Sales Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar for inputs
# -------------------------
st.sidebar.header("💰 Set Your Advertising Budget")
tv = st.sidebar.slider("TV Budget ($1000s)", 0.0, 500.0, 100.0)
radio = st.sidebar.slider("Radio Budget ($1000s)", 0.0, 50.0, 20.0)
newspaper = st.sidebar.slider("Newspaper Budget ($1000s)", 0.0, 100.0, 10.0)

# -------------------------
# Main Title
# -------------------------
st.title("📊 Advertising Sales Predictor")
st.markdown(
    """
Predict sales based on advertising budgets for **TV**, **Radio**, and **Newspaper**.
Adjust your budgets using the sliders in the sidebar.
"""
)

# -------------------------
# Predict Button & Metrics
# -------------------------
if st.button("🚀 Predict Sales"):
    X_new = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(X_new)[0]

    # Use metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric("TV Budget ($1000s)", f"{tv}")
    col2.metric("Radio Budget ($1000s)", f"{radio}")
    col3.metric("Newspaper Budget ($1000s)", f"{newspaper}")

    st.success(f"✅ Predicted Sales: **{prediction:.2f} thousand units**")

# -------------------------
# Load dataset for visualization
# -------------------------
df = pd.read_csv('../advertising.csv')

# -------------------------
# Feature Influence Charts
# -------------------------
st.subheader("📈 Feature Influence on Sales")
col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(5,4))
    sns.scatterplot(data=df, x='TV', y='Sales', color='crimson')
    ax.set_title("TV vs Sales")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5,4))
    sns.scatterplot(data=df, x='Radio', y='Sales', color='royalblue')
    ax.set_title("Radio vs Sales")
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(5,4))
    sns.scatterplot(data=df, x='Newspaper', y='Sales', color='green')
    ax.set_title("Newspaper vs Sales")
    st.pyplot(fig)

# -------------------------
# Actual vs Predicted Sales
# -------------------------
st.subheader("🔍 Actual vs Predicted Sales (Top 10)")
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
y_pred = model.predict(X)

comparison = pd.DataFrame({
    "Actual Sales": y,
    "Predicted Sales": y_pred
})
st.dataframe(comparison.head(10))

# -------------------------
# Optional: Heatmap
# -------------------------
st.subheader("🌡️ Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig)
