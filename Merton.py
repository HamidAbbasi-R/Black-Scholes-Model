import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Title and Introduction
st.title("Derivation of the Merton Model for Corporate Default Risk")
st.markdown("""
The **Merton Model** is a structural credit risk model that estimates the probability of default (PD) of a firm based on its asset value, debt structure, and market conditions. This dashboard explains the model step-by-step and allows you to interactively explore its components.
""")

# Step 1: Basic Principles of the Merton Model
st.header("Step 1: Key Principles of the Merton Model")
st.markdown("""
The Merton Model is based on the following principles:
1. A firm's **equity** can be viewed as a call option on its assets.
2. Default occurs when the firm's **asset value** falls below the face value of its debt at maturity.
3. The **probability of default (PD)** is derived using the Black-Scholes-Merton framework.

#### Key Variables:
- $ V $: Firm's asset value
- $ D $: Face value of the firm's debt (default boundary)
- $ \\sigma $: Volatility of the firm's assets
- $ T $: Time to maturity
- $ r $: Risk-free interest rate

#### Outputs:
- **Distance to Default (DD)**: How far the firm's asset value is from the default point.
- **Probability of Default (PD)**: Derived from the standard normal cumulative distribution function (CDF).
""")

# Step 2: Mathematical Formulation
st.header("Step 2: Mathematical Formulation")
st.markdown("""
The Merton Model uses the Black-Scholes-Merton option pricing framework to calculate the equity value and derive the probability of default.

1. **Equity Value as a Call Option**:
   The equity value $ E $ is modeled as a call option on the firm's assets:
   $$
   E = V \\cdot N(d_1) - D \\cdot e^{-rT} \\cdot N(d_2)
   $$
   where:
   $$
   d_1 = \\frac{\\ln(V / D) + (r + \\sigma^2 / 2)T}{\\sigma \\sqrt{T}}, \\quad
   d_2 = d_1 - \\sigma \\sqrt{T}
   $$

2. **Distance to Default (DD)**:
   The distance to default measures how far the firm's asset value is from the default point:
   $$
   DD = \\frac{\ln(V / D) + (r - \\sigma^2 / 2)T}{\\sigma \\sqrt{T}}
   $$

3. **Probability of Default (PD)**:
   The probability of default is derived from the standard normal CDF:
   $$
   PD = N(-DD)
   $$
""")

# Step 3: Interactive Inputs
st.header("Step 3: Interactive Exploration of the Merton Model")
st.markdown("""
Adjust the parameters below to see how they affect the Distance to Default (DD) and Probability of Default (PD).
""")

# Input widgets
V = st.slider("Firm's Asset Value ($ V $)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
D = st.slider("Face Value of Debt ($ D $)", min_value=50.0, max_value=200.0, value=80.0, step=1.0)
sigma = st.slider("Volatility of Assets ($ \sigma $)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
T = st.slider("Time to Maturity ($ T $)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
r = st.slider("Risk-Free Rate ($ r $)", min_value=0.0, max_value=0.1, value=0.05, step=0.01)

# Step 4: Calculations
def merton_model(V, D, sigma, T, r):
    """
    Calculate the Distance to Default (DD) and Probability of Default (PD) using the Merton Model.
    """
    # Calculate d1 and d2
    d1 = (np.log(V / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Distance to Default (DD)
    DD = (np.log(V / D) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Probability of Default (PD)
    PD = norm.cdf(-DD)
    
    return DD, PD

# Calculate DD and PD
DD, PD = merton_model(V, D, sigma, T, r)

# Display results
st.subheader("Results")
st.markdown(f"""
- **Distance to Default (DD)**: {DD:.2f}
- **Probability of Default (PD)**: {PD:.4f}
""")

# Step 5: Visualization
st.header("Step 5: Visualizing the Merton Model")
st.markdown("""
The plot below shows the relationship between the firm's asset value and its equity value. The red dashed line represents the default boundary (present value of debt).
""")

# Generate data for plotting
asset_values = np.linspace(0.1, 2 * V, 500)
default_boundary = D * np.exp(-r * T)  # Present value of debt

# Calculate equity values
d1 = (np.log(asset_values / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
equity_values = asset_values * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

# Create Plotly figure
fig = go.Figure()

# Add equity value curve
fig.add_trace(go.Scatter(
    x=asset_values,
    y=equity_values,
    mode='lines',
    name="Equity Value",
    line=dict(color="blue", width=2)
))

# Add default boundary
fig.add_trace(go.Scatter(
    x=[default_boundary, default_boundary],
    y=[0, max(equity_values)],
    mode='lines',
    name="Default Boundary",
    line=dict(color="red", dash="dash")
))

# Update layout
fig.update_layout(
    title="Merton Model: Equity Value vs. Asset Value",
    xaxis_title="Firm's Asset Value",
    yaxis_title="Equity Value",
    template="plotly_white"
)

# Display the plot
st.plotly_chart(fig)

# Step 6: Interpretation
st.header("Step 6: Interpretation")
st.markdown("""
- **Asset Value ($ V $)**: As the firm's asset value increases relative to its debt, the probability of default decreases.
- **Debt Value ($ D $)**: Higher debt levels increase the likelihood of default.
- **Volatility ($ \sigma $)**: Higher volatility increases uncertainty, raising the probability of default.
- **Time to Maturity ($ T $)**: Longer time to maturity gives the firm more time to recover, reducing the probability of default.
- **Risk-Free Rate ($ r $)**: A higher risk-free rate reduces the present value of debt, slightly lowering the default probability.
""")