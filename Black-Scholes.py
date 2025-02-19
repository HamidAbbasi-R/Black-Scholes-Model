import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sliders for user input
S_max = st.sidebar.slider("Stock Price (S)", min_value=1, max_value=200, value=100, step=1)
K_max = st.sidebar.slider("Strike Price (K)", min_value=1, max_value=200, value=100, step=1)
S = np.linspace(1, S_max, 100)
K = np.linspace(1, K_max, 100)
S, K = np.meshgrid(S, K)
r = st.sidebar.slider("Risk-Free Rate % (r)", min_value=0, max_value=20, value=5, step=1) /100
sigma = st.sidebar.slider("Volatility % (σ)", min_value=1, max_value=100, value=20, step=1) / 100
T = st.sidebar.slider("Time to Maturity, years (T)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Title of the Dashboard
st.title("Interactive Black-Scholes Formula Derivation")
st.markdown("""
The **Black-Scholes model** is one of the most important tools in finance, used to price **European options**. Before diving into the derivation, let's briefly summarize what a European option is and why the Black-Scholes model is designed for it:

#### What is a European Option?
A **European option** is a financial derivative that gives the holder the right, but not the obligation, to buy (call) or sell (put) an underlying asset at a fixed **strike price**, but **only on the expiration date**. Key points:
- It cannot be exercised early (unlike American options).
- Its payoff depends on the price of the underlying asset at maturity:
  - **Call Option Payoff**: $ \max(S_T - K, 0) $
  - **Put Option Payoff**: $ \max(K - S_T, 0) $
- The simplicity of European options allows for a closed-form pricing solution.

#### Why Use the Black-Scholes Model?
The Black-Scholes model is specifically designed for European options because:
- It assumes the option can only be exercised at maturity, simplifying the pricing process.
- It uses risk-neutral valuation and the Feynman-Kac theorem to derive a closed-form solution.
- The model assumes stock prices follow **geometric Brownian motion**, ensuring positivity and capturing exponential growth.

Now, let’s explore the assumptions and step-by-step derivation of the Black-Scholes formula.
""")

# Step 1: Assumptions of the Black-Scholes Model
st.header("Step 1: Assumptions of the Black-Scholes Model")
st.markdown("""
The Black-Scholes model is based on several key assumptions:
1. The stock price follows a geometric Brownian motion.
2. There are no transaction costs or taxes.
3. The risk-free interest rate is constant.
4. No dividends are paid during the life of the option.
5. Markets are efficient (no arbitrage opportunities).
6. The option can only be exercised at maturity (European option).
""")

# Step 2: Stock Price Dynamics
st.header("Step 2: Stock Price Dynamics and Geometric Brownian Motion")
st.markdown("""
To model stock prices, we use **geometric Brownian motion (GBM)** rather than standard Brownian motion. Let's explore why this is the case and derive the GBM equation step-by-step.

#### Why Not Standard Brownian Motion?
Standard Brownian motion $ W_t $ has the following properties:
1. It has independent, normally distributed increments.
2. Its mean is zero, and its variance grows linearly with time ($ \\text{Var}(W_t) = t $).
3. It can take negative values.

However, stock prices cannot be negative in reality. If we were to model stock prices using standard Brownian motion:
$$
S_t = S_0 + \\mu t + \\sigma W_t
$$
The term $ \\sigma W_t $ could cause $ S_t $ to become negative, which is unrealistic for stock prices. Additionally, stock prices tend to grow exponentially over time due to compounding returns, which standard Brownian motion does not capture.

#### Why Geometric Brownian Motion?
Geometric Brownian motion (GBM) addresses these issues by modeling the **logarithmic returns** of the stock price as a Brownian motion. This ensures that:
1. Stock prices remain positive ($ S_t > 0 $).
2. The percentage changes in stock prices are normally distributed, reflecting the observed behavior of financial markets.
3. The exponential growth of stock prices is captured naturally.

The dynamics of GBM are given by:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$
Where:
- $ S_t $: Stock price at time $ t $.
- $ \mu $: Drift term (expected return of the stock).
- $ \sigma $: Volatility of the stock.
- $ W_t $: Standard Brownian motion.

#### Derivation of Geometric Brownian Motion
To derive GBM, we start with the assumption that the logarithmic returns of the stock price follow a Brownian motion. Let $ X_t = \ln(S_t) $, the natural logarithm of the stock price. The dynamics of $ X_t $ are modeled as:
$$
dX_t = \\nu dt + \sigma dW_t
$$
Where:
- $ \\nu $: Drift term for the logarithmic returns.
- $ \sigma $: Volatility of the logarithmic returns.

Using Ito's Lemma, we can relate $ X_t $ back to $ S_t $. Recall that $ X_t = \ln(S_t) $, so $ S_t = e^{X_t} $. Applying Ito's Lemma to $ f(X_t) = e^{X_t} $, we get:
$$
dS_t = \\frac{\\partial f}{\\partial X_t} dX_t + \\frac{1}{2} \\frac{\\partial^2 f}{\\partial X_t^2} (dX_t)^2
$$

Compute the derivatives:
- $ \\frac{\\partial f}{\\partial X_t} = e^{X_t} = S_t $
- $ \\frac{\\partial^2 f}{\\partial X_t^2} = e^{X_t} = S_t $

Substitute $ dX_t = \\nu dt + \sigma dW_t $ and $ (dX_t)^2 = \sigma^2 dt $ (from the quadratic variation of Brownian motion):
$$
dS_t = S_t (\\nu dt + \\sigma dW_t) + \\frac{1}{2} S_t (\\sigma^2 dt)
$$

Combine terms:
$$
dS_t = S_t \\left( (\\nu + \\frac{1}{2} \\sigma^2) dt + \\sigma dW_t \\right)
$$

Let $ \\mu = \\nu + \\frac{1}{2} \\sigma^2 $, which represents the expected return of the stock. The final GBM equation becomes:
$$
dS_t = \\mu S_t dt + \\sigma S_t dW_t
$$

#### Key Properties of GBM
1. **Positivity**: Since $ S_t = S_0 e^{(\\mu - \\frac{1}{2} \\sigma^2)t + \\sigma W_t} $, the stock price remains positive for all $ t $.
2. **Lognormal Distribution**: The logarithmic returns $ \\ln(S_t / S_0) $ are normally distributed:
   $$
   \\ln(S_t / S_0) \\sim N\\left((\\mu - \\frac{1}{2} \\sigma^2)t, \\sigma^2 t\\right)
   $$
3. **Exponential Growth**: The drift term $ \mu $ captures the long-term exponential growth of the stock price.

#### Why GBM Fits Financial Markets
GBM aligns with empirical observations of stock prices:
- Stock prices exhibit random fluctuations but remain positive.
- Percentage changes in stock prices (returns) are approximately normally distributed.
- Over time, stock prices tend to grow exponentially due to compounding.

By modeling stock prices as GBM, we ensure that the mathematical framework aligns with real-world financial behavior, making it suitable for pricing derivatives like options.
""")

class StochasticProcesses:
    def __init__(self):
        pass

    def brownian_motion(self, S0, mu, sigma, T, dt, simulations):
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] + mu * dt + sigma * np.sqrt(dt) * Z

        return times, paths

    def geometric_brownian_motion(self, S0, mu, sigma, T, dt, simulations):
        time_steps = int(T / dt)
        times = np.linspace(0, T, time_steps)
        paths = np.zeros((simulations, time_steps))
        paths[:, 0] = S0

        for t in range(1, time_steps):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return times, paths


def plot_paths(paths, times=None, N_show=10):
    if times is None:
        times = np.arange(paths.shape[1])
    
    simulations = paths.shape[0]
    if N_show > simulations:
        N_show = simulations

    fig = make_subplots(
    rows=1, cols=2, 
    shared_yaxes=True, 
    horizontal_spacing=0.02,
    column_widths=[0.8, 0.2]
    )
    for i in range(min(simulations, N_show)):  # Plot only the first 10 paths for clarity
        fig.add_trace(go.Scatter(
        x=times, 
        y=paths[i], 
        mode='lines', 
        name=f"Path {i+1}",
        line=dict(width=0.7),
        showlegend=False,
        ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        y=paths[:, -1],
        marker=dict(color='gray'),
        showlegend=False,
        orientation='h',
    ), row=1, col=2)

    fig.update_layout(
        # title=f'N = {simulations} paths',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis2_title='Count',
        template='seaborn',
    )

    return fig

# add short description on the graphs
st.write("### Stock Price Paths: Geometric Brownian Motion vs Brownian Motion")
st.markdown("""
The plots below show simulated stock price paths using **Geometric Brownian Motion (GBM)** and **Brownian Motion**.
You can see that for large number of paths, GBP results are log-normal distributed, while BM results are normally distributed.
""")

# Create an instance of the StochasticProcesses class
sp = StochasticProcesses()
paths_st = st.slider("Number of Stock Price Paths", min_value=100, max_value=10_000, value=1000, step=10)
paths_GBM = sp.geometric_brownian_motion(S0=S_max, mu=r, sigma=sigma, T=T, dt=1/252, simulations=paths_st)
paths_BM = sp.brownian_motion(S0=S_max, mu=r, sigma=sigma, T=T, dt=1/252, simulations=paths_st)

# Plot the stock price paths
st.write("#### Geometric Brownian Motion")
fig_GBM = plot_paths(paths_GBM[1], times=paths_GBM[0], N_show=50)
st.plotly_chart(fig_GBM)

st.write("#### Brownian Motion")
fig_BM = plot_paths(paths_BM[1], times=paths_BM[0], N_show=50)
st.plotly_chart(fig_BM)


# Step 3: Risk-Neutral Valuation
st.header("Step 3: Risk-Neutral Valuation")
st.markdown("""
Risk-neutral valuation is a fundamental concept in financial mathematics. It allows us to simplify the pricing of derivatives by assuming that all investors are indifferent to risk. Here's a detailed breakdown:

#### What is Risk-Neutral Valuation?
In real-world markets, investors demand a risk premium for holding risky assets like stocks. This means the expected return ($ \mu $) of a stock is typically higher than the risk-free rate ($ r $).

However, under the **risk-neutral measure**, we assume that all investors are indifferent to risk. This means:
- The expected return of the stock is equal to the risk-free rate ($ r $).
- Investors do not require a risk premium.

Mathematically, this adjustment changes the stock price dynamics from:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$
to:
$$
dS_t = r S_t dt + \sigma S_t dW_t
$$
Where $ r $ is the risk-free rate.

#### Why Use Risk-Neutral Valuation?
1. **Eliminates Arbitrage**: By assuming risk-neutrality, we ensure that there are no arbitrage opportunities in the market. This is a key requirement for any pricing model.
2. **Simplifies Pricing**: Instead of modeling investor preferences and risk premiums, we can directly use the risk-free rate to calculate the expected payoff of a derivative.
3. **Focuses on Probabilities**: Under the risk-neutral measure, the probabilities of future outcomes are adjusted so that the expected return equals the risk-free rate. This allows us to price derivatives without needing to know the true probabilities of stock price movements.

#### How Does It Work in Practice?
To price a derivative (e.g., an option), we:
1. Assume the stock price follows the risk-neutral dynamics:
   $$
   dS_t = r S_t dt + \sigma S_t dW_t
   $$
2. Calculate the expected payoff of the derivative under the risk-neutral measure.
3. Discount the expected payoff back to the present using the risk-free rate.

For example, the price of a European call option is given by:
$$
C(S, t) = e^{-r(T-t)} \mathbb{E}^Q[\max(S_T - K, 0)]
$$
Where:
- $ \mathbb{E}^Q $: Expectation under the risk-neutral measure.
- $ S_T $: Stock price at maturity.
- $ K $: Strike price.

By using risk-neutral valuation, we avoid the need to estimate the true drift ($ \mu $) of the stock, which is difficult to determine in practice.
""")

# Step 4: Option Pricing via Partial Differential Equation (PDE)
st.header("Step 4: Deriving the Black-Scholes PDE")
st.markdown("""
To derive the Black-Scholes PDE, we use a combination of Ito's Lemma, a hedging argument, and the principle of no-arbitrage. Let's break this down step-by-step:

#### Step 4.1: The Option Price as a Function of Stock Price and Time
Let $ V(S, t) $ represent the price of the option as a function of the stock price $ S $ and time $ t $. 
The stock price $ S_t $ follows the risk-neutral dynamics:
$$
dS_t = r S_t dt + \sigma S_t dW_t
$$
Where:
- $ r $: Risk-free rate
- $ \sigma $: Volatility of the stock
- $ W_t $: Wiener process (standard Brownian motion)

Using **Ito's Lemma**, we can derive the dynamics of $ V(S, t) $. Ito's Lemma states that for a function $ V(S, t) $ of two variables ($ S $ and $ t $), the differential $ dV $ is given by:
$$
dV = \\frac{\\partial V}{\\partial t} dt + \\frac{\\partial V}{\\partial S} dS + \\frac{1}{2} \\frac{\\partial^2 V}{\\partial S^2} (dS)^2
$$

Substituting $ dS_t = r S_t dt + \sigma S_t dW_t $ into the equation, and noting that $ (dS)^2 = \sigma^2 S^2 dt $ (from the properties of Brownian motion), we get:
$$
dV = \\left( \\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} \\right) dt + \\sigma S \\frac{\\partial V}{\\partial S} dW_t
$$

#### Step 4.2: Constructing a Risk-Free Portfolio
To eliminate the stochastic term ($ dW_t $) and create a risk-free portfolio, we construct a portfolio $ \\Pi $ consisting of:
- One option $ V(S, t) $
- $ -\\Delta $ shares of the stock

The value of the portfolio is:
$$
\\Pi = V - \\Delta S
$$

The change in the portfolio value over time is:
$$
d\\Pi = dV - \\Delta dS
$$

Substitute $ dV $ and $ dS $ into this equation:
$$
d\\Pi = \\left( \\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} \\right) dt + \\sigma S \\frac{\\partial V}{\\partial S} dW_t - \\Delta \\left( rS dt + \\sigma S dW_t \\right)
$$

Group terms involving $ dt $ and $ dW_t $:
$$
d\\Pi = \\left( \\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - \\Delta rS \\right) dt + \\left( \\sigma S \\frac{\\partial V}{\\partial S} - \\Delta \\sigma S \\right) dW_t
$$

To eliminate the stochastic term ($ dW_t $), we choose $ \\Delta = \\frac{\\partial V}{\\partial S} $. This is called the **delta-hedging** strategy. Substituting $ \\Delta = \\frac{\\partial V}{\\partial S} $, the equation simplifies to:
$$
d\\Pi = \\left( \\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rS \\frac{\\partial V}{\\partial S} \\right) dt
$$

#### Step 4.3: No-Arbitrage Condition
Since the portfolio $ \\Pi $ is now risk-free (no $ dW_t $ term), its return must equal the risk-free rate $ r $. Therefore:
$$
d\\Pi = r \\Pi dt
$$

Substitute $ \\Pi = V - \\Delta S $ and $ \\Delta = \\frac{\\partial V}{\\partial S} $:
$$
d\\Pi = r \\left( V - \\frac{\\partial V}{\\partial S} S \\right) dt
$$

Equating the two expressions for $ d\\Pi $:
$$
\\left( \\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rS \\frac{\\partial V}{\\partial S} \\right) dt = r \\left( V - \\frac{\\partial V}{\\partial S} S \\right) dt
$$

Cancel $ dt $ and simplify:
$$
\\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rS \\frac{\\partial V}{\\partial S} = rV - rS \\frac{\\partial V}{\\partial S}
$$

Combine like terms:
$$
\\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rV = 0
$$

This is the **Black-Scholes Partial Differential Equation (PDE)**:
$$
\\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rV = 0
$$

#### Why is the PDE Important?
The Black-Scholes PDE governs the price of any derivative whose payoff depends on the stock price $ S $ and time $ t $. By solving this PDE with appropriate boundary conditions, we can find the price of European options, such as calls and puts.
""")

# Step 5: Solving the Black-Scholes PDE
st.header("Step 5: Solving the Black-Scholes PDE Using the Feynman-Kac Theorem")
st.markdown("""
The Black-Scholes PDE governs the price of a European option. To solve it, we use the **Feynman-Kac theorem**, which connects partial differential equations (PDEs) to stochastic processes. Here's a detailed breakdown of the derivation:

#### Step 5.1: The Black-Scholes PDE
The Black-Scholes PDE is:
$$
\\frac{\\partial V}{\\partial t} + rS \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} - rV = 0
$$
With the boundary condition for a European call option:
$$
V(S, T) = \\max(S - K, 0)
$$
Where:
- $ V(S, t) $: Option price as a function of stock price $ S $ and time $ t $.
- $ r $: Risk-free rate.
- $ \sigma $: Volatility of the stock.
- $ K $: Strike price.
- $ T $: Time to maturity.

#### Step 5.2: Feynman-Kac Theorem
The Feynman-Kac theorem states that the solution to a PDE of the form:
$$
\\frac{\\partial V}{\\partial t} + \mu(S, t) \\frac{\\partial V}{\\partial S} + \\frac{1}{2} \sigma^2(S, t) \\frac{\\partial^2 V}{\\partial S^2} - rV = 0
$$
Can be expressed as an expectation under a stochastic process. Specifically, if $ S_t $ follows the dynamics:
$$
dS_t = \mu(S, t) dt + \sigma(S, t) dW_t
$$
Then the solution to the PDE is:
$$
V(S, t) = e^{-r(T-t)} \\mathbb{E}^Q[ \\text{Payoff}(S_T) | S_t = S ]
$$
Where:
- $ \\mathbb{E}^Q $: Expectation under the risk-neutral measure.
- $ S_T $: Stock price at maturity $ T $.
- $ \\text{Payoff}(S_T) $: Payoff of the option at maturity.

For a European call option, the payoff is:
$$
\\text{Payoff}(S_T) = \max(S_T - K, 0)
$$

#### Step 5.3: Risk-Neutral Dynamics of the Stock Price
Under the risk-neutral measure, the stock price $ S_t $ follows:
$$
dS_t = r S_t dt + \sigma S_t dW_t
$$
This implies that $ S_T $, the stock price at maturity, is lognormally distributed:
$$
S_T = S_t \\exp\\left(\\left(r - \\frac{1}{2} \\sigma^2\\right)(T-t) + \\sigma \\sqrt{T-t} Z\\right)
$$
Where $ Z \sim N(0, 1) $ is a standard normal random variable.

#### Step 5.4: Expressing the Option Price as an Expectation
Using the Feynman-Kac theorem, the price of the European call option is:
$$
C(S, t) = e^{-r(T-t)} \\mathbb{E}^Q[\max(S_T - K, 0)]
$$
Substitute the distribution of $ S_T $ into the expectation:
$$
C(S, t) = e^{-r(T-t)} \\int_{K}^{\\infty} (S_T - K) f(S_T) dS_T
$$
Where $ f(S_T) $ is the probability density function of $ S_T $. Since $ S_T $ is lognormal, its logarithm is normally distributed:
$$
\\ln(S_T) \\sim N\\left(\\ln(S_t) + (r - \\frac{1}{2} \\sigma^2)(T-t), \\sigma^2 (T-t)\\right)
$$

#### Step 5.5: Simplifying the Expectation
To simplify the expectation, we introduce two key variables:
$$
d_1 = \\frac{\ln(S/K) + (r + \\frac{1}{2} \\sigma^2)(T-t)}{\\sigma \\sqrt{T-t}}
$$
$$
d_2 = d_1 - \\sigma \\sqrt{T-t}
$$

The cumulative distribution function $ N(x) $ of the standard normal distribution is used to express the probabilities:
- $ N(d_1) $: Probability that the option ends up in the money, weighted by the stock price.
- $ N(d_2) $: Probability that the option ends up in the money, weighted by the discounted strike price.

The final formula for the European call option price is:
$$
C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)
$$
            
Similarly, it can be shown that the price of a European put option is given by:
$$
P(S, t) = K e^{-r(T-t)} N(-d_2) - S N(-d_1)
$$


#### Step 5.6: Why This Works
The Feynman-Kac theorem allows us to transform the PDE into an expectation problem, leveraging the probabilistic nature of the stock price dynamics. By solving the expectation under the risk-neutral measure, we obtain a closed-form solution for the option price.

#### Summary
The Black-Scholes formula is derived by:
1. Writing the PDE for the option price.
2. Applying the Feynman-Kac theorem to express the solution as an expectation.
3. Using the risk-neutral dynamics of the stock price to compute the expectation.
4. Simplifying the result to obtain the closed-form formula.

This elegant approach combines stochastic calculus, PDE theory, and financial intuition to provide a powerful tool for pricing options.
""")

# Step 6: Interactive Inputs

st.header("Step 6: Interact with the Black-Scholes Formula")
st.markdown("""
Adjust the parameters below to see how they affect the price of a European call option. The plot below shows the evolution of the option price over time.
""")

# Black-Scholes formula implementation
def black_scholes(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price
    

# Generate data for the plot
call_price, put_price = black_scholes(S, K, r, sigma, T)

# Create a Plotly figure
st.write("### Call Option Price Surface Plot")
hover_text = np.array([f"Stock Price: {s:.1f}<br>Strike Price: {k:.1f}<br>Option Price: {c:.1f}" for s, k, c in zip(S.flatten(), K.flatten(), call_price.flatten())]).reshape(S.shape)
fig_call = go.Figure()
fig_call.add_trace(go.Surface(
    x=S, y=K, z=call_price, 
    colorscale="Viridis",
    contours={  # Add contours for x and y axes
        "x": {"show": True, "color": "black", "highlight": False},
        "y": {"show": True, "color": "black", "highlight": False},
    },
    hoverinfo="text",
    text=hover_text,
))
fig_call.update_layout(
    scene=dict(
        xaxis_title="Stock Price (S)", 
        yaxis_title="Strike Price (K)", 
        zaxis_title="Option Price (C)",
        ),
    margin=dict(l=0, r=0, b=0, t=0),
)
st.plotly_chart(fig_call)

st.write("### Put Option Price Surface Plot")
hover_text = np.array([f"Stock Price: {s:.1f}<br>Strike Price: {k:.1f}<br>Option Price: {p:.1f}" for s, k, p in zip(S.flatten(), K.flatten(), put_price.flatten())]).reshape(S.shape)
fig_put = go.Figure()
fig_put.add_trace(go.Surface(
    x=S, y=K, z=put_price, 
    colorscale="Viridis",
    contours={  # Add contours for x and y axes
        "x": {"show": True, "color": "black", "highlight": False},
        "y": {"show": True, "color": "black", "highlight": False},
    },
    hoverinfo="text",
    text=hover_text,
))
fig_put.update_layout(
    scene=dict(
        xaxis_title="Stock Price (S)", 
        yaxis_title="Strike Price (K)", 
        zaxis_title="Option Price (P)"),
    margin=dict(l=0, r=0, b=0, t=0),
)
st.plotly_chart(fig_put)


# Step 7: Importance of the Black-Scholes Formula
st.header("Step 7: Importance of the Black-Scholes Formula")
st.markdown("""
The Black-Scholes formula revolutionized finance by providing a closed-form solution for pricing European options. 
Key implications include:
1. **Risk Management**: It allows traders to hedge their positions dynamically.
2. **Market Efficiency**: It assumes markets are efficient, which aligns with modern financial theory.
3. **Foundation for Extensions**: The model serves as a basis for more complex models (e.g., stochastic volatility models).

However, it has limitations:
- Assumes constant volatility and interest rates.
- Does not account for transaction costs or dividends.
""")

# Display the final formula
st.subheader("Final Black-Scholes Formula")
st.latex(r"""
C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)
""")
st.latex(r"""
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma \sqrt{T-t}}, \quad d_2 = d_1 - \sigma \sqrt{T-t}
""")

# Step 8: Monte Carlo Simulation for Option Pricing
st.header("Step 8: Monte Carlo Simulation for Option Pricing")
st.markdown("""
Monte Carlo simulation is a flexible method for pricing options by simulating multiple potential stock price paths. Here, we use Geometric Brownian Motion (GBM) to simulate stock price paths and calculate the option price. The result is then compared with the Black-Scholes price.

#### How It Works:
1. Simulate $ N $ stock price paths using GBM.
2. Calculate the payoff for each path at maturity.
3. Average the discounted payoffs to estimate the option price.
4. Compare the Monte Carlo result with the Black-Scholes price.

Adjust the number of simulated paths below to see how it affects the accuracy of the Monte Carlo estimate.
""")

# Sliders for user input
S, K = S_max, K_max

# Black-Scholes formula implementation
def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Monte Carlo simulation for European call option
def monte_carlo_call(S, K, r, sigma, T, num_paths):
    np.random.seed(42)  # For reproducibility
    dt = T  # Single-step simulation for simplicity
    Z = np.random.normal(size=num_paths)  # Standard normal random variables
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)  # Stock price at maturity
    payoffs = np.maximum(ST - K, 0)  # Payoff of the call option
    discounted_payoffs = np.exp(-r * T) * payoffs  # Discounted payoffs
    mc_price = np.mean(discounted_payoffs)  # Average to estimate option price
    return mc_price

# Calculate prices
num_paths = st.slider("Number of Simulated Paths", min_value=1_000, max_value=50_000, value=5_000, step=1_000)
bs_price = black_scholes_call(S, K, r, sigma, T)
mc_price = monte_carlo_call(S, K, r, sigma, T, num_paths)

# Display results
st.subheader("Option Prices")
st.markdown(f"""
- **Black-Scholes Call Price**: ${bs_price:.2f}
- **Monte Carlo Call Price**: ${mc_price:.2f}
- **Difference**: ${abs(bs_price - mc_price):.2f}
""")

# Plot convergence of Monte Carlo price as number of paths increases
paths_range = np.linspace(100, num_paths, 300, dtype=int)
mc_prices = [monte_carlo_call(S, K, r, sigma, T, int(paths)) for paths in paths_range]

fig = go.Figure()
fig.add_trace(go.Scatter(x=paths_range, y=mc_prices, mode='lines', name='Monte Carlo Price'))
fig.add_hline(y=bs_price, line_dash="dash", line_color="red", annotation_text="Black-Scholes Price")
fig.update_layout(
    title="Convergence of Monte Carlo Call Price to Black-Scholes Call Price",
    xaxis_title="Number of Simulated Paths",
    yaxis_title="Option Call Price",
    template="plotly_white"
)
st.plotly_chart(fig)


# Step 9: Implied Volatility Calculator
st.header("Step 9: Implied Volatility Calculator")
st.markdown("""
The Black-Scholes model provides a theoretical price for options, but in practice, market prices often deviate from these theoretical values. This discrepancy arises because the Black-Scholes model assumes constant volatility, while real-world markets exhibit varying levels of volatility depending on factors like strike price and time to maturity.

#### Why Black-Scholes Prices Differ from Market Prices:
1. **Constant Volatility Assumption**:
   - The Black-Scholes model assumes that volatility is constant over the life of the option. In reality, volatility changes over time and varies across strike prices (volatility smile/skew).
2. **Market Sentiment**:
   - Market participants incorporate expectations of future events (e.g., earnings announcements, economic data) into option prices, which may not align with historical volatility.
3. **Supply and Demand**:
   - Option prices are influenced by trading activity, liquidity, and demand for specific strikes or maturities.

#### What is Implied Volatility?
Implied volatility (IV) is the volatility input to the Black-Scholes model that makes the theoretical price equal to the observed market price of the option. It reflects the market's expectation of future stock price movements. Key points:
- IV is derived by solving the Black-Scholes equation for $ \sigma $ given the market price.
- Higher IV indicates higher expected price swings, while lower IV suggests stability.
- IV is a forward-looking measure, unlike historical volatility, which is based on past price movements.

#### Implied Volatility Calculator:
Use the slider below to input the market price of a call option as a percentage of the Black-Scholes price given a historical volatility you entered on the sidebar.
The calculator will estimate the implied volatility that matches the market price.
""")

# Function to calculate implied volatility using Newton-Raphson method
def implied_volatility(S, K, r, T, market_price, option_type='call', tol=1e-6, max_iter=100):
    sigma = 0.2  # Initial guess for volatility
    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        vega = S * norm.pdf(d1) * np.sqrt(T)  # Derivative of price w.r.t. sigma
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return sigma

# create a simple calculator for implied volatility
market_price = st.slider(f"Market Call Price (as % of Black-Scholes Call Price, given a historical volatility of {sigma:.0%})", min_value=80.0, max_value=120.0, value=100.0, step=1.0) / 100
iv = implied_volatility(S, K, r, T, market_price * bs_price)
st.write(f"### Implied Volatility: {iv:.1%}")
if iv > sigma:
    st.write("""
        The implied volatility is higher than the historical volatility.
        Market anticipates greater price fluctuations in the future than what has been observed historically.
        This could be due to upcoming events such as:
        * Earnings reports.
        * Economic data releases (e.g., inflation reports, interest rate decisions).
        * Geopolitical risks (e.g., elections, trade tensions).
        High IV may indicate fear or uncertainty (e.g., ahead of earnings announcements or macroeconomic events).
    """)
elif iv < sigma:
    st.write("""
        The implied volatility is lower than the historical volatility.
        Market expects lower price swings in the future compared to historical levels.
        This could be due to factors like:
        * Stable economic conditions.
        * Low market uncertainty.
        * Lack of significant upcoming events.
        Low IV may suggest complacency or confidence in stable prices.
    """)