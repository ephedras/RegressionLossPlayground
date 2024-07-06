'''
This app is a playground for checking different loss function behaviour used for regression task.
v.1.0.3
'''


import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Define Huber Loss function
def huber_loss(pred, true, delta=1.0):
  return np.where(np.abs(pred - true) < delta, 0.5 * (pred - true)**2, delta * (np.abs(pred - true) - 0.5 * delta))

# Define Quantile Loss function
def quantile_loss(pred, true, q=0.5):
  return np.maximum(q * (true - pred), (q - 1) * (true - pred))

# Define Mean Squared Error (MSE)
def mse_loss(pred, true):
  return (pred - true) ** 2

# Define Mean Absolute Error (MAE)
def mae_loss(pred, true):
  return np.abs(pred - true)

# Define Huberized Q-Loss function (combination of Huber and Quantile Loss)
def huberized_q_loss(pred, true, delta=1.0, q=0.5):
  hloss = huber_loss(pred, true, delta)
  return np.where(true > pred, q * hloss, (1 - q) * hloss)


# Define Log Loss function (for binary classification)
def log_loss(pred, true):
  epsilon = 1e-15  # Add a small value to avoid log(0) errors
  return -(true * np.log(pred + epsilon) + (1 - true) * np.log(1 - pred + epsilon))

# Define Kullback-Leibler Divergence (KL Divergence)
def kl_divergence(pred, true):
  epsilon = 1e-15  # Add a small value to avoid log(0) errors
  return true * np.log(true / (pred + epsilon)) + (1 - true) * np.log((1 - true) / (1 - pred + epsilon))


st.set_page_config(page_title="Loss Function Playground",page_icon='.\logo\icon_clear.png')

#logo
st.sidebar.image('icon_clear.png')
st.sidebar.image("logo_clear.png")


# Streamlit app layout
st.title("Loss Function Selector")

# Sidebar dropdown for selecting loss function
selected_loss_functions = st.sidebar.multiselect("Select Loss Functions", 
                                                 ["Huber Loss", "Quantile Loss", "Huberized Q-Loss", "Mean Squared Error", "Mean Absolute Error"])

# Sidebar sliders (common for all functions)
true_value = st.sidebar.slider("Select true value", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# Function-specific sliders (shown conditionally)
delta_col, q_col, hq_delta_col, hq_q_col , whl_delta_col, whl_alpha_col, whl_beta_col= st.sidebar,st.sidebar,st.sidebar,st.sidebar,st.sidebar,st.sidebar,st.sidebar
delta = delta_col.slider("Huber Loss - delta", min_value=0.1, max_value=2.0, value=1.0, step=0.1) if "Huber Loss" in selected_loss_functions else None
q = q_col.slider("Quantile Loss - q", min_value=0.1, max_value=0.9, value=0.5, step=0.1) if "Quantile Loss" in selected_loss_functions else None
hq_delta = hq_delta_col.slider("Huberized Q - delta", min_value=0.1, max_value=2.0, value=1.0, step=0.1) if "Huberized Q-Loss" in selected_loss_functions else None
hq_q = hq_q_col.slider("Huberized Q - q", min_value=0.1, max_value=0.9, value=0.5, step=0.1) if "Huberized Q-Loss" in selected_loss_functions else None

# Generate predicted values
pred_values = np.linspace(-2, 2, 400)

# Initialize empty lists to store loss values
loss_values = []
loss_function_names = []

# Calculate loss values for selected functions

for function_name in selected_loss_functions:
  if function_name == "Huber Loss":
    loss = huber_loss(pred_values, true_value, delta)
  elif function_name == "Quantile Loss":
    loss = quantile_loss(pred_values, true_value, q)
  elif function_name == "Huberized Q-Loss":
    loss = huberized_q_loss(pred_values, true_value, hq_delta, hq_q)
  elif function_name == "Mean Squared Error":
    loss = mse_loss(pred_values, true_value)
  elif function_name == "Mean Absolute Error":
    loss = mae_loss(pred_values, true_value)
  else:
    continue  # Skip unsupported functions
  loss_values.append(loss)
  loss_function_names.append(function_name)


# Create Plotly figure
fig = go.Figure()

# Add traces for each loss function
for i in range(len(loss_values)):
  fig.add_trace(go.Scatter(x=pred_values, y=loss_values[i], mode='lines', name=loss_function_names[i]))

fig.update_layout(
  title="Loss Functions",
  xaxis_title="Predicted Value",
  yaxis_title="Loss",
  hovermode="x unified",
)

# Display Plotly chart
st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == "__main__":
  pass
