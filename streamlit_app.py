import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Error handling for file path
try:
  file_path = 'category_sales_dat.csv'  # Update the file path if needed
  data = pd.read_csv(file_path)
except FileNotFoundError:
  st.error(f"Error: File '{file_path}' not found. Please check the file path and try again.")

# Set the 'product_type' as the index with error handling
try:
  data.set_index('product_type', inplace=True)
except KeyError:
  st.error(f"Error: Column 'product_type' not found in the data. Please check your CSV file.")

# Transpose the data
data = data.T

# Remove the 'Grand Total' row
data = data.drop(['Grand Total'], axis=0)

# Convert the index to datetime with error handling
try:
  data.index = pd.to_datetime(data.index, format='%Y-%b')
except ValueError:
  st.error(f"Error: Invalid date format. Please ensure the date format in your CSV matches '%Y-%b'.")

# Check the number of observations for each category
observations_count = data.count()

# Separate categories based on length of data
categories_with_sufficient_data = observations_count[observations_count >= 24].index
categories_with_insufficient_data = observations_count[observations_count < 24].index

# Streamlit App
st.title('Seasonality Analysis of Product Categories')

# Category selection
selected_category = st.selectbox(
    'Select a Product Category',
    data.columns
)

# Display the selected category data
st.write(f"Sales data for {selected_category}:")

# Display original sales data
fig, ax = plt.subplots()
ax.plot(data.index, data[selected_category], label='Original Sales')
ax.set_title(f'Sales Data of {selected_category}')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
st.pyplot(fig)

# Perform analysis based on data sufficiency
if selected_category in categories_with_sufficient_data:
    # Seasonal decomposition
    try:
        result = seasonal_decompose(data[selected_category].dropna(), model='multiplicative', period=12)

        # Plot decomposition
        st.write(f"Seasonal Decomposition of {selected_category}:")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        result.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        fig.suptitle(f'Seasonal Decomposition of {selected_category}', y=0.92)
        st.pyplot(fig)
    except:
        st.error(f"Error: An error occurred during seasonal decomposition. Please check your data.")

else:
    # Calculate moving average
    window_size = 3  # 3-month moving average
    data[selected_category] = data[selected_category].astype(float)  # Ensure data is in float format
    moving_avg = data[selected_category].rolling(window=window_size).mean()

    # Plot moving average
    st.write(f"Moving Average of {selected_category}:")
    fig, ax = plt.subplots()
    ax.plot(data.index, data[selected_category], label='Original Sales')
    ax.plot(data.index, moving_avg, label='Moving Average', color='red')
    ax.set_title(f'Moving Average of {selected_category}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    st.pyplot(fig)

# Run the Streamlit
