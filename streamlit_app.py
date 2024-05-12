import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to load and process data (assuming CSV file is uploaded)
def load_data(file_uploader):
    try:
        data = pd.read_csv(file_uploader)
        data.set_index('product_type', inplace=True)
        data = data.T
        data.drop('Grand Total', axis=0, inplace=True)
        data.index = pd.to_datetime(data.index, format='%Y-%b')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Streamlit app layout
st.title('Seasonality Analysis')

# Upload data section
uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")

# Load data if a file is uploaded
if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # Available categories for dropdown
        category_options = data.columns.tolist()

        # Select category using dropdown
        selected_category = st.selectbox('Select Product Category:', category_options)

        # Show/Hide labels toggle checkbox
        show_labels = st.checkbox('Show Component Labels', value=True)

        # Perform seasonal decomposition and display plot
        if selected_category in data.columns:
            try:
                result = seasonal_decompose(data[selected_category].dropna(), model='multiplicative', period=12)

                # Create seasonal decomposition plot using Plotly
                fig_decomp = make_subplots(rows=4, cols=1, subplot_titles=[
                    'Observed' if show_labels else '',
                    'Trend' if show_labels else '',
                    'Seasonal' if show_labels else '',
                    'Residual' if show_labels else ''
                ])
                
                fig_decomp.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
                fig_decomp.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
                fig_decomp.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                fig_decomp.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)

                fig_decomp.update_layout(height=800, width=800, title_text=f'Seasonal Decomposition of {selected_category}')
                
                # Display the plot
                st.plotly_chart(fig_decomp)
            except Exception as e:
                st.error(f"Error generating plot for {selected_category}: {e}")
        else:
            st.error(f"Category '{selected_category}' not found.")
    else:
        st.info("Data loading failed. Please try again.")
else:
    st.info("Please upload a CSV file to proceed.")
