import streamlit as st
import pandas as pd
import plotly.express as px  # Assuming upgraded Plotly Express

# Alternative (if upgrade not possible):
# import plotly.graph_objects as go  # Import for Plotly.py subplots

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

                # Create seasonal decomposition plot using Plotly Express (upgraded)
                fig_decomp = px.subplots(rows=4, cols=1)
                fig_decomp.add_trace(px.line(result, x=result.index, y='observed', title='Observed' if show_labels else ''), showlegend=show_labels)
                fig_decomp.add_trace(px.line(result, x=result.index, y='trend', title='Trend' if show_labels else ''), row=2, col=1, showlegend=show_labels)
                fig_decomp.add_trace(px.line(result, x=result.index, y='seasonal', title='Seasonal' if show_labels else ''), row=3, col=1, showlegend=show_labels)
                fig_decomp.add_trace(px.line(result, x=result.index, y='resid', title='Residual' if show_labels else ''), row=4, col=1, showlegend=show_labels)
                fig_decomp.update_layout(
                    title=f'Seasonal Decomposition of {selected_category}',
                    xaxis_title='Date'
                )

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
