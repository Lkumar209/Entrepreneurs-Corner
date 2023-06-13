import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Define the different pages
pages = {
    "Startup Matchmaker": "matchmaker",
    "Entrepreneurship Mentor Chatbot": "chatbot",
    "Business Idea Generator": "idea_generator",
    "Stock Market Predictions Tool": "stock_predictions"
}

# Add a title for the navigation bar
st.title("Entrepreneurship Corner")

# Create a horizontal option menu for navigation
selected_page = st.sidebar.selectbox("Select a feature:", list(pages.keys()))

# Render the selected page
if selected_page == "Startup Matchmaker":
    # Render the Startup Matchmaker page
    st.title("Startup Matchmaker")
    # Add your code for the Startup Matchmaker page here

elif selected_page == "Entrepreneurship Mentor Chatbot":
    # Render the Entrepreneurship Mentor Chatbot page
    st.title("Entrepreneurship Mentor Chatbot")
    # Add your code for the Entrepreneurship Mentor Chatbot page here

elif selected_page == "Business Idea Generator":
    # Render the Business Idea Generator page
    st.title("Business Idea Generator")
    # Add your code for the Business Idea Generator page here

elif selected_page == "Stock Market Predictions Tool":
    # Render the Stock Market Predictions Tool page
    st.title("Stock Market Predictions Tool")
    symbol = st.text_input("Enter a stock symbol (e.g., AAPL):")
    if symbol:
        # Retrieve historical stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        
        # Prepare the data for prediction
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].map(lambda x: x.toordinal())
        
        # Split the data into training and testing sets
        X = data[['Date']].values
        y = data['Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert ordinal dates to datetime format
        plot_data = data.copy()
        plot_data['Date'] = plot_data['Date'].map(lambda x: datetime.fromordinal(int(x)))
        
        # Create a DataFrame for predicted values
        predictions = pd.DataFrame({'Date': plot_data['Date'].values[X_train.shape[0]:], 'Close': y_pred})
        
        # Create a combined DataFrame for plotting
        plot_data = pd.concat([plot_data, predictions], ignore_index=True)
        
        # Generate the graph
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data['Date'], plot_data['Close'], color='blue', label='Actual')
        plt.plot(plot_data['Date'], plot_data['Close'].shift(-X_train.shape[0]), color='orange', linestyle='--', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Stock Performance and Predictions')
        plt.legend()
        plt.grid(True)
        
        # Display the graph
        st.pyplot(plt)
        
        # Display the results
        st.write("Stock Data:")
        st.write(data)
        st.write("Predicted Close Prices:")
        st.write(predictions)
        st.write("Mean Squared Error:", mse)
        st.write("R^2 Score:", r2)
