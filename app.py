import streamlit as st
import yfinance as yf

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
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        st.write("Stock Data:")
        st.write(data)
