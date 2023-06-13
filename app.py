import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_option_menu import option_menu
from streamlit_chat import message
import openai


st.markdown(
    """
    <style>
    .block-container {
        text-align: center;

    }

    .title {
        align-self: flex-start;
     </style>
    """,
    unsafe_allow_html=True
)

logo = "eclogo.png"
st.image(logo, use_column_width=True)

selected_page = option_menu(
    menu_title=None,
    options=["BizMatch", "BizBot", "Idea Oasis", "EnvisionX"],
    icons=["map", "person-circle", "info", "geo"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


# Render the selected page
if selected_page == "BizMatch":
    # Render the Startup Matchmaker page
    st.title("Startup Matchmaker")
    # Add your code for the Startup Matchmaker page here

elif selected_page == "BizBot":
    openai.api_key = "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"

    def generate_response(prompt):
        completions = openai.Completion.create( 
            engine = "text-davinci-003",
            prompt =prompt,
            max_tokens = 1024,
            n = 1,
            stop = None,
            temperature = 0.5,
        )
        message = completions.choices[0].text
        return message
    
    st.header("BizBot - Your AI Business Assistant")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state: 
        st.session_state['past'] = []
    
    def get_text():
        input_text = st.text_input("You: ", "How can I get in touch with startup incubators?", key="input")
        return input_text
    
    user_input = get_text()

    if user_input:
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))
            


elif selected_page == "Idea Oasis":
    # Render the Business Idea Generator page
    st.title("Business Idea Generator")
    # Add your code for the Business Idea Generator page here

elif selected_page == "EnvisionX":
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
        
        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create a DataFrame for predicted values
        predictions = pd.DataFrame({'Date': data['Date'].values[X_train.shape[0]:], 'Close': y_pred})
        
        # Create a combined DataFrame for plotting
        plot_data = pd.concat([data, predictions], ignore_index=True)
        
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
