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
import sqlite3
from streamlit_tags import st_tags
from datetime import datetime

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

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


if selected_page == "BizMatch":
    conn = sqlite3.connect('investors.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS investors
                (name TEXT, description TEXT, funding REAL, industry TEXT, contact TEXT)''')
    conn.commit()

    # Page 1: Investor Profile
    def investor_profile():
        st.title("Investor Profile")
        name = st.text_input("Name")
        description = st.text_area("Description")
        funding = st.number_input("Funding Amount", min_value=0.0)
        interests = st.text_input("Interested Industries (comma-separated)")
        contact = st.text_input("Contact Information")
        if st.button("Submit"):
            interests_list = [interest.strip().lower()
                              for interest in interests.split(",")]
            for interest in interests_list:
                c.execute("INSERT INTO investors VALUES (?, ?, ?, ?, ?)",
                          (name, description, funding, interest, contact))
            conn.commit()
            st.success("Profile submitted successfully!")

    # Page 2: Startup Founder
    def startup_founder():
        st.title("Startup Founder")
        industries = set()
        # Fetch all unique industries from the database
        for row in c.execute("SELECT DISTINCT industry FROM investors"):
            industries.add(row[0])
        selected_industry = st.selectbox(
            "Select an Industry", list(industries))
        if st.button("Find Investors"):
            # Fetch investors with similar interests from the database (case-insensitive)
            c.execute(
                "SELECT * FROM investors WHERE LOWER(industry) = LOWER(?)", (selected_industry,))
            results = c.fetchall()
            st.subheader("Matching Investors:")
            for result in results:
                st.write("Name:", result[0])
                st.write("Description:", result[1])
                st.write("Funding:", result[2])
                st.write("Contact:", result[4])
                st.write("---")

    # Streamlit App
    def main():
        st.header("Investor-Founder Matching App")

        page = option_menu(
            menu_title=None,
            options=["Investor Profile", "Startup Founder"],
            icons=["map", "person-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            key="nav_bar"
        )
        st.markdown("<br>", unsafe_allow_html=True)  # Add a line break

        if page == "Investor Profile":
            investor_profile()
        elif page == "Startup Founder":
            startup_founder()

    if __name__ == '__main__':
        main()

    # Render the Startup Matchmaker page
#     st.header("Get Matched With Your Ideal Startup Team")

#     # Collect Personal Information
#     name = st.text_input("Name")
#     age = st.number_input("Age", min_value=0, max_value=100, value=0, step=1)
#     location = st.text_input("Location")
#     contact_information = st.text_area("Contact Information")

#     # Collect Skills and Expertise

#     skills = st_tags(
#     label='# Enter Skills:',
#     text='Press enter to add more',
#     suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
#     maxtags=10,
#     key='1'
# )

# # Convert the list of skills into a comma-separated string
#     expertise = st_tags(
#     label='# Enter areas of expertise:',
#     text='Press enter to add more',
#     suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
#     maxtags=10,
#     key='2'
# )

#     qualifications = st_tags(
#     label='# Enter Qualifications:',
#     text='Press enter to add more',
#     suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
#     maxtags=10,
#     key='3'
# )

#     # Collect Interests and Goals

#     # Collect Industry Experience
#     previous_work_experience = st.radio(
#     "Have you worked before?",
#     ('Yes', 'No'))

#     if previous_work_experience == 'Yes':
#         workYears = st.number_input('Years Worked', min_value=0, max_value=100, value=0, step=1)

#     # Collect Business Idea or Concept
#     business_idea = st.text_area("Business Idea or Concept")

#     # Collect Preferences and Criteria
#     geographic_location = st.text_input("Geographic Location")
#     commitment_level = st.selectbox("Desired Level of Commitment", ["Full-time", "Part-time"])
#     equity_split = st.number_input("Desired Equity Split")

#     # Store the data
#     user_data = {
#         "name": name,
#         "age": age,
#         "location": location,
#         "contact_information": contact_information,
#         "skills": skills,
#         "expertise": expertise,
#         "qualifications": qualifications,
#         "previous_work_experience": previous_work_experience,
#         "workYears": workYears,
#         "business_idea": business_idea,
#         "geographic_location": geographic_location,
#         "commitment_level": commitment_level,
#         "equity_split": equity_split,
#     }

    # You can then store this user_data dictionary in a database or use it for further processing


elif selected_page == "BizBot":
    openai.api_key = "insert the key here !!!"

    def generate_response(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = response.choices[0].message.content

        return generated_text

    st.header("BizBot - Your AI Business Assistant")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input(
            "You: ", "How can I get in touch with startup incubators?", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Create a DataFrame for predicted values
        predictions = pd.DataFrame(
            {'Date': data['Date'].values[X_train.shape[0]:], 'Close': y_pred})

        # Create a combined DataFrame for plotting
        plot_data = pd.concat([data, predictions], ignore_index=True)

        # Generate the graph
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data['Date'], plot_data['Close'],
                 color='blue', label='Actual')
        plt.plot(plot_data['Date'], plot_data['Close'].shift(-X_train.shape[0]),
                 color='orange', linestyle='--', label='Predicted')
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
