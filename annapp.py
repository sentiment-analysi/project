import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import io
import mysql.connector
from xlsxwriter import Workbook

nltk.download('stopwords')

# Load the trained model and preprocessing objects
classifier = load_model('trained_model.h5')
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
sc = pickle.load(open('Standard-Scaler.pkl','rb'))
import mysql.connector

# Establish a connection to your MySQL database
conn = mysql.connector.connect(
    host="sql12.freesqldatabase.com",
    user="sql12619244",
    password="NP2lGRPxFL",
    database="sql12619244"
)
c = conn.cursor()


ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'



# Function to perform sentiment analysis
def predict_sentiment(input_review):
    if not input_review:
        return "No review"
    
    input_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_review)
    input_review = input_review.lower()
    input_review_words = input_review.split()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_review_words = [word for word in input_review_words if not word in stop_words]
    ps = PorterStemmer()
    input_review = [ps.stem(word) for word in input_review_words]
    input_review = ' '.join(input_review)
    input_X = cv.transform([input_review]).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    if pred[0][0]:
        return "Positive review"
    else:
        return "Negative review"


def predict_sentiment1(df, column_name):
    # Check if the specified column exists in the dataframe
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")
    
    # Check if the specified column contains string values
    if df[column_name].dtype != 'object':
        warnings.warn(f"Column '{column_name}' is not a string column", UserWarning)
    input_reviews = df[column_name].tolist()
    input_reviews = [review for review in input_reviews if isinstance(review, str) and review.strip()] # Ignore empty reviews
    input_reviews = [re.sub(pattern='[^a-zA-Z]', repl=' ', string=review) for review in input_reviews]
    input_reviews = [review.lower() for review in input_reviews]
    input_reviews = [review.split() for review in input_reviews]
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_reviews = [[word for word in review_words if not word in stop_words] for review_words in input_reviews]
    ps = PorterStemmer()
    input_reviews = [[ps.stem(word) for word in review_words] for review_words in input_reviews]
    input_reviews = [' '.join(review_words) for review_words in input_reviews]
    input_X = cv.transform(input_reviews).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    sentiment = ['Positive review' if p else 'Negative review' for p in pred]
    return sentiment

# Function to show the analytics in a separate tab

def show_analytics(df, column_name):
    # Check if the selected column qualifies for sentiment analysis
    if df[column_name].dtype not in [np.dtype('O'), np.dtype('<U')]:
        st.error('Selected column is not a text column')
        return
    
    # Apply sentiment analysis to specified column
    sentiments = predict_sentiment1(df, column_name)
    
    # Get the count of reviews and positive/negative reviews
    total_reviews = len(sentiments)
    positive_reviews = sentiments.count('Positive review')
    negative_reviews = sentiments.count('Negative review')
    
    # Print the count of reviews and positive/negative reviews
    st.write(f"Total number of reviews: {total_reviews}")
    st.write(f"Number of positive reviews: {positive_reviews}")
    st.write(f"Number of negative reviews: {negative_reviews}")
    
    # Plot the sentiment analysis results using matplotlib
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [positive_reviews, negative_reviews], color=['blue', 'orange'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    
    # Add count labels inside each bar
    for i, count in enumerate([positive_reviews, negative_reviews]):
        ax.text(i, count, str(count), ha='center', va='bottom', color='white')
    
    # Set the ylim to include the count labels
    ax.set_ylim(top=max(positive_reviews, negative_reviews) * 1.1)
    
    st.pyplot(fig)

 
# Define function to delete reviews
def delete_reviews():
    # Get all available USNs
    usns = [rows for row in c.execute("SELECT usn FROM reviews").fetchall()]

    # Show dropdown to select a USN
    selected_usn = st.selectbox('Select a USN:', options=usns)

    # Delete the selected review
    if selected_usn:
        delete_button = st.button('Delete')

        if delete_button:
            c.execute("DELETE FROM reviews WHERE usn=?", (selected_usn,))
            conn.commit()
            c.execute("VACUUM")  # This optimizes the database
            st.success(f'Review for {selected_usn} has been deleted.')
            reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
            st.experimental_rerun()

    else:
        st.warning('Please select a USN to delete.')


    
def show_sentiment_wise_analytics(reviews_df):
    num_pos_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Positive review'])
    num_pos_reviewsfor2 = len(reviews_df[reviews_df['sentiment2'] == 'Positive review'])
    num_pos_reviewsfor3 = len(reviews_df[reviews_df['sentiment3'] == 'Positive review'])
    num_neg_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Negative review'])
    num_neg_reviewsfor2 = len(reviews_df[reviews_df['sentiment2'] == 'Negative review'])
    num_neg_reviewsfor3 = len(reviews_df[reviews_df['sentiment3'] == 'Negative review'])
    totalnum_pos_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Positive review']) + \
                          len(reviews_df[reviews_df['sentiment2'] == 'Positive review']) + \
                          len(reviews_df[reviews_df['sentiment3'] == 'Positive review'])
    totalnum_neg_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Negative review']) + \
                          len(reviews_df[reviews_df['sentiment2'] == 'Negative review']) + \
                          len(reviews_df[reviews_df['sentiment3'] == 'Negative review'])

    st.subheader("Question 1 - Course_experience")
    st.write(f"Positive reviews: {num_pos_reviewsfor1}")
    st.write(f"Negative reviews: {num_neg_reviewsfor1}")
    
    st.subheader("Question 2 - About Instructor")
    st.write(f"Positive reviews: {num_pos_reviewsfor2}")
    st.write(f"Negative reviews: {num_neg_reviewsfor2}")
    
    st.subheader("Question 3 - Material Feedback")
    st.write(f"Positive reviews: {num_pos_reviewsfor3}")
    st.write(f"Negative reviews: {num_neg_reviewsfor3}")    
    
    
    st.subheader("Total Reviews")
    st.write(f"Positive reviews: {totalnum_pos_reviews}")
    st.write(f"Negative reviews: {totalnum_neg_reviews}")
    st.write(f"Total reviews recorded: {totalnum_pos_reviews+totalnum_neg_reviews}")

    # Create a bar graph of the sentiment analysis results
    fig, ax = plt.subplots(figsize=(10,5))
    sentiment_labels = ['Positive', 'Negative']
    question_labels = ['Q1', 'Q2', 'Q3', 'Total']
    pos_counts = [num_pos_reviewsfor1, num_pos_reviewsfor2, num_pos_reviewsfor3, totalnum_pos_reviews]
    neg_counts = [num_neg_reviewsfor1, num_neg_reviewsfor2, num_neg_reviewsfor3, totalnum_neg_reviews]
    x = np.arange(len(question_labels))
    width = 0.35
    ax.bar(x - width/2, pos_counts, width, label='Positive', color='green')
    ax.bar(x + width/2, neg_counts, width, label='Negative', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(question_labels)
    ax.legend()
    ax.set_ylabel('Number of Reviews')
    ax.set_xlabel('Questions')
    ax.set_title('Sentiment Analysis Results')
    st.pyplot(fig)

    


 
# Define function to delete reviews
def delete_reviews():
    # Get all available USNs
    usns = [rows for row in c.execute("SELECT usn FROM reviews").fetchall()]

    # Show dropdown to select a USN
    selected_usn = st.selectbox('Select a USN:', options=usns)

    # Delete the selected review
    if selected_usn:
        delete_button = st.button('Delete')

        if delete_button:
            c.execute("DELETE FROM reviews WHERE usn=?", (selected_usn,))
            conn.commit()
            c.execute("VACUUM")  # This optimizes the database
            st.success(f'Review for {selected_usn} has been deleted.')
            reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
            st.experimental_rerun()

    else:
        st.warning('Please select a USN to delete.')


    
def show_sentiment_wise_analytics(reviews_df):
    num_pos_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Positive review'])
    num_pos_reviewsfor2 = len(reviews_df[reviews_df['sentiment2'] == 'Positive review'])
    num_pos_reviewsfor3 = len(reviews_df[reviews_df['sentiment3'] == 'Positive review'])
    num_neg_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Negative review'])
    num_neg_reviewsfor2 = len(reviews_df[reviews_df['sentiment2'] == 'Negative review'])
    num_neg_reviewsfor3 = len(reviews_df[reviews_df['sentiment3'] == 'Negative review'])
    totalnum_pos_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Positive review']) + \
                          len(reviews_df[reviews_df['sentiment2'] == 'Positive review']) + \
                          len(reviews_df[reviews_df['sentiment3'] == 'Positive review'])
    totalnum_neg_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Negative review']) + \
                          len(reviews_df[reviews_df['sentiment2'] == 'Negative review']) + \
                          len(reviews_df[reviews_df['sentiment3'] == 'Negative review'])

    st.subheader("Question 1 - Course_experience")
    st.write(f"Positive reviews: {num_pos_reviewsfor1}")
    st.write(f"Negative reviews: {num_neg_reviewsfor1}")

    st.subheader("Question 2 - About Instructor")
    st.write(f"Positive reviews: {num_pos_reviewsfor2}")
    st.write(f"Negative reviews: {num_neg_reviewsfor2}")

    st.subheader("Question 3 - Material Feedback")
    st.write(f"Positive reviews: {num_pos_reviewsfor3}")
    st.write(f"Negative reviews: {num_neg_reviewsfor3}")

    st.subheader("Total Reviews")
    st.write(f"Positive reviews: {totalnum_pos_reviews}")
    st.write(f"Negative reviews: {totalnum_neg_reviews}")
    st.write(f"Total reviews recorded: {totalnum_pos_reviews+totalnum_neg_reviews}")

    # Create a bar graph of the sentiment analysis results
    fig, ax = plt.subplots(figsize=(14, 10))
    sentiment_labels = ['Positive', 'Negative']
    question_labels = ['Q1', 'Q2', 'Q3', 'Total']
    pos_counts = [num_pos_reviewsfor1, num_pos_reviewsfor2, num_pos_reviewsfor3, totalnum_pos_reviews]
    neg_counts = [num_neg_reviewsfor1, num_neg_reviewsfor2, num_neg_reviewsfor3, totalnum_neg_reviews]
    x = np.arange(len(question_labels))
    width = 0.35
    ax.bar(x - width / 2, pos_counts, width, label='Positive', color='green')
    ax.bar(x + width / 2, neg_counts, width, label='Negative', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(question_labels)
    ax.legend()
    ax.set_ylabel('Number of Reviews')
    ax.set_xlabel('Questions')
    ax.set_title('Sentiment Analysis Results')

    # Add count labels above each bar
    for i, pos_count in enumerate(pos_counts):
        ax.text(i - width / 2, pos_count + 0.1, str(pos_count), ha='center', color='black')
    for i, neg_count in enumerate(neg_counts):
        ax.text(i + width / 2, neg_count + 0.1, str(neg_count), ha='center', color='black')

    st.pyplot(fig)





# Main function to run the app
def run_sentiment_app():
    st.title('Student Sentiment Analysis')

    # Add a menu with two options
    menu = ['Home', 'Analytics', 'Forms']
    choice = st.sidebar.selectbox('Select page', menu)

    # Show the home page if the user selects the 'Home' option
    if choice == 'Home':

        st.subheader('Answer the following questions')
        # Get the user inputs
        review1 = st.text_input('How was the course experience?')
        review2 = st.text_input('Tell us about the instructor?')
        review3 = st.text_input('Was the material provided useful?')
        # Perform sentiment analysis and show the results
        if st.button('Predict'):
            result1 = predict_sentiment(review1)
            result2 = predict_sentiment(review2)
            result3 = predict_sentiment(review3)
            st.success(f"Course experience: {result1}")
            st.success(f"Instructor: {result2}")
            st.success(f"Material: {result3}")

            # Count the number of positive and negative reviews
            results = {'Course experience': result1, 'Instructor': result2, 'Useful material': result3}
            positive_count = sum([1 for r in results.values() if r == 'Positive review'])
            negative_count = sum([1 for r in results.values() if r == 'Negative review'])
            st.write(f"Number of positive reviews: {positive_count}")
            st.write(f"Number of negative reviews: {negative_count}")

            # Show analytics using a bar chart
            df_counts = pd.DataFrame({'Sentiment': ['Positive review', 'Negative review'], 
                                      'Count': [positive_count, negative_count]})
            fig, ax = plt.subplots()
            ax.bar(df_counts['Sentiment'], df_counts['Count'], color=['blue', 'orange'])
            ax.set_title('Sentiment Analysis Results')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)

    elif choice == 'Forms':
        st.subheader('Course Evaluation/Feedback Form :')
        # Check if user is an admin
        is_admin = st.sidebar.checkbox('Admin access')

        # Login process for admin
        if is_admin:
            st.subheader('Admin Login')
            username = st.text_input('Username:')
            password = st.text_input('Password:', type='password')
            if st.button('Login'):
                if username == 'admin' and password == 'password':
                    st.success('Logged in as admin.')
                else:
                    st.error('Incorrect username or password.')

        # User review form
        else:
            with st.form(key='review_form'):
                usn = st.text_input('Enter USN:')
                name = st.text_input('Your Name:')
                review1 = st.text_input('How was the course experience?')
                review2 = st.text_input('Tell us about the instructor?')
                review3 = st.text_input('Was the material provided useful?')
                submitted = st.form_submit_button('Submit')

                # Store the reviews in the database
                if submitted:
                    usn_pattern = r'^[1-9][A-Za-z]{2}\d{2}[A-Za-z]{2}\d{3}$'
                    if not usn or not name or not review1 or not review2 or not review3:
                        st.error('Please fill in all fields.')
                    elif len(usn) != 10:
                        st.error('Incorrect USN. Please enter a 10 character USN.')
                    elif not re.match(usn_pattern, usn):
                        st.error('Incorrect USN. Please enter a valid USN (eg:4JK16CS001). ')
                    else:
                        c.execute("SELECT * FROM reviews WHERE usn=%s", (usn,))
                        existing_review = c.fetchone()
                        if existing_review:
                            # If the usn already exists, show an error message
                            st.error(f"Review for {usn} already exists.")
                        else:
                            sentiment1 = predict_sentiment(review1)
                            sentiment2 = predict_sentiment(review2)
                            sentiment3 = predict_sentiment(review3)
                            c.execute("INSERT INTO reviews (usn, name, course_experience, sentiment1, instructor, sentiment2, material, sentiment3) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",(usn, name, review1, sentiment1, review2, sentiment2, review3, sentiment3))

                            conn.commit()
                            st.success('Thank you, Your feedback is submitted.')

        # Display reviews for admin
        if is_admin and username == 'admin' and password == 'password':
            reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
            # Check if there are any reviews to display
            if len(reviews_df) == 0:
                st.warning('No reviews to display.')
            else:
                st.header('Reviews Table')
                st.dataframe(reviews_df)
                
                if st.button('Refresh'):
                    reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
                    st.experimental_rerun()
                    
                
                
                # Create a beta expander for delete reviews feature
                with st.expander('QUICK TOOLS MENU'):
                    st.subheader('Download all reviews')
                    st.write('Downloads reviews in xlsx format')
                    if st.button('Download'):
                        # Convert DataFrame to Excel file in memory
                        excel_file = io.BytesIO()
                        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                            reviews_df.to_excel(writer, index=False, sheet_name='Reviews')
                        excel_file.seek(0)

                        # Set up the download link
                        st.download_button('Download Database', data=excel_file, file_name='reviews_database.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    st.subheader('Delete all reviews')
                    st.write('Use this button to delete all reviews at one click')
                    if st.button('Delete All Reviews'):
                        c.execute("DELETE FROM reviews")
                        conn.commit()
                        st.success('All reviews have been deleted.')
                        reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
                        st.experimental_rerun()
                    else:
                        st.subheader('Delete a particular reviews')
                        st.write('Use this button to delete particular review based on USN')
                        usns = ['Select USN'] + reviews_df['usn'].unique().tolist()  # Add initial placeholder option
                        selected_usn = st.selectbox('Select USN:', usns)
                        if selected_usn != 'Select USN':  # Check if a valid USN is selected
                            if st.button('Delete'):
                                c.execute("DELETE FROM reviews WHERE usn=%s", (selected_usn,))
                                conn.commit()
                                st.success(f"Review for {selected_usn} deleted.")
                                reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)
                                st.experimental_rerun()

                show_sentiment_wise_analytics(reviews_df)

    # Show the analytics page if the user selects the 'Analytics' option
    elif choice == 'Analytics':
        st.subheader('Upload an excel file to perform sentiment analysis')
        file = st.file_uploader("Upload an existing Excel file", type=["xlsx", "xls"])
        if file is not None:
            df = pd.read_excel(file)
            column_name = st.selectbox('Select column to analyze:', df.columns)
            st.write(df)

            # Show analytics in a separate tab on click of a button
            if st.button('Show Analytics'):
                show_analytics(df, column_name)


# Run the app
if __name__ == '__main__':
    run_sentiment_app()
