import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import streamlit as st
import itertools
# /Users/ashirshah/Desktop/mpdp10-attachments/pick-3_20240201-0000_20240430-0956.csv
pick3_data = pd.read_csv('/Users/ashirshah/Desktop/mpdp10-attachments/pick-3_20240201-0000_20240430-0956.csv')
pick4_data = pd.read_csv('/Users/ashirshah/Desktop/mpdp10-attachments/pick-4_20240201-0000_20240430-1007.csv')

# Preprocessing data
def preprocess_data(df, num_balls):
    df['drawn_numbers'] = df.apply(lambda row: ''.join([str(row[f'Ball {i+1}']) for i in range(num_balls)]), axis=1)
    return df

pick3_data = preprocess_data(pick3_data, 3)
pick4_data = preprocess_data(pick4_data, 4)

# Generating alternate numbers
def generate_alternate_numbers(number):
    alternates = []
    for digit in number:
        alternates.append((int(digit) - 1) % 10)
        alternates.append((int(digit) + 1) % 10)
    return alternates

pick3_data['alternates'] = pick3_data['drawn_numbers'].apply(generate_alternate_numbers)
pick4_data['alternates'] = pick4_data['drawn_numbers'].apply(generate_alternate_numbers)

# Feature extraction
def extract_features(data, num_balls):
    features = []
    for index, row in data.iterrows():
        drawn_numbers = [int(digit) for digit in row['drawn_numbers']]
        alternates = row['alternates']
        features.append(alternates + drawn_numbers)
    return features

features_pick3 = extract_features(pick3_data, 3)
features_pick4 = extract_features(pick4_data, 4)

target_pick3 = pick3_data['drawn_numbers'].apply(lambda x: int(x))
target_pick4 = pick4_data['drawn_numbers'].apply(lambda x: int(x))

# Split data into training and testing sets
X_train_pick3, X_test_pick3, y_train_pick3, y_test_pick3 = train_test_split(features_pick3, target_pick3, test_size=0.2, random_state=42)
X_train_pick4, X_test_pick4, y_train_pick4, y_test_pick4 = train_test_split(features_pick4, target_pick4, test_size=0.2, random_state=42)

# Model training
model_pick3 = RandomForestClassifier()
model_pick3.fit(X_train_pick3, y_train_pick3)

model_pick4 = RandomForestClassifier()
model_pick4.fit(X_train_pick4, y_train_pick4)

# Compute accuracy
y_pred_pick3 = model_pick3.predict(X_test_pick3)
y_pred_pick4 = model_pick4.predict(X_test_pick4)

accuracy_pick3 = accuracy_score(y_test_pick3, y_pred_pick3)
accuracy_pick4 = accuracy_score(y_test_pick4, y_pred_pick4)

print(f"Pick 3 model accuracy: {accuracy_pick3:.2f}")
print(f"Pick 4 model accuracy: {accuracy_pick4:.2f}")

# Save the models
joblib.dump(model_pick3, 'model_pick3.pkl')
joblib.dump(model_pick4, 'model_pick4.pkl')


# Load the saved models
model_pick3 = joblib.load('model_pick3.pkl')
model_pick4 = joblib.load('model_pick4.pkl')

# Function to generate alternate numbers
def generate_alternate_numbers(number, num_balls):
    alternates = []
    for digit in number:
        alternates.append((int(digit) - 1) % 10)
        alternates.append((int(digit) + 1) % 10)
    if num_balls == 4:
        # Add 1 extra number for Pick 4 as per the provided logic
        alternates.append((int(number[-1]) + 1) % 10)
    return alternates

# Prediction and ranking with unique values
def predict_and_rank(model, last_draw, num_balls, num_sets):
    alternates = generate_alternate_numbers(last_draw, num_balls)
    all_combinations = [list(comb) + [int(digit) for digit in last_draw] for comb in itertools.product(alternates, repeat=num_balls)]
    all_combinations = [comb[:2*num_balls] + [int(digit) for digit in last_draw] for comb in all_combinations]
    
    # Remove duplicates
    unique_combinations = []
    [unique_combinations.append(item) for item in all_combinations if item not in unique_combinations]
    
    probabilities = model.predict_proba(unique_combinations)[:, 1]
    ranked_combinations = [x for _, x in sorted(zip(probabilities, unique_combinations), reverse=True)]
    return ranked_combinations[:num_sets]

# Streamlit app
st.title("Lottery Prediction")

# Pick 3 Section
st.header("Pick 3")
last_draw_pick3 = st.text_input("Enter the last draw (3 digits):", key="pick3_draw")
draw_time_pick3 = st.radio("Select draw time for Pick 3:", ('Day', 'Night'), key="pick3_time")
num_sets_pick3 = st.selectbox("Select the number of sets to generate:", [5, 10, 15], key="pick3_sets")

if st.button("Run program for Pick 3", key="pick3_button"):
    if last_draw_pick3 and len(last_draw_pick3) == 3:
        predicted_pick3 = predict_and_rank(model_pick3, last_draw_pick3, 3, num_sets_pick3)
        st.subheader("Predicted Pick 3 sets:")
        for i, pred in enumerate(predicted_pick3, 1):
            st.write(f"{i}: {pred}")
    else:
        st.error("Please enter a valid 3-digit last draw for Pick 3.")

# Pick 4 Section
st.header("Pick 4")
last_draw_pick4 = st.text_input("Enter the last draw (4 digits):", key="pick4_draw")
draw_time_pick4 = st.radio("Select draw time for Pick 4:", ('Day', 'Night'), key="pick4_time")
num_sets_pick4 = st.selectbox("Select the number of sets to generate:", [5, 10, 15], key="pick4_sets")

if st.button("Run program for Pick 4", key="pick4_button"):
    if last_draw_pick4 and len(last_draw_pick4) == 4:
        predicted_pick4 = predict_and_rank(model_pick4, last_draw_pick4, 4, num_sets_pick4)
        st.subheader("Predicted Pick 4 sets:")
        for i, pred in enumerate(predicted_pick4, 1):
            st.write(f"{i}: {pred}")
    else:
        st.error("Please enter a valid 4-digit last draw for Pick 4.")
