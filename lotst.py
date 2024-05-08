import streamlit as st
from itertools import permutations
import pandas as pd
import numpy as np

def calculate_alternate_numbers(numbers):
    return [(number + 5) % 10 for number in numbers]

def generate_combinations(numbers, num_digits):
    return set(permutations(numbers, num_digits))

def rank_combinations(combinations, historical_data):
    # Example simple ranking based on frequency of occurrence in historical data
    combination_counts = {}
    for combo in combinations:
        # Convert tuple to string to match historical format
        combo_str = ''.join(map(str, combo))
        combination_counts[combo] = historical_data.get(combo_str, 0)
    # Sort combinations by historical frequency
    ranked_combinations = sorted(combination_counts.items(), key=lambda item: item[1], reverse=True)
    return [combo for combo, count in ranked_combinations]

# Historical frequency data simulation
historical_data_pick_3 = {'467': 10, '476': 5, '647': 8, '674': 3, '746': 15, '764': 7}
historical_data_pick_4 = {'1229': 6, '1292': 2, '1922': 9, '2129': 12, '2192': 4, '2219': 1}

# Streamlit interface
st.title('Lottery Combination Generator for Pick 3 and Pick 4')

st.header('Input the last draw numbers for Pick 3')
ball1_pick3 = st.number_input('Enter Ball 1 for Pick 3:', min_value=0, max_value=9, value=0, step=1)
ball2_pick3 = st.number_input('Enter Ball 2 for Pick 3:', min_value=0, max_value=9, value=0, step=1)
ball3_pick3 = st.number_input('Enter Ball 3 for Pick 3:', min_value=0, max_value=9, value=0, step=1)

st.header('Input the last draw numbers for Pick 4')
ball1_pick4 = st.number_input('Enter Ball 1 for Pick 4:', min_value=0, max_value=9, value=0, step=1)
ball2_pick4 = st.number_input('Enter Ball 2 for Pick 4:', min_value=0, max_value=9, value=0, step=1)
ball3_pick4 = st.number_input('Enter Ball 3 for Pick 4:', min_value=0, max_value=9, value=0, step=1)
ball4_pick4 = st.number_input('Enter Ball 4 for Pick 4:', min_value=0, max_value=9, value=0, step=1)

if st.button('Generate and Rank Combinations'):
    last_draw_pick_3 = [ball1_pick3, ball2_pick3, ball3_pick3]
    last_draw_pick_4 = [ball1_pick4, ball2_pick4, ball3_pick4, ball4_pick4]
    
    alternate_numbers_pick_3 = calculate_alternate_numbers(last_draw_pick_3)
    alternate_numbers_pick_4 = calculate_alternate_numbers(last_draw_pick_4)
    
    combinations_pick_3 = generate_combinations(alternate_numbers_pick_3, 3)
    combinations_pick_4 = generate_combinations(alternate_numbers_pick_4, 4)
    
    ranked_combinations_pick_3 = rank_combinations(combinations_pick_3, historical_data_pick_3)
    ranked_combinations_pick_4 = rank_combinations(combinations_pick_4, historical_data_pick_4)
    
    st.subheader("Ranked Combinations for Pick 3:")
    st.write(ranked_combinations_pick_3)
    
    st.subheader("Ranked Combinations for Pick 4:")
    st.write(ranked_combinations_pick_4)
