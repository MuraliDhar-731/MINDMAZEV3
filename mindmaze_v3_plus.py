import streamlit as st
import random
import time
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="MindMaze V3+", layout="centered")
st.title("🧠 MindMaze V3+ – Multi-Puzzle Challenge")

# User login
user = st.text_input("👤 Enter your name to begin:")
if not user:
    st.stop()

# Theme toggle
theme = st.radio("🎨 Choose Theme:", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Puzzle type selection
puzzle_type = st.radio("🧠 Choose Puzzle Type:", ["Number Sort", "Math Grid (Lite)", "Math Grid (3x3)", "Word Logic"])

# Load ML model
@st.cache_resource
def load_model():
    try:
        return joblib.load("difficulty_model.pkl")
    except:
        return None

model = load_model()

# Logging
def log_result(user, puzzle_type, level, solve_time, prediction):
    log = pd.DataFrame([{
        "user": user,
        "puzzle_type": puzzle_type,
        "level": level,
        "solve_time": solve_time,
        "predicted_level": prediction,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    log.to_csv("mindmaze_leaderboard.csv", mode='a',
               header=not os.path.exists("mindmaze_leaderboard.csv"), index=False)

# ========== Number Sort ==========
if puzzle_type == "Number Sort":
    st.subheader("🔢 Arrange numbers in order")
    level = st.selectbox("🎯 Choose Difficulty", ["Easy", "Medium", "Hard"])
    level_map = {"Easy": 4, "Medium": 6, "Hard": 9}
    n_numbers = level_map[level]

    if "start_time" not in st.session_state:
        st.session_state["start_time"] = time.time()
    if "puzzle" not in st.session_state or st.session_state.get("size") != n_numbers:
        st.session_state["puzzle"] = random.sample(range(1, 20), n_numbers)
        st.session_state["size"] = n_numbers
        st.session_state["solved"] = False

    st.write("🧩 Puzzle:", st.session_state["puzzle"])

    if st.checkbox("💡 Show me a hint (sorted version)"):
        st.info(f"🔍 Hint: {sorted(st.session_state['puzzle'])}")

    user_input = st.text_input("✏️ Type numbers in order, e.g., 1,2,3:")

    if user_input:
        try:
            guess = list(map(lambda x: int(x.strip()), user_input.split(',')))
            if guess == sorted(st.session_state["puzzle"]):
                if not st.session_state["solved"]:
                    solve_time = round(time.time() - st.session_state["start_time"], 2)
                    st.success(f"🎉 Solved in {solve_time} seconds!")
                    prediction = model.predict([[n_numbers, solve_time]])[0] if model else "Unknown"
                    st.info(f"🔮 Predicted difficulty: {prediction}")
                    log_result(user, "Number Sort", level, solve_time, prediction)
                    st.session_state["solved"] = True
            else:
                st.warning("❌ Not sorted. Try again.")
        except:
            st.error("⚠️ Invalid input.")

# ========== Math Grid (Lite) ==========
elif puzzle_type == "Math Grid (Lite)":
    st.subheader("🧮 Match the sum")

    if "math_lite_nums" not in st.session_state:
        all_possible = list(range(1, 25))
        pair = random.sample(all_possible, 2)
        target_sum = sum(pair)

        remaining = list(set(all_possible) - set(pair))
        extra = random.sample(remaining, 2)
        numbers = pair + extra
        random.shuffle(numbers)

        st.session_state["math_lite_pair"] = pair
        st.session_state["math_lite_target"] = target_sum
        st.session_state["math_lite_nums"] = numbers
        st.session_state["start_time"] = time.time()

    target_sum = st.session_state["math_lite_target"]
    numbers = st.session_state["math_lite_nums"]
    pair = st.session_state["math_lite_pair"]

    st.write(f"🎯 Target Sum: {target_sum}")
    st.write(f"🧩 Numbers: {numbers}")

    if st.checkbox("💡 Show me a hint"):
        valid_pairs = [(a, b) for i, a in enumerate(numbers)
                       for j, b in enumerate(numbers)
                       if i < j and a + b == target_sum]
        if valid_pairs:
            hint_pair = random.choice(valid_pairs)
            st.info(f"✅ Here's a valid pair: {hint_pair}")
        else:
            st.warning("⚠️ No valid pairs found.")

    math_guess = st.text_input("✏️ Enter 2 numbers that sum to target (e.g., 5,10):")

    if math_guess:
        try:
            nums = list(map(lambda x: int(x.strip()), math_guess.split(',')))
            nums = sorted(nums)
            is_valid = (
                len(nums) == 2 and
                nums[0] != nums[1] and
                all(n in numbers for n in nums) and
                sum(nums) == target_sum
            )

            st.caption(f"🔍 You entered: {nums} | Target: {target_sum}")

            if is_valid:
                st.success("✅ Correct!")
                solve_time = round(time.time() - st.session_state["start_time"], 2)
                prediction = model.predict([[2, solve_time]])[0] if model else "Unknown"
                log_result(user, "Math Grid (Lite)", "Custom", solve_time, prediction)
                for key in ["math_lite_pair", "math_lite_target", "math_lite_nums", "start_time"]:
                    st.session_state.pop(key, None)
            else:
                st.error("❌ Incorrect. Make sure:")
                st.markdown("- The **sum** matches the target")
                st.markdown("- The **numbers are different**")
                st.markdown("- The numbers are from the given set")
        except:
            st.error("⚠️ Please enter two valid numbers separated by comma.")

    if st.button("🔄 New Puzzle"):
        for key in ["math_lite_pair", "math_lite_target", "math_lite_nums", "start_time"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

# ========== Math Grid (3x3) ==========
elif puzzle_type == "Math Grid (3x3)":
    st.subheader("🧮 Fill a 3x3 Grid to Match Target Row & Column Sums")
    target_sum = st.slider("🎯 Set target sum per row/column", 10, 30, 15)
    available_numbers = random.sample(range(1, 20), 9)
    st.write("🧩 Use these numbers (no repeats):", available_numbers)

    if "grid3x3" not in st.session_state:
        st.session_state.grid3x3 = [[0]*3 for _ in range(3)]
        st.session_state.grid_start = time.time()

    if st.checkbox("💡 Show a filled row hint"):
        hint_row = random.randint(0, 2)
        row_hint = random.sample(available_numbers, 3)
        while sum(row_hint) != target_sum:
            row_hint = random.sample(available_numbers, 3)
        st.info(f"Row {hint_row+1} Hint: {row_hint}")

    grid_input = []
    for row in range(3):
        cols = st.columns(3)
        row_vals = []
        for col in range(3):
            key = f"cell_{row}_{col}"
            val = st.number_input(f"", min_value=0, max_value=99, key=key, label_visibility="collapsed")
            row_vals.append(val)
        grid_input.append(row_vals)

    if st.button("✅ Check Grid"):
        grid_np = np.array(grid_input)
        rows_correct = all(sum(r) == target_sum for r in grid_np)
        cols_correct = all(sum(c) == target_sum for c in grid_np.T)
        all_values = grid_np.flatten().tolist()

        if rows_correct and cols_correct and set(all_values) == set(available_numbers):
            solve_time = round(time.time() - st.session_state.grid_start, 2)
            st.success(f"🎉 Solved in {solve_time} seconds!")
            prediction = model.predict([[9, solve_time]])[0] if model else "Unknown"
            st.info(f"🔮 Predicted difficulty: {prediction}")
            log_result(user, "Math Grid (3x3)", "3x3", solve_time, prediction)
        else:
            st.error("❌ Incorrect! Check row/col sums and number usage.")

# ========== Word Logic ==========
elif puzzle_type == "Word Logic":
    st.subheader("🔤 Guess the 5-letter word in 3 tries")
    word_list = ["apple", "grape", "brain", "table", "smile"]
    if "secret_word" not in st.session_state:
        st.session_state["secret_word"] = random.choice(word_list)
        st.session_state["attempts"] = 0

    guess = st.text_input("📝 Your guess:")

    if st.checkbox("💡 Reveal one correct letter in the correct spot"):
        word = st.session_state["secret_word"]
        idx = random.randint(0, 4)
        st.info(f"🧩 The letter at position {idx+1} is **{word[idx].upper()}**")

    if guess:
        st.session_state["attempts"] += 1
        if guess == st.session_state["secret_word"]:
            st.success("🎉 Correct!")
            solve_time = st.session_state["attempts"] * 5
            prediction = model.predict([[5, solve_time]])[0] if model else "Unknown"
            log_result(user, "Word Logic", "5-letter", solve_time, prediction)
            st.session_state["secret_word"] = random.choice(word_list)
            st.session_state["attempts"] = 0
        elif st.session_state["attempts"] >= 3:
            st.error(f"❌ Out of tries. The word was: {st.session_state['secret_word']}")
            st.session_state["secret_word"] = random.choice(word_list)
            st.session_state["attempts"] = 0
        else:
            feedback = []
            secret = st.session_state["secret_word"]
            for i in range(len(guess)):
                if guess[i] == secret[i]:
                    feedback.append("✅")
                elif guess[i] in secret:
                    feedback.append("⚠️")
                else:
                    feedback.append("❌")
            st.write("Feedback:", " ".join(feedback))
