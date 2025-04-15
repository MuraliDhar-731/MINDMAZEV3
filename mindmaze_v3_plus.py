import streamlit as st
import random
import time
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="MindMaze V3+", layout="centered")
st.title("🧠 MindMaze V3+ – Multi-Puzzle Challenge")

# Sidebar
demo_mode = st.sidebar.checkbox("🧪 Play in Demo Mode (No Logging / Prediction)")

st.sidebar.markdown("📘 **MindMaze Puzzle Demos**")
if st.sidebar.checkbox("👀 Show Examples for All Modes"):
    st.sidebar.markdown("### 🔢 Number Sort")
    st.sidebar.markdown("- **Easy**: [6, 3, 12, 8] → ✅ 3,6,8,12")
    st.sidebar.markdown("- **Medium**: [14, 2, 8, 19, 7, 4] → ✅ 2,4,7,8,14,19")
    st.sidebar.markdown("- **Hard**: [5,11,1,16,8,4,9,13,2] → ✅ 1,2,4,5,8,9,11,13,16")
    st.sidebar.markdown("### 🧮 Math Grid (Lite)")
    st.sidebar.markdown("- Target: 21 | [13, 8, 4, 6] → ✅ 13,8")
    st.sidebar.markdown("### 🧮 Math Grid (3x3)")
    st.sidebar.markdown("- Target: 15 | [1–9] → ✅ Row Hint: [4,5,6]")
    st.sidebar.markdown("### 🔤 Word Logic")
    st.sidebar.markdown("- Word: `smile` → ❌ `table`, ✅ `smile`")

# Theme toggle
theme = st.radio("🎨 Choose Theme:", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

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

# Performance badge
def performance_feedback(solve_time):
    if solve_time <= 7:
        st.balloons()
        st.success("🏅 Speed Star! Excellent job!")
    elif solve_time <= 15:
        st.info("⭐ Good pace – keep it up!")
    else:
        st.warning("⏱️ Try to solve faster next time!")

# User login
user = st.text_input("👤 Enter your name to begin:")
if not user:
    st.stop()

# Puzzle selection
puzzle_type = st.radio("🧠 Choose Puzzle Type:", ["Number Sort", "Math Grid (Lite)", "Math Grid (3x3)", "Word Logic"])


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
            guess = list(map(int, user_input.split(',')))
            if guess == sorted(st.session_state["puzzle"]):
                if not st.session_state["solved"]:
                    solve_time = round(time.time() - st.session_state["start_time"], 2)
                    st.success(f"🎉 Solved in {solve_time} seconds!")
                    prediction = model.predict([[n_numbers, solve_time]])[0] if model and not demo_mode else "Demo"
                    st.info(f"🔮 Predicted difficulty: {prediction}")
                    performance_feedback(solve_time)
                    if not demo_mode:
                        log_result(user, "Number Sort", level, solve_time, prediction)
                    st.session_state["solved"] = True
            else:
                st.warning("❌ Not sorted. Try again.")
        except:
            st.error("⚠️ Invalid input.")


elif puzzle_type == "Math Grid (Lite)":
    st.subheader("🧮 Match the sum")

    if "math_lite_nums" not in st.session_state:
        all_possible = list(range(1, 25))
        pair = random.sample(all_possible, 2)
        target_sum = sum(pair)
        extra = random.sample(list(set(all_possible) - set(pair)), 2)
        numbers = pair + extra
        random.shuffle(numbers)
        st.session_state.update({
            "math_lite_pair": pair,
            "math_lite_target": target_sum,
            "math_lite_nums": numbers,
            "start_time": time.time()
        })

    target_sum = st.session_state["math_lite_target"]
    numbers = st.session_state["math_lite_nums"]

    st.write(f"🎯 Target Sum: {target_sum}")
    st.write(f"🧩 Numbers: {numbers}")

    if st.checkbox("💡 Show me a hint"):
        valid_pairs = [(a, b) for i, a in enumerate(numbers)
                       for j, b in enumerate(numbers)
                       if i < j and a + b == target_sum]
        if valid_pairs:
            st.info(f"✅ Try this pair: {random.choice(valid_pairs)}")
        else:
            st.warning("⚠️ No valid pairs!")

    guess = st.text_input("✏️ Enter 2 numbers that sum to target (e.g., 5,10):")
    if guess:
        try:
            nums = list(map(int, guess.split(',')))
            nums = sorted(nums)
            is_valid = (
                len(nums) == 2 and nums[0] != nums[1] and
                all(n in numbers for n in nums) and
                sum(nums) == target_sum
            )
            if is_valid:
                solve_time = round(time.time() - st.session_state["start_time"], 2)
                st.success(f"✅ Correct! Solved in {solve_time}s")
                prediction = model.predict([[2, solve_time]])[0] if model and not demo_mode else "Demo"
                st.info(f"🔮 Predicted difficulty: {prediction}")
                performance_feedback(solve_time)
                if not demo_mode:
                    log_result(user, "Math Grid (Lite)", "Custom", solve_time, prediction)
                for k in ["math_lite_pair", "math_lite_target", "math_lite_nums"]:
                    st.session_state.pop(k, None)
            else:
                st.error("❌ Incorrect. Try again!")
        except:
            st.error("⚠️ Invalid format")

    if st.button("🔄 New Puzzle"):
        for k in ["math_lite_pair", "math_lite_target", "math_lite_nums"]:
            st.session_state.pop(k, None)
        st.experimental_rer_


elif puzzle_type == "Math Grid (3x3)":
    st.subheader("🧮 Fill a 3x3 Grid to Match Target Row & Column Sums")

    @st.cache_data(show_spinner=True)
    def get_feasible_targets_with_labels():
        valid = []
        with st.spinner("🔄 Finding valid target sums..."):
            for t in range(15, 61):
                for _ in range(1000):
                    nums = random.sample(range(1, 25), 9)
                    grid = np.array(nums).reshape(3, 3)
                    if all(sum(row) == t for row in grid) and all(sum(col) == t for col in grid.T):
                        label = "Easy" if t <= 30 else "Medium" if t <= 45 else "Hard"
                        valid.append((t, label))
                        break
        return sorted(valid)

    def generate_solvable_grid(target_sum):
        for _ in range(5000):
            nums = random.sample(range(1, 25), 9)
            grid = np.array(nums).reshape(3, 3)
            if all(sum(row) == target_sum for row in grid) and all(sum(col) == target_sum for col in grid.T):
                return nums
        return None

    feasible_targets = get_feasible_targets_with_labels()

    # 🧠 First-time load: auto-init grid
    if "grid_initialized" not in st.session_state:
        if feasible_targets:
            chosen_pair = random.choice(feasible_targets)
        else:
            chosen_pair = (45, "Medium")  # Fallback
            st.warning("⚠️ Using default target sum: 45 (Medium)")
        st.session_state["target_sum"] = chosen_pair[0]
        st.session_state["difficulty"] = chosen_pair[1]
        st.session_state["available_numbers"] = generate_solvable_grid(st.session_state["target_sum"])
        st.session_state["grid_initialized"] = True

    # 🎲 Manual reset
    if st.checkbox("🎲 Generate Random Solvable Grid"):
        feasible_targets = get_feasible_targets_with_labels()
        if feasible_targets:
            chosen_pair = random.choice(feasible_targets)
        else:
            chosen_pair = (45, "Medium")
            st.warning("⚠️ Default target used: 45 (Medium)")
        st.session_state["target_sum"] = chosen_pair[0]
        st.session_state["difficulty"] = chosen_pair[1]
        st.session_state["available_numbers"] = generate_solvable_grid(st.session_state["target_sum"])

    target_sum = st.session_state["target_sum"]
    available_numbers = st.session_state["available_numbers"]

    if not available_numbers:
        st.error("🚫 Could not generate a valid 3x3 grid. Try again.")
        st.stop()

    st.info(f"🎯 Target Sum: **{target_sum} ({st.session_state['difficulty']})**")
    st.write("🧩 Use these numbers (no repeats):", available_numbers)

    if "grid3x3" not in st.session_state:
        st.session_state.grid3x3 = [[0]*3 for _ in range(3)]
        st.session_state.grid_start = time.time()

    if st.checkbox("💡 Show a filled row hint"):
        row_hint = random.sample(available_numbers, 3)
        while sum(row_hint) != target_sum:
            row_hint = random.sample(available_numbers, 3)
        st.info(f"🧩 Hint Row: {row_hint} (sum: {target_sum})")

    grid_input = []
    for row in range(3):
        cols = st.columns(3)
        row_vals = []
        for col in range(3):
            key = f"cell_{row}_{col}"
            val = cols[col].number_input(" ", min_value=0, max_value=99, key=key, label_visibility="collapsed")
            row_vals.append(val)
        grid_input.append(row_vals)

    if st.button("✅ Check Grid"):
        grid_np = np.array(grid_input)
        all_values = grid_np.flatten().tolist()
        row_sums = [sum(r) for r in grid_np]
        col_sums = [sum(c) for c in grid_np.T]

        for i, rs in enumerate(row_sums):
            st.write(f"🔢 Row {i+1} sum: {rs} {'✅' if rs == target_sum else '❌'}")
        for j, cs in enumerate(col_sums):
            st.write(f"🔢 Column {j+1} sum: {cs} {'✅' if cs == target_sum else '❌'}")

        if all(r == target_sum for r in row_sums) and all(c == target_sum for c in col_sums) and set(all_values) == set(available_numbers):
            solve_time = round(time.time() - st.session_state.grid_start, 2)
            st.success(f"🎉 Solved in {solve_time} seconds!")
            prediction = model.predict([[9, solve_time]])[0] if model and not demo_mode else "Demo"
            st.info(f"🔮 Predicted difficulty: {prediction}")
            performance_feedback(solve_time)
            if not demo_mode:
                log_result(user, "Math Grid (3x3)", "3x3", solve_time, prediction)
        else:
            st.warning("❌ Some values or sums are incorrect. Try again!")




elif puzzle_type == "Word Logic":
    st.subheader("🔤 Guess the 5-letter word in 3 tries")
    word_list = ["apple", "grape", "brain", "table", "smile"]

    # Initialize session state
    if "secret_word" not in st.session_state:
        st.session_state["secret_word"] = random.choice(word_list)
        st.session_state["attempts"] = 0
        st.session_state["guess_history"] = []

    secret = st.session_state["secret_word"]

    if st.checkbox("💡 Reveal last letter"):
        st.info(f"🧠 The last letter is **{secret[-1].upper()}**")

    guess = st.text_input("📝 Your guess (5-letter word):")

    if guess:
        st.session_state["attempts"] += 1
        feedback = []
        for i in range(len(guess)):
            if guess[i] == secret[i]:
                feedback.append("🟩")  # Correct letter and position
            elif guess[i] in secret:
                feedback.append("🟨")  # Correct letter, wrong position
            else:
                feedback.append("⬜")  # Not in word
        st.session_state["guess_history"].append((guess, " ".join(feedback)))

        if guess == secret:
            solve_time = st.session_state["attempts"] * 5
            st.success("🎉 Correct!")
            st.write("🔢 Feedback:", " ".join(feedback))
            prediction = model.predict([[5, solve_time]])[0] if model and not demo_mode else "Demo"
            st.info(f"🔮 Predicted difficulty: {prediction}")
            if not demo_mode:
                log_result(user, "Word Logic", "5-letter", solve_time, prediction)

            # Feedback stars
            if st.session_state["attempts"] == 1:
                st.success("🌟 Genius! Guessed in 1 try!")
            elif st.session_state["attempts"] == 2:
                st.info("⭐ Smart guesser!")
            else:
                st.warning("🔁 Took a few tries, but you made it!")

            # Reset game
            st.session_state["secret_word"] = random.choice(word_list)
            st.session_state["attempts"] = 0
            st.session_state["guess_history"] = []

        elif st.session_state["attempts"] >= 3:
            st.error(f"❌ Out of tries. The word was: **{secret.upper()}**")
            st.session_state["secret_word"] = random.choice(word_list)
            st.session_state["attempts"] = 0
            st.session_state["guess_history"] = []
        else:
            st.info("📜 Guess History:")
            for past_guess, past_fb in st.session_state["guess_history"]:
                st.markdown(f"`{past_guess}` → {past_fb}")
