import streamlit as st
import sqlite3
import pandas as pd
import uuid
from datetime import datetime

# --- Configuration ---
DB_NAME = "stress_readiness.db"
# IMPORTANT: Change this default password for actual use!
TRAINER_PASSWORD = "password123"

# --- Database Setup ---

def setup_database():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create Clients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Clients (
            client_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            unique_code TEXT UNIQUE NOT NULL
        )
    """)
    # Create Responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            fatigue INTEGER,
            sleep_quality INTEGER,
            sleep_hours INTEGER,
            soreness INTEGER,
            stress INTEGER,
            FOREIGN KEY (client_id) REFERENCES Clients (client_id)
        )
    """)
    conn.commit()
    conn.close()

# --- Database Operations ---

def add_client(name):
    """Adds a new client, generates a unique code, and returns the code."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    unique_code = uuid.uuid4().hex[:8].upper() # Generate an 8-char uppercase code
    try:
        cursor.execute("INSERT INTO Clients (name, unique_code) VALUES (?, ?)", (name, unique_code))
        conn.commit()
        return unique_code
    except sqlite3.IntegrityError:
        # Handle rare case where generated code already exists (very unlikely)
        conn.rollback()
        return None # Indicate failure
    finally:
        conn.close()

def get_client_by_code(code):
    """Retrieves client details (id, name) based on the unique code."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT client_id, name FROM Clients WHERE unique_code = ?", (code,))
    client = cursor.fetchone()
    conn.close()
    return client # Returns (client_id, name) or None if not found

def get_all_clients():
    """Retrieves all client names and their unique codes."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch client_id along with name and code for easier mapping later
    cursor.execute("SELECT client_id, name, unique_code FROM Clients ORDER BY name")
    clients = cursor.fetchall()
    conn.close()
    return clients # Returns list of (client_id, name, unique_code) tuples

def add_response(client_id, fatigue, sleep_quality, sleep_hours, soreness, stress):
    """Adds a new questionnaire response to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute("""
        INSERT INTO Responses (client_id, timestamp, fatigue, sleep_quality, sleep_hours, soreness, stress)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (client_id, timestamp, fatigue, sleep_quality, sleep_hours, soreness, stress))
    conn.commit()
    conn.close()

def get_responses_by_client(client_id):
    """Retrieves all responses for a specific client, ordered by timestamp."""
    conn = sqlite3.connect(DB_NAME)
    # Ensure client_id is an integer
    try:
        client_id_int = int(client_id)
    except (ValueError, TypeError):
        st.error("Invalid Client ID format for fetching responses.")
        return pd.DataFrame()

    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, fatigue, sleep_quality, sleep_hours, soreness, stress
        FROM Responses
        WHERE client_id = ?
        ORDER BY timestamp ASC
    """, (client_id_int,))
    responses = cursor.fetchall()
    conn.close()
    # Convert to DataFrame
    if responses:
        df = pd.DataFrame(responses, columns=['Timestamp', 'Fatigue', 'Sleep Quality', 'Sleep Hours', 'Soreness', 'Stress'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
    else:
        return pd.DataFrame() # Return empty DataFrame if no responses

# --- NEW: Function to count responses ---
def count_responses_by_client(client_id):
    """Counts the total number of responses submitted by a specific client."""
    conn = sqlite3.connect(DB_NAME)
     # Ensure client_id is an integer
    try:
        client_id_int = int(client_id)
    except (ValueError, TypeError):
        st.error("Invalid Client ID format for counting responses.")
        return 0

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Responses WHERE client_id = ?", (client_id_int,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

# --- Streamlit App ---

# Initialize database on first run
setup_database()

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['client_id'] = None
    st.session_state['client_name'] = None
if 'trainer_logged_in' not in st.session_state:
    st.session_state['trainer_logged_in'] = False
if 'selected_client_id_trainer' not in st.session_state:
    st.session_state['selected_client_id_trainer'] = None


# --- Trainer Sidebar / Access ---
st.sidebar.title("Trainer Panel")
if not st.session_state['trainer_logged_in']:
    password = st.sidebar.text_input("Trainer Password", type="password")
    if st.sidebar.button("Login"):
        if password == TRAINER_PASSWORD:
            st.session_state['trainer_logged_in'] = True
            st.rerun() # Rerun to update view after login
        else:
            st.sidebar.error("Incorrect password")
else:
    # --- Trainer Logged In View ---
    st.sidebar.success("Trainer Logged In")

    # Client Management
    st.sidebar.subheader("Client Management")
    new_client_name = st.sidebar.text_input("New Client Name")
    if st.sidebar.button("Add Client"):
        if new_client_name:
            generated_code = add_client(new_client_name)
            if generated_code:
                st.sidebar.success(f"Client '{new_client_name}' added!")
                st.sidebar.info(f"Unique Code: {generated_code}")
            else:
                st.sidebar.error("Failed to add client (code collision?). Try again.")
        else:
            st.sidebar.warning("Please enter a client name.")

    # View Clients
    st.sidebar.subheader("Existing Clients")
    clients_list = get_all_clients() # Gets list of (id, name, code)
    if clients_list:
        # Display only name and code in sidebar table
        clients_display_list = [(name, code) for cid, name, code in clients_list]
        clients_df = pd.DataFrame(clients_display_list, columns=['Name', 'Unique Code'])
        st.sidebar.dataframe(clients_df, hide_index=True)
    else:
        st.sidebar.write("No clients added yet.")

    # Client Data Selection
    st.sidebar.subheader("View Client Data")
    # Create {name: client_id} mapping from the fetched list
    client_name_id_dict = {name: cid for cid, name, code in clients_list}

    client_names_list = [name for cid, name, code in clients_list] # Get just names for selectbox
    selected_client_name = st.sidebar.selectbox("Select Client", options=client_names_list, index=None, placeholder="Choose a client...")

    if selected_client_name:
        st.session_state['selected_client_id_trainer'] = client_name_id_dict[selected_client_name]
    else:
        st.session_state['selected_client_id_trainer'] = None

    # Logout Button
    if st.sidebar.button("Trainer Logout"):
        st.session_state['trainer_logged_in'] = False
        st.session_state['selected_client_id_trainer'] = None # Clear selection on logout
        st.rerun()


# --- Main Area Logic ---

if st.session_state['trainer_logged_in']:
    # --- Trainer Dashboard View ---
    st.title("Trainer Dashboard")

    if st.session_state['selected_client_id_trainer']:
        client_id = st.session_state['selected_client_id_trainer']
        # Find client name from ID for display
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM Clients WHERE client_id = ?", (client_id,))
        client_info = cursor.fetchone()
        conn.close()
        client_name = client_info[0] if client_info else "Unknown Client"

        st.header(f"Viewing Data for: {client_name}")

        responses_df = get_responses_by_client(client_id)

        if not responses_df.empty:
            st.subheader("Response History (Table)")
            st.dataframe(responses_df)

            st.subheader("Response Trends (Charts)")
            # Ensure numeric columns for plotting
            numeric_cols = ['Fatigue', 'Sleep Quality', 'Sleep Hours', 'Soreness', 'Stress']
            for col in numeric_cols:
                 responses_df[col] = pd.to_numeric(responses_df[col], errors='coerce')

            # Plot each metric - Higher score is worse (except maybe sleep hours depending on interpretation)
            st.line_chart(responses_df['Fatigue'], use_container_width=True)
            st.caption("Fatigue (1=No fatigue, 7=Exhausted)")

            st.line_chart(responses_df['Sleep Quality'], use_container_width=True)
            st.caption("Sleep Quality (1=Outstanding, 7=Horrible)")

            st.line_chart(responses_df['Sleep Hours'], use_container_width=True)
            st.caption("Sleep Hours (1=10+, 7=5 or less)")

            st.line_chart(responses_df['Soreness'], use_container_width=True)
            st.caption("Muscle Soreness (1=No soreness, 7=Extremely sore)")

            st.line_chart(responses_df['Stress'], use_container_width=True)
            st.caption("Psychological Stress (1=Feeling great, 7=Very Stressed)")

        else:
            st.info("No responses recorded for this client yet.")
    else:
        st.info("Select a client from the sidebar to view their data.")

elif not st.session_state['logged_in']:
    # --- Client Login View ---
    st.title("Client Questionnaire Login")
    st.write("Please enter the unique code provided by your trainer.")

    client_code = st.text_input("Your Unique Code", key="client_code_input")

    if st.button("Login", key="client_login_button"):
        if client_code:
            client_info = get_client_by_code(client_code.upper()) # Ensure code is checked in uppercase
            if client_info:
                st.session_state['logged_in'] = True
                st.session_state['client_id'] = client_info[0]
                st.session_state['client_name'] = client_info[1]
                st.rerun() # Rerun to show questionnaire
            else:
                st.error("Invalid code. Please check the code and try again.")
        else:
            st.warning("Please enter your unique code.")

else:
    # --- Client Questionnaire View ---
    st.title(f"Welcome, {st.session_state['client_name']}!")

    # --- NEW: Display total submissions count ---
    total_submissions = count_responses_by_client(st.session_state['client_id'])
    st.metric(label="Total Check-ins", value=total_submissions)
    st.divider() # Add a visual separator

    st.subheader("Please answer the following questions based on how you feel *right now*.")
    st.caption("Scale: 1 = Positive/Good, 7 = Negative/Bad")

    # Define questions and scale labels accurately
    # Using lists of tuples: (display_label, score_value)
    scale_options = lambda descriptions: [(f"{i+1}: {desc}", i+1) for i, desc in enumerate(descriptions)]

    fatigue_desc = ["No fatigue", "Minimal fatigue", "Better than normal", "Normal", "Worse than normal", "Very fatigued", "Exhausted / major fatigue"]
    sleep_q_desc = ["Outstanding", "Very good", "Better than normal", "Normal", "Worse than normal", "Very Disrupted", "Horrible / no sleep"] # Using assumed labels for 4 & 6
    sleep_h_desc = ["10 + hours", "9-10 hours", "8-9 hours", "8 hours", "7-8 hours", "5-7 hours", "5 or less hours"]
    soreness_desc = ["No soreness", "Very little soreness", "Better than normal", "Normal", "Worse than normal", "Very sore/tight", "Extremely sore/tight"]
    stress_desc = ["Feeling great / very relaxed", "Feeling good - relaxed", "Better than normal", "Normal", "Worse than normal", "Stressed", "Very Stressed"]

    with st.form(key="questionnaire_form"):
        st.markdown("**1. How fatigued are you?**")
        # Use index=3 to default selection to '4: Normal' (optional, provides a default)
        fatigue_resp = st.radio("", options=scale_options(fatigue_desc), format_func=lambda x: x[0], horizontal=True, key="q_fatigue", index=3)

        st.markdown("**2. How was your sleep last night?**")
        sleep_q_resp = st.radio("", options=scale_options(sleep_q_desc), format_func=lambda x: x[0], horizontal=True, key="q_sleep_q", index=3)

        st.markdown("**3. How many hours did you sleep last night?**")
        sleep_h_resp = st.radio("", options=scale_options(sleep_h_desc), format_func=lambda x: x[0], horizontal=True, key="q_sleep_h", index=3)

        st.markdown("**4. Please rate your level of muscle soreness.**")
        soreness_resp = st.radio("", options=scale_options(soreness_desc), format_func=lambda x: x[0], horizontal=True, key="q_soreness", index=3)

        st.markdown("**5. How are you feeling psychologically (Mentally)?**")
        stress_resp = st.radio("", options=scale_options(stress_desc), format_func=lambda x: x[0], horizontal=True, key="q_stress", index=3)

        # --- UPDATED: Button label ---
        submitted = st.form_submit_button("Check in")

        if submitted:
            # Extract the score value (the second element of the tuple)
            fatigue_score = fatigue_resp[1]
            sleep_q_score = sleep_q_resp[1]
            sleep_h_score = sleep_h_resp[1]
            soreness_score = soreness_resp[1]
            stress_score = stress_resp[1]

            # Add response to database
            add_response(
                st.session_state['client_id'],
                fatigue_score,
                sleep_q_score,
                sleep_h_score,
                soreness_score,
                stress_score
            )
            st.success("Thank you! Your responses have been recorded.")
            # Optionally clear state or provide logout after submission
            # Consider adding a small delay or button before rerun if clearing state
            # Using st.rerun() here would immediately refresh, potentially before user sees success msg
            # For now, let the user see the success message and the updated count on next interaction or manual refresh/logout.

    # Logout button for client
    if st.button("Logout", key="client_logout_button"):
        st.session_state['logged_in'] = False
        st.session_state['client_id'] = None
        st.session_state['client_name'] = None
        st.rerun()
