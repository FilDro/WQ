# Stress & Readiness App - Supabase Version
# Connects to a Supabase (PostgreSQL) backend for data persistence.

import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timezone
from supabase import create_client, Client # Import Supabase client

# --- Supabase Initialization ---
# Initialize Supabase client only once using caching for efficiency
@st.cache_resource
def init_supabase_client():
    """Initializes and returns the Supabase client."""
    try:
        # Fetch credentials from Streamlit secrets
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"❌ Missing Supabase credential in Streamlit Secrets: {e}")
        st.stop() # Stop execution if secrets are missing
    except Exception as e:
        st.error(f"❌ Error initializing Supabase client: {e}")
        st.stop()

supabase: Client = init_supabase_client()

# --- Database Operations (Supabase Version) ---

def add_client(name: str) -> str | None:
    """Adds a new client to Supabase, generates a unique code, and returns the code."""
    unique_code = uuid.uuid4().hex[:8].upper()
    try:
        response = supabase.table('Clients').insert({
            "name": name,
            "unique_code": unique_code
        }).execute()

        # Check response structure (API V1: response.data)
        if response.data and len(response.data) > 0:
             st.success(f"Client '{name}' added successfully.") # More user feedback
             return unique_code
        else:
             # Log potential errors returned by Supabase even if no exception
             st.error(f"Supabase insert failed (add_client). Response: {response}")
             return None
    except Exception as e:
        # Catch specific Supabase/PostgREST errors if needed, otherwise general exception
        st.error(f"❌ Error adding client to Supabase: {e}")
        return None

def get_client_by_code(code: str) -> tuple[int, str] | None:
    """Retrieves client details (id, name) from Supabase based on the unique code."""
    try:
        response = supabase.table('Clients').select(
            'client_id, name'
            ).eq(
                'unique_code', code.upper() # Ensure code is uppercase for matching
            ).limit(1).execute()

        if response.data and len(response.data) > 0:
            client = response.data[0] # Get the first dictionary in the list
            return client['client_id'], client['name'] # Return tuple (id, name)
        else:
            return None # Not found
    except Exception as e:
        st.error(f"❌ Error fetching client by code from Supabase: {e}")
        return None

def get_all_clients() -> list[tuple[int, str, str]]:
    """Retrieves all client ids, names and unique codes from Supabase."""
    try:
        response = supabase.table('Clients').select(
            'client_id, name, unique_code'
            ).order('name').execute() # Order by name

        if response.data and len(response.data) > 0:
            # Return list of (id, name, code) tuples
            return [(client['client_id'], client['name'], client['unique_code']) for client in response.data]
        else:
            return [] # Return empty list if no clients
    except Exception as e:
        st.error(f"❌ Error fetching all clients from Supabase: {e}")
        return []

def add_response(client_id: int, fatigue: int, sleep_quality: int, sleep_hours: int, soreness: int, stress: int) -> bool:
    """Adds a new questionnaire response to Supabase. Returns True on success."""
    try:
        # Supabase expects ISO 8601 format timestamp with timezone
        timestamp_now = datetime.now(timezone.utc).isoformat()
        response = supabase.table('Responses').insert({
            "client_id": client_id,
            "timestamp": timestamp_now, # Use the formatted timestamp
            "fatigue": fatigue,
            "sleep_quality": sleep_quality,
            "sleep_hours": sleep_hours,
            "soreness": soreness,
            "stress": stress
        }).execute()

        if response.data and len(response.data) > 0:
            return True # Indicate success
        else:
            st.error(f"Supabase insert failed (add_response). Response: {response}")
            return False
    except Exception as e:
        st.error(f"❌ Error adding response to Supabase: {e}")
        return False

def get_responses_by_client(client_id: int) -> pd.DataFrame:
    """Retrieves all responses for a specific client from Supabase, ordered by timestamp."""
    try:
        # Ensure client_id is an integer before querying
        client_id_int = int(client_id)
        response = supabase.table('Responses').select(
            'timestamp, fatigue, sleep_quality, sleep_hours, soreness, stress' # Select specific columns
        ).eq('client_id', client_id_int).order('timestamp', desc=False).execute() # Order ascending

        if response.data and len(response.data) > 0:
            responses = response.data
            df = pd.DataFrame(responses)
            # Rename 'timestamp' column before converting
            df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp']) # Convert to datetime objects
            df.set_index('Timestamp', inplace=True)
            # Ensure correct column names for consistency
            df.columns = ['Fatigue', 'Sleep Quality', 'Sleep Hours', 'Soreness', 'Stress']
            return df
        else:
            return pd.DataFrame() # Return empty DataFrame if no responses
    except (ValueError, TypeError):
        # This case should ideally be prevented by UI logic, but good to have
        st.error("❌ Invalid Client ID format for fetching responses.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching responses from Supabase: {e}")
        return pd.DataFrame()


def count_responses_by_client(client_id: int) -> int:
    """Counts the total number of responses submitted by a specific client in Supabase."""
    try:
         # Ensure client_id is an integer before querying
        client_id_int = int(client_id)
        # Use head=True and count='exact' for efficient counting
        response = supabase.table('Responses').select(
            'response_id', # Select any non-null column
            count='exact', # Specify exact count needed
            head=True # Fetch only count, not data
        ).eq('client_id', client_id_int).execute()

        # The count is returned directly in the response object
        return response.count if response.count is not None else 0
    except (ValueError, TypeError):
        st.error("❌ Invalid Client ID format for counting responses.")
        return 0
    except Exception as e:
        st.error(f"❌ Error counting responses from Supabase: {e}")
        return 0

# --- Streamlit App UI Code ---

# Initialize session state variables
def init_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        'logged_in': False,
        'client_id': None,
        'client_name': None,
        'trainer_logged_in': False,
        'selected_client_id_trainer': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Trainer Sidebar / Access ---
st.sidebar.title("Trainer Panel")
if not st.session_state.trainer_logged_in:
    password = st.sidebar.text_input("Trainer Password", type="password", key="trainer_pw_input")
    if st.sidebar.button("Login", key="trainer_login_btn"):
        try:
            # Read password from secrets
            trainer_pw_secret = st.secrets["TRAINER_PASSWORD"]
            if password == trainer_pw_secret:
                st.session_state.trainer_logged_in = True
                st.rerun() # Rerun to update view after login
            else:
                st.sidebar.error("Incorrect password")
        except KeyError:
             st.sidebar.error("❗️ Trainer password not found in secrets.")
        except Exception as e:
             st.sidebar.error(f"Login error: {e}")

else:
    # --- Trainer Logged In View ---
    st.sidebar.success("Trainer Logged In")

    # Client Management
    st.sidebar.subheader("Client Management")
    with st.sidebar.form("add_client_form"):
        new_client_name = st.text_input("New Client Name")
        submitted = st.form_submit_button("Add Client")
        if submitted:
            if new_client_name:
                generated_code = add_client(new_client_name) # Uses Supabase function
                if generated_code:
                    # Success message handled within add_client
                    st.sidebar.info(f"Unique Code: {generated_code}")
                # Error handling is now inside add_client
            else:
                st.warning("Please enter a client name.")

    # View Clients
    st.sidebar.subheader("Existing Clients")
    clients_list = get_all_clients() # Uses Supabase function
    if clients_list:
        # Display only name and code in sidebar table
        clients_display_list = [(name, code) for cid, name, code in clients_list]
        clients_df = pd.DataFrame(clients_display_list, columns=['Name', 'Unique Code'])
        st.sidebar.dataframe(clients_df, hide_index=True, use_container_width=True)
    else:
        st.sidebar.write("No clients added yet.")

    # Client Data Selection
    st.sidebar.subheader("View Client Data")
    # Create {name: client_id} mapping from the fetched list
    client_name_id_dict = {name: cid for cid, name, code in clients_list}

    client_names_list = [""] + [name for cid, name, code in clients_list] # Add empty option for placeholder
    selected_client_name = st.sidebar.selectbox(
        "Select Client",
        options=client_names_list,
        index=0, # Default to empty selection
        placeholder="Choose a client...",
        key="client_selector"
        )

    if selected_client_name and selected_client_name in client_name_id_dict:
        st.session_state.selected_client_id_trainer = client_name_id_dict[selected_client_name]
    else:
        st.session_state.selected_client_id_trainer = None

    # Logout Button
    if st.sidebar.button("Trainer Logout", key="trainer_logout_btn"):
        # Clear relevant session state on logout
        st.session_state.trainer_logged_in = False
        st.session_state.selected_client_id_trainer = None
        st.rerun()


# --- Main Area Logic ---

if st.session_state.trainer_logged_in:
    # --- Trainer Dashboard View ---
    st.title("Trainer Dashboard")

    if st.session_state.selected_client_id_trainer:
        client_id = st.session_state.selected_client_id_trainer
        # Find client name from ID for display (Could optimize by fetching name along with ID earlier)
        client_name = "Unknown Client" # Default
        # Find the name from the dictionary we already built
        for name, cid in client_name_id_dict.items():
            if cid == client_id:
                client_name = name
                break

        st.header(f"Viewing Data for: {client_name}")

        responses_df = get_responses_by_client(client_id) # Uses Supabase function

        if not responses_df.empty:
            st.subheader("Response History (Table)")
            # Display DataFrame, ensuring Timestamp index is handled correctly by st.dataframe
            # Convert Timestamp index to a column for better display control
            display_df = responses_df.reset_index()
            # Format timestamp for readability
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_df, use_container_width=True, hide_index=True)


            st.subheader("Response Trends (Charts)")
            # Ensure numeric columns for plotting
            numeric_cols = ['Fatigue', 'Sleep Quality', 'Sleep Hours', 'Soreness', 'Stress']
            plot_df = responses_df.copy() # Work on a copy for plotting
            # Convert columns to numeric, coercing errors
            for col in numeric_cols:
                 if col in plot_df.columns: # Check if column exists
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                 else:
                     st.warning(f"Column '{col}' not found for plotting.") # Warn if data is missing

            # Plot each metric - Higher score is worse (except maybe sleep hours depending on interpretation)
            # Check if columns exist before plotting to avoid errors
            if 'Fatigue' in plot_df.columns:
                st.line_chart(plot_df['Fatigue'], use_container_width=True)
                st.caption("Fatigue (1=No fatigue, 7=Exhausted)")
            if 'Sleep Quality' in plot_df.columns:
                st.line_chart(plot_df['Sleep Quality'], use_container_width=True)
                st.caption("Sleep Quality (1=Outstanding, 7=Horrible)")
            if 'Sleep Hours' in plot_df.columns:
                st.line_chart(plot_df['Sleep Hours'], use_container_width=True)
                st.caption("Sleep Hours (1=10+, 7=5 or less)")
            if 'Soreness' in plot_df.columns:
                st.line_chart(plot_df['Soreness'], use_container_width=True)
                st.caption("Muscle Soreness (1=No soreness, 7=Extremely sore)")
            if 'Stress' in plot_df.columns:
                st.line_chart(plot_df['Stress'], use_container_width=True)
                st.caption("Psychological Stress (1=Feeling great, 7=Very Stressed)")

        else:
            st.info("No responses recorded for this client yet.")
    else:
        st.info("Select a client from the sidebar to view their data.")

elif not st.session_state.logged_in:
    # --- Client Login View ---
    st.title("Client Questionnaire Login")
    st.write("Please enter the unique code provided by your trainer.")

    client_code = st.text_input("Your Unique Code", key="client_code_input", value="") # Ensure value is controlled

    if st.button("Login", key="client_login_button"):
        if client_code:
            client_info = get_client_by_code(client_code) # Uses Supabase function
            if client_info:
                st.session_state.logged_in = True
                st.session_state.client_id = client_info[0] # ID is first element
                st.session_state.client_name = client_info[1] # Name is second element
                st.rerun() # Rerun to show questionnaire
            else:
                st.error("Invalid code. Please check the code and try again.")
        else:
            st.warning("Please enter your unique code.")

else:
    # --- Client Questionnaire View ---
    st.title(f"Welcome, {st.session_state.client_name}!")

    # Display total submissions count
    total_submissions = count_responses_by_client(st.session_state.client_id) # Uses Supabase function
    st.metric(label="Total Check-ins", value=total_submissions)
    st.divider()

    st.subheader("Please answer the following questions based on how you feel *right now*.")
    st.caption("Scale: 1 = Positive/Good, 7 = Negative/Bad")

    # Define questions and scale labels accurately
    scale_options = lambda descriptions: [(f"{i+1}: {desc}", i+1) for i, desc in enumerate(descriptions)]
    fatigue_desc = ["No fatigue", "Minimal fatigue", "Better than normal", "Normal", "Worse than normal", "Very fatigued", "Exhausted / major fatigue"]
    sleep_q_desc = ["Outstanding", "Very good", "Better than normal", "Normal", "Worse than normal", "Very Disrupted", "Horrible / no sleep"]
    sleep_h_desc = ["10 + hours", "9-10 hours", "8-9 hours", "8 hours", "7-8 hours", "5-7 hours", "5 or less hours"]
    soreness_desc = ["No soreness", "Very little soreness", "Better than normal", "Normal", "Worse than normal", "Very sore/tight", "Extremely sore/tight"]
    stress_desc = ["Feeling great / very relaxed", "Feeling good - relaxed", "Better than normal", "Normal", "Worse than normal", "Stressed", "Very Stressed"]

    # Use unique keys for radio buttons within the form
    with st.form(key="questionnaire_form"):
        st.markdown("**1. How fatigued are you?**")
        fatigue_resp = st.radio("Fatigue Scale", options=scale_options(fatigue_desc), format_func=lambda x: x[0], horizontal=True, key="q_fatigue_radio", label_visibility="collapsed", index=3)
        st.markdown("**2. How was your sleep last night?**")
        sleep_q_resp = st.radio("Sleep Quality Scale", options=scale_options(sleep_q_desc), format_func=lambda x: x[0], horizontal=True, key="q_sleep_q_radio", label_visibility="collapsed", index=3)
        st.markdown("**3. How many hours did you sleep last night?**")
        sleep_h_resp = st.radio("Sleep Hours Scale", options=scale_options(sleep_h_desc), format_func=lambda x: x[0], horizontal=True, key="q_sleep_h_radio", label_visibility="collapsed", index=3)
        st.markdown("**4. Please rate your level of muscle soreness.**")
        soreness_resp = st.radio("Soreness Scale", options=scale_options(soreness_desc), format_func=lambda x: x[0], horizontal=True, key="q_soreness_radio", label_visibility="collapsed", index=3)
        st.markdown("**5. How are you feeling psychologically (Mentally)?**")
        stress_resp = st.radio("Stress Scale", options=scale_options(stress_desc), format_func=lambda x: x[0], horizontal=True, key="q_stress_radio", label_visibility="collapsed", index=3)

        submitted = st.form_submit_button("Check in")

        if submitted:
            # Extract the score value (the second element of the tuple)
            fatigue_score = fatigue_resp[1]
            sleep_q_score = sleep_q_resp[1]
            sleep_h_score = sleep_h_resp[1]
            soreness_score = soreness_resp[1]
            stress_score = stress_resp[1]

            # Add response to Supabase database
            success = add_response(
                st.session_state.client_id,
                fatigue_score,
                sleep_q_score,
                sleep_h_score,
                soreness_score,
                stress_score
            )
            if success:
                st.success("✅ Thank you! Your responses have been recorded.")
                # Optionally clear form or add delay before potential rerun
            # Error message handled within add_response

    # Logout button for client
    if st.button("Logout", key="client_logout_button"):
        # Clear client-specific session state
        st.session_state.logged_in = False
        st.session_state.client_id = None
        st.session_state.client_name = None
        st.rerun()


