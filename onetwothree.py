# Stress & Readiness App - Enhanced Polish Version
# Connects to a Supabase (PostgreSQL) backend for data persistence.

import streamlit as st
import pandas as pd
import uuid
import io
import base64
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client

# Debug mode - set to True to bypass Supabase requirement
DEBUG_MODE = False

# Set page configuration
st.set_page_config(
    page_title="Aplikacja Wellness i Trening",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* Card styling */
    .stcard {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    /* Headers */
    h1 {
        color: #2C3E50;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #2C3E50;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #2C3E50;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }

    /* Form elements */
    .stRadio > label {
        font-weight: 500;
        color: #2C3E50;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 0.5rem;
    }
    .stRadio label {
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        border: 1px solid #ddd;
        background-color: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stRadio label:hover {
        background-color: #f8f9fa;
    }

    /* Buttons */
    .stButton button {
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Metric tiles */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #4a86e8;
    }
    .metric-card.warning {
        border-top: 4px solid #FFC107;
    }
    .metric-card.danger {
        border-top: 4px solid #F44336;
    }
    .metric-card.success {
        border-top: 4px solid #4CAF50;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
    }
    .metric-trend {
        font-size: 0.8rem;
        color: #888;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #4a86e8;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Scale legend */
    .scale-legend {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .scale-legend-item {
        text-align: center;
        font-size: 0.7rem;
        color: #555;
    }

    /* Login screen */
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Download links */
    .download-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4a86e8;
        color: white;
        text-decoration: none;
        border-radius: 0.3rem;
        text-align: center;
        margin-top: 0.5rem;
        transition: all 0.2s;
    }
    .download-link:hover {
        background-color: #3a76d8;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Supabase Initialization ---
@st.cache_resource
def init_supabase_client():
    """Initializes and returns the Supabase client."""
    if DEBUG_MODE:
        st.warning("⚠️ Running in DEBUG mode - Supabase connection disabled")
        return None

    try:
        # Fetch credentials from Streamlit secrets
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        client = create_client(supabase_url, supabase_key)
        return client
    except KeyError as e:
        st.error(f"❌ Brak poświadczeń Supabase w Streamlit Secrets: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji klienta Supabase: {e}")
        return None


# Initialize Supabase client - now with better error handling
supabase = init_supabase_client()


# --- Database Operations (Supabase Version) ---

def add_client(name: str) -> str | None:
    """Adds a new client to Supabase, generates a unique code, and returns the code."""
    if not supabase:
        if DEBUG_MODE:
            return "DEBUG_CODE"
        return None

    unique_code = uuid.uuid4().hex[:8].upper()
    try:
        response = supabase.table('Clients').insert({
            "name": name,
            "unique_code": unique_code
        }).execute()

        if response.data and len(response.data) > 0:
            st.success(f"Klient '{name}' dodany pomyślnie.")
            return unique_code
        else:
            st.error(f"Błąd dodawania klienta (add_client). Odpowiedź: {response}")
            return None
    except Exception as e:
        st.error(f"❌ Błąd dodawania klienta do Supabase: {e}")
        return None


def get_client_by_code(code: str) -> tuple[int, str] | None:
    """Retrieves client details (id, name) from Supabase based on the unique code."""
    if not supabase:
        if DEBUG_MODE:
            return (1, "Debug Client")
        return None

    try:
        response = supabase.table('Clients').select(
            'client_id, name'
        ).eq(
            'unique_code', code.upper()  # Ensure code is uppercase for matching
        ).limit(1).execute()

        if response.data and len(response.data) > 0:
            client = response.data[0]  # Get the first dictionary in the list
            return client['client_id'], client['name']  # Return tuple (id, name)
        else:
            return None  # Not found
    except Exception as e:
        st.error(f"❌ Błąd pobierania klienta z kodu z Supabase: {e}")
        return None


def get_all_clients() -> list[tuple[int, str, str]]:
    """Retrieves all client ids, names and unique codes from Supabase."""
    if not supabase:
        if DEBUG_MODE:
            return [(1, "Debug Client", "DEBUG123")]
        return []

    try:
        response = supabase.table('Clients').select(
            'client_id, name, unique_code'
        ).order('name').execute()  # Order by name

        if response.data and len(response.data) > 0:
            # Return list of (id, name, code) tuples
            return [(client['client_id'], client['name'], client['unique_code']) for client in response.data]
        else:
            return []  # Return empty list if no clients
    except Exception as e:
        st.error(f"❌ Błąd pobierania wszystkich klientów z Supabase: {e}")
        return []


def add_response(client_id: int, fatigue: int, sleep_quality: int, sleep_hours: int,
                 soreness: int, stress: int, training_difficulty: int, note: str = "") -> bool:
    """Adds a new questionnaire response to Supabase. Returns True on success."""
    if not supabase:
        if DEBUG_MODE:
            return True
        return False

    try:
        # Supabase expects ISO 8601 format timestamp with timezone
        timestamp_now = datetime.now(timezone.utc).isoformat()
        response = supabase.table('Responses').insert({
            "client_id": client_id,
            "timestamp": timestamp_now,  # Use the formatted timestamp
            "fatigue": fatigue,
            "sleep_quality": sleep_quality,
            "sleep_hours": sleep_hours,
            "soreness": soreness,
            "stress": stress,
            "training_difficulty": training_difficulty,
            "note": note  # New field for notes
        }).execute()

        if response.data and len(response.data) > 0:
            return True  # Indicate success
        else:
            st.error(f"Błąd dodawania odpowiedzi (add_response). Odpowiedź: {response}")
            return False
    except Exception as e:
        st.error(f"❌ Błąd dodawania odpowiedzi do Supabase: {e}")
        return False


def get_responses_by_client(client_id: int, days: int = None) -> pd.DataFrame:
    """Retrieves responses for a specific client, optionally filtered by date range."""
    if not supabase:
        if DEBUG_MODE:
            # Return a sample DataFrame for debugging
            dates = pd.date_range(end=datetime.now(), periods=7)
            df = pd.DataFrame({
                'Zmęczenie': [3, 4, 3, 5, 2, 3, 4],
                'Jakość Snu': [2, 3, 4, 3, 5, 4, 3],
                'Godziny Snu': [5, 6, 7, 5, 8, 7, 6],
                'Ból Mięśni': [4, 5, 3, 4, 2, 3, 4],
                'Stres': [3, 4, 5, 3, 2, 3, 4],
                'Trudność Treningu': [4, 5, 3, 4, 3, 4, 5],
                'Notatka': ['', '', '', '', '', '', '']
            }, index=dates)
            return df
        return pd.DataFrame()

    try:
        # Ensure client_id is an integer before querying
        client_id_int = int(client_id)
        query = supabase.table('Responses').select(
            'timestamp, fatigue, sleep_quality, sleep_hours, soreness, stress, training_difficulty, note'
        ).eq('client_id', client_id_int)

        # Apply date filter if specified
        if days:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            query = query.gte('timestamp', cutoff_date)

        response = query.order('timestamp', desc=False).execute()

        if response.data and len(response.data) > 0:
            responses = response.data
            df = pd.DataFrame(responses)
            # Rename 'timestamp' column before converting
            df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert to datetime objects
            df.set_index('Timestamp', inplace=True)
            # Ensure correct column names for consistency
            df.rename(columns={
                'fatigue': 'Zmęczenie',
                'sleep_quality': 'Jakość Snu',
                'sleep_hours': 'Godziny Snu',
                'soreness': 'Ból Mięśni',
                'stress': 'Stres',
                'training_difficulty': 'Trudność Treningu',
                'note': 'Notatka'
            }, inplace=True)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no responses
    except (ValueError, TypeError):
        st.error("❌ Nieprawidłowy format ID klienta do pobierania odpowiedzi.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Błąd pobierania odpowiedzi z Supabase: {e}")
        return pd.DataFrame()


def count_responses_by_client(client_id: int) -> int:
    """Counts the total number of responses submitted by a specific client in Supabase."""
    if not supabase:
        if DEBUG_MODE:
            return 7  # Sample count for debugging
        return 0

    try:
        client_id_int = int(client_id)
        response = supabase.table('Responses').select(
            'response_id',
            count='exact',
            head=True
        ).eq('client_id', client_id_int).execute()

        return response.count if response.count is not None else 0
    except (ValueError, TypeError):
        st.error("❌ Nieprawidłowy format ID klienta do liczenia odpowiedzi.")
        return 0
    except Exception as e:
        st.error(f"❌ Błąd liczenia odpowiedzi z Supabase: {e}")
        return 0


def add_trainer_note(client_id: int, note: str, date: datetime = None) -> bool:
    """Adds a trainer note for a specific client."""
    if not supabase:
        if DEBUG_MODE:
            return True
        return False

    try:
        if date is None:
            date = datetime.now(timezone.utc)

        timestamp = date.isoformat()
        response = supabase.table('TrainerNotes').insert({
            "client_id": client_id,
            "timestamp": timestamp,
            "note": note
        }).execute()

        if response.data and len(response.data) > 0:
            return True
        else:
            st.error(f"Błąd dodawania notatki trenera. Odpowiedź: {response}")
            return False
    except Exception as e:
        st.error(f"❌ Błąd dodawania notatki trenera: {e}")
        return False


def get_trainer_notes(client_id: int) -> pd.DataFrame:
    """Retrieves all trainer notes for a specific client."""
    if not supabase:
        if DEBUG_MODE:
            # Sample trainer notes for debugging
            return pd.DataFrame({
                'Data': ['2025-04-19 15:30', '2025-04-15 09:45'],
                'Notatka': ['Dobry postęp w treningu siłowym', 'Zalecana zmiana planu treningowego']
            })
        return pd.DataFrame()

    try:
        client_id_int = int(client_id)
        response = supabase.table('TrainerNotes').select(
            'timestamp, note'
        ).eq('client_id', client_id_int).order('timestamp', desc=True).execute()

        if response.data and len(response.data) > 0:
            notes = response.data
            df = pd.DataFrame(notes)
            df.rename(columns={'timestamp': 'Data'}, inplace=True)
            df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m-%d %H:%M')
            df.rename(columns={'note': 'Notatka'}, inplace=True)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Błąd pobierania notatek trenera: {e}")
        return pd.DataFrame()


# --- Helper Functions ---

def calculate_wellness_score(df: pd.DataFrame) -> float:
    """Calculates an overall wellness score based on all metrics."""
    if df.empty:
        return 0

    # Get the most recent entry
    latest = df.iloc[-1]

    # Reverse scales where needed so higher is always better
    # Original scale: 1 is good, 7 is bad
    reversed_metrics = {
        'Zmęczenie': 8 - latest['Zmęczenie'],
        'Jakość Snu': latest['Jakość Snu'],  # Already in correct scale
        'Godziny Snu': latest['Godziny Snu'],  # Already in correct scale
        'Ból Mięśni': 8 - latest['Ból Mięśni'],
        'Stres': 8 - latest['Stres'],
        'Trudność Treningu': 8 - latest['Trudność Treningu']
    }

    # Calculate average (now 1 is bad, 7 is good)
    avg_score = sum(reversed_metrics.values()) / len(reversed_metrics)

    # Convert to percentage (0-100 scale)
    return round((avg_score - 1) * 100 / 6, 1)


def get_color_for_value(value: float, is_reversed: bool = True) -> str:
    """Returns a color based on the value (1-7 scale) and whether higher values are good."""
    if is_reversed:  # If 1 is good, 7 is bad (default for our app)
        if value <= 2.5:
            return "#4CAF50"  # Green
        elif value <= 4.5:
            return "#FFC107"  # Yellow
        else:
            return "#F44336"  # Red
    else:  # If 1 is bad, 7 is good
        if value >= 5.5:
            return "#4CAF50"  # Green
        elif value >= 3.5:
            return "#FFC107"  # Yellow
        else:
            return "#F44336"  # Red


def create_gauge_chart(value: float, title: str, is_reversed: bool = True):
    """Creates a gauge chart for a metric."""
    # Convert to 0-100 scale
    if is_reversed:  # If 1 is good, 7 is bad
        gauge_value = ((value - 1) / 6) * 100
    else:  # If 1 is bad, 7 is good
        gauge_value = ((7 - value) / 6) * 100

    color = get_color_for_value(value, is_reversed)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [1, 7], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 3], 'color': '#c5e8c5' if is_reversed else '#ffcccb'},
                {'range': [3, 5], 'color': '#ffe4b2'},
                {'range': [5, 7], 'color': '#ffcccb' if is_reversed else '#c5e8c5'}
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="white",
        font={'color': "#2C3E50"}
    )

    return fig


def create_trend_chart(df: pd.DataFrame, metrics: list, days: int = 14):
    """Creates a trend chart for selected metrics."""
    if df.empty:
        return None

    # Filter to the selected number of days
    if len(df) > days:
        df = df.iloc[-days:]

    # Create figure
    fig = go.Figure()

    colors = {
        'Zmęczenie': '#4a86e8',
        'Jakość Snu': '#FFC107',
        'Godziny Snu': '#9C27B0',
        'Ból Mięśni': '#F44336',
        'Stres': '#FF9800',
        'Trudność Treningu': '#2196F3'
    }

    # Add traces for each metric
    for metric in metrics:
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors.get(metric, '#333'), width=2),
                marker=dict(size=8)
            ))

    # Update layout
    fig.update_layout(
        title=f"Trendy ostatnich {len(df)} dni",
        xaxis_title="Data",
        yaxis_title="Wartość (1-7)",
        yaxis=dict(range=[0.5, 7.5]),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        font={'color': "#2C3E50"}
    )

    return fig


def create_radar_chart(df: pd.DataFrame):
    """Creates a radar chart comparing the latest values with previous average."""
    if df.empty or len(df) < 2:
        return None

    metrics = ['Zmęczenie', 'Jakość Snu', 'Godziny Snu', 'Ból Mięśni', 'Stres', 'Trudność Treningu']

    # Get latest values and previous average
    latest = df.iloc[-1]
    previous_avg = df.iloc[:-1].mean()

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[latest[m] for m in metrics],
        theta=metrics,
        fill='toself',
        name='Ostatni pomiar',
        line_color='#4a86e8'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[previous_avg[m] for m in metrics],
        theta=metrics,
        fill='toself',
        name='Średnia poprzednich',
        line_color='rgba(255, 193, 7, 0.8)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 7]
            )
        ),
        title="Porównanie z poprzednimi pomiarami",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        font={'color': "#2C3E50"}
    )

    return fig


def get_download_link(df, filename, text):
    """Generates a download link for a DataFrame."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href


def get_pdf_download_link(client_name, df, text):
    """Generates a PDF report for download."""
    buffer = io.BytesIO()

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(len(df.columns) - 1 if 'Notatka' in df.columns else len(df.columns),
                            1, figsize=(11, 2 * len(df.columns)))
    fig.suptitle(f'Raport dla: {client_name}', fontsize=16)

    # Plot each metric
    plot_cols = [col for col in df.columns if col != 'Notatka']
    for i, col in enumerate(plot_cols):
        if len(plot_cols) > 1:
            ax = axs[i]
        else:
            ax = axs

        ax.plot(df.index, df[col], 'o-', color='#4a86e8')
        ax.set_title(col)
        ax.set_xlabel('Data')
        ax.set_ylabel('Wartość (1-7)')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save to buffer
    plt.savefig(buffer, format='pdf')
    buffer.seek(0)

    # Generate download link
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="raport_{client_name}.pdf" class="download-link">{text}</a>'
    return href


# --- Streamlit App UI Code ---

# Initialize session state variables
def init_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        'logged_in': False,
        'client_id': None,
        'client_name': None,
        'trainer_logged_in': False,
        'selected_client_id_trainer': None,
        'questionnaire_page': 1,
        'questionnaire_total_pages': 6,
        'remember_client_code': False,
        'saved_client_code': '',
        'filter_days': 14,
        'selected_metrics': ['Zmęczenie', 'Jakość Snu', 'Ból Mięśni'],
        'view_mode': 'line',  # line or radar
        'show_client_dashboard': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Main App Logic ---

def show_login_screen():
    """Displays the login screen for clients and trainers."""
    st.title("Aplikacja Wellness i Trening")

    # Create container with custom styling
    login_container = st.container()
    with login_container:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # Tabs for client and trainer login
        tab1, tab2 = st.tabs(["Logowanie Klienta", "Logowanie Trenera"])

        with tab1:
            st.subheader("Logowanie Klienta")

            # Show saved code if remembering
            default_code = st.session_state.saved_client_code if st.session_state.remember_client_code else ""
            client_code = st.text_input("Wprowadź swój unikalny kod:", value=default_code).strip().upper()

            remember = st.checkbox("Zapamiętaj mój kod", value=st.session_state.remember_client_code)

            if st.button("Zaloguj", key="client_login"):
                if client_code:
                    client_info = get_client_by_code(client_code)
                    if client_info:
                        client_id, client_name = client_info
                        st.session_state.client_id = client_id
                        st.session_state.client_name = client_name
                        st.session_state.logged_in = True
                        st.session_state.remember_client_code = remember
                        if remember:
                            st.session_state.saved_client_code = client_code
                        st.rerun()
                    else:
                        st.error("❌ Nieprawidłowy kod. Spróbuj ponownie lub skontaktuj się z trenerem.")
                else:
                    st.warning("⚠️ Wprowadź swój kod, aby się zalogować.")

        with tab2:
            st.subheader("Panel Trenera")
            trainer_password = st.text_input("Hasło trenera:", type="password")

            if st.button("Zaloguj jako Trener"):
                # Simple password check - in production, use proper authentication
                if trainer_password == "admin123":  # Replace with secure auth
                    st.session_state.trainer_logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Nieprawidłowe hasło trenera.")

        st.markdown('</div>', unsafe_allow_html=True)


def show_client_questionnaire():
    """Displays the wellness questionnaire for clients to fill out."""
    st.title(f"Witaj, {st.session_state.client_name}!")

    if 'questionnaire_page' not in st.session_state:
        st.session_state.questionnaire_page = 1

    total_pages = 6  # Total number of pages in the questionnaire

    # Page navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.session_state.questionnaire_page > 1:
            if st.button("⬅️ Wstecz"):
                st.session_state.questionnaire_page -= 1
                st.rerun()

    with col2:
        st.write(f"Strona {st.session_state.questionnaire_page} z {total_pages}")
        progress = st.session_state.questionnaire_page / total_pages
        st.progress(progress)

    with col3:
        if st.session_state.questionnaire_page < total_pages:
            if st.button("Dalej ➡️"):
                st.session_state.questionnaire_page += 1
                st.rerun()

    # Store responses
    if 'responses' not in st.session_state:
        st.session_state.responses = {
            'fatigue': 4,
            'sleep_quality': 4,
            'sleep_hours': 4,
            'soreness': 4,
            'stress': 4,
            'training_difficulty': 4,
            'note': ""
        }

    # Display current questionnaire page
    with st.form(key=f"page_{st.session_state.questionnaire_page}"):
        if st.session_state.questionnaire_page == 1:
            st.header("Poziom Zmęczenia")
            st.write("Jak zmęczony/a czujesz się dzisiaj?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: Wypoczęty/a</div>
                <div class="scale-legend-item">4: Średnie zmęczenie</div>
                <div class="scale-legend-item">7: Bardzo zmęczony/a</div>
            </div>
            """, unsafe_allow_html=True)

            fatigue = st.slider("", 1, 7, st.session_state.responses['fatigue'], key="fatigue_slider")
            st.session_state.responses['fatigue'] = fatigue

        elif st.session_state.questionnaire_page == 2:
            st.header("Jakość Snu")
            st.write("Jak oceniasz jakość swojego snu w zeszłą noc?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: Bardzo słaba</div>
                <div class="scale-legend-item">4: Przeciętna</div>
                <div class="scale-legend-item">7: Doskonała</div>
            </div>
            """, unsafe_allow_html=True)

            sleep_quality = st.slider("", 1, 7, st.session_state.responses['sleep_quality'], key="sleep_quality_slider")
            st.session_state.responses['sleep_quality'] = sleep_quality

            st.header("Ilość Snu")
            st.write("Ile godzin spałeś/aś w zeszłą noc?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: < 5 godzin</div>
                <div class="scale-legend-item">4: ~7 godzin</div>
                <div class="scale-legend-item">7: > 9 godzin</div>
            </div>
            """, unsafe_allow_html=True)

            sleep_hours = st.slider("", 1, 7, st.session_state.responses['sleep_hours'], key="sleep_hours_slider")
            st.session_state.responses['sleep_hours'] = sleep_hours

        elif st.session_state.questionnaire_page == 3:
            st.header("Ból Mięśni")
            st.write("Jaki jest poziom odczuwanego bólu mięśni?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: Brak bólu</div>
                <div class="scale-legend-item">4: Średni ból</div>
                <div class="scale-legend-item">7: Silny ból</div>
            </div>
            """, unsafe_allow_html=True)

            soreness = st.slider("", 1, 7, st.session_state.responses['soreness'], key="soreness_slider")
            st.session_state.responses['soreness'] = soreness

        elif st.session_state.questionnaire_page == 4:
            st.header("Poziom Stresu")
            st.write("Jak bardzo zestresowany/a czujesz się dzisiaj?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: Zrelaksowany/a</div>
                <div class="scale-legend-item">4: Umiarkowany stres</div>
                <div class="scale-legend-item">7: Bardzo zestresowany/a</div>
            </div>
            """, unsafe_allow_html=True)

            stress = st.slider("", 1, 7, st.session_state.responses['stress'], key="stress_slider")
            st.session_state.responses['stress'] = stress

        elif st.session_state.questionnaire_page == 5:
            st.header("Trudność Ostatniego Treningu")
            st.write("Jak trudny był Twój ostatni trening?")

            # Scale explanation
            st.markdown("""
            <div class="scale-legend">
                <div class="scale-legend-item">1: Łatwy</div>
                <div class="scale-legend-item">4: Średni</div>
                <div class="scale-legend-item">7: Bardzo trudny</div>
            </div>
            """, unsafe_allow_html=True)

            training_difficulty = st.slider("", 1, 7, st.session_state.responses['training_difficulty'],
                                            key="training_difficulty_slider")
            st.session_state.responses['training_difficulty'] = training_difficulty

        elif st.session_state.questionnaire_page == 6:
            st.header("Dodatkowe Informacje")
            st.write("Chcesz dodać coś jeszcze? (opcjonalne)")

            note = st.text_area("Notatka:", value=st.session_state.responses['note'], height=150)
            st.session_state.responses['note'] = note

            # Submit button only on last page
            submit_button = st.form_submit_button("Wyślij Ankietę")
            if submit_button:
                # Save responses to database
                success = add_response(
                    client_id=st.session_state.client_id,
                    fatigue=st.session_state.responses['fatigue'],
                    sleep_quality=st.session_state.responses['sleep_quality'],
                    sleep_hours=st.session_state.responses['sleep_hours'],
                    soreness=st.session_state.responses['soreness'],
                    stress=st.session_state.responses['stress'],
                    training_difficulty=st.session_state.responses['training_difficulty'],
                    note=st.session_state.responses['note']
                )

                if success:
                    st.success("✅ Twoje odpowiedzi zostały zapisane!")
                    # Reset form for next time
                    st.session_state.responses = {
                        'fatigue': 4,
                        'sleep_quality': 4,
                        'sleep_hours': 4,
                        'soreness': 4,
                        'stress': 4,
                        'training_difficulty': 4,
                        'note': ""
                    }
                    st.session_state.questionnaire_page = 1

                    # Show dashboard after submission
                    st.session_state.show_client_dashboard = True
                    st.rerun()
                else:
                    st.error("❌ Wystąpił błąd podczas zapisywania odpowiedzi. Spróbuj ponownie.")

        if st.session_state.questionnaire_page < total_pages:
            st.form_submit_button("Zapisz i Kontynuuj")

    # Button to view dashboard
    if st.button("Zobacz swoje statystyki"):
        st.session_state.show_client_dashboard = True
        st.rerun()


def show_client_dashboard():
    """Shows the wellness dashboard for an individual client."""
    st.title(f"Panel Klienta: {st.session_state.client_name}")

    # Get client responses
    responses_df = get_responses_by_client(st.session_state.client_id)
    total_responses = count_responses_by_client(st.session_state.client_id)

    # Display stats
    if not responses_df.empty:
        # Calculate wellness score
        wellness_score = calculate_wellness_score(responses_df)

        st.header("Twój Obecny Status")

        # Wellness score display
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Całkowity Wynik Wellness", f"{wellness_score}%")
            st.write(f"Całkowita liczba ankiet: {total_responses}")

        with col2:
            # Latest values
            if not responses_df.empty:
                latest_values = responses_df.iloc[-1]
                latest_date = responses_df.index[-1].strftime("%Y-%m-%d %H:%M")
                st.write(f"Ostatnia ankieta: {latest_date}")

                # Metrics container
                st.markdown("""
                <div class="metric-container">
                """, unsafe_allow_html=True)

                # First row of metrics
                cols = st.columns(3)
                with cols[0]:
                    fig = create_gauge_chart(latest_values['Zmęczenie'], "Zmęczenie", is_reversed=True)
                    st.plotly_chart(fig, use_container_width=True)

                with cols[1]:
                    fig = create_gauge_chart(latest_values['Jakość Snu'], "Jakość Snu", is_reversed=False)
                    st.plotly_chart(fig, use_container_width=True)

                with cols[2]:
                    fig = create_gauge_chart(latest_values['Ból Mięśni'], "Ból Mięśni", is_reversed=True)
                    st.plotly_chart(fig, use_container_width=True)

                # Second row of metrics
                cols = st.columns(3)
                with cols[0]:
                    fig = create_gauge_chart(latest_values['Stres'], "Stres", is_reversed=True)
                    st.plotly_chart(fig, use_container_width=True)

                with cols[1]:
                    fig = create_gauge_chart(latest_values['Trudność Treningu'], "Trudność Treningu", is_reversed=True)
                    st.plotly_chart(fig, use_container_width=True)

                with cols[2]:
                    fig = create_gauge_chart(latest_values['Godziny Snu'], "Godziny Snu", is_reversed=False)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # Chart settings
        st.header("Trendy")

        col1, col2 = st.columns([1, 3])
        with col1:
            filter_days = st.slider("Ilość dni do pokazania:", min_value=7, max_value=90,
                                    value=st.session_state.filter_days)
            st.session_state.filter_days = filter_days

            metrics = st.multiselect(
                "Wybierz metryki do pokazania:",
                options=['Zmęczenie', 'Jakość Snu', 'Godziny Snu', 'Ból Mięśni', 'Stres', 'Trudność Treningu'],
                default=st.session_state.selected_metrics
            )
            st.session_state.selected_metrics = metrics

            view_mode = st.radio("Typ wykresu:", ["Liniowy", "Radarowy"],
                                 index=0 if st.session_state.view_mode == "line" else 1)
            st.session_state.view_mode = "line" if view_mode == "Liniowy" else "radar"

        with col2:
            if st.session_state.view_mode == "line":
                trend_chart = create_trend_chart(responses_df, metrics, filter_days)
                if trend_chart:
                    st.plotly_chart(trend_chart, use_container_width=True)
                else:
                    st.info("Niewystarczająca ilość danych do wygenerowania wykresu.")
            else:
                radar_chart = create_radar_chart(responses_df)
                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True)
                else:
                    st.info("Niewystarczająca ilość danych do wygenerowania wykresu radarowego (minimum 2 ankiety).")

        # Notes section
        st.header("Notatki Trenera")
        trainer_notes = get_trainer_notes(st.session_state.client_id)

        if not trainer_notes.empty:
            st.dataframe(trainer_notes, use_container_width=True)
        else:
            st.info("Brak notatek od trenera.")

        # Download links
        st.header("Exportuj Dane")

        col1, col2 = st.columns(2)
        with col1:
            if not responses_df.empty:
                st.markdown(get_download_link(
                    responses_df,
                    f"dane_{st.session_state.client_name}.csv",
                    "Pobierz dane jako CSV"
                ), unsafe_allow_html=True)

        with col2:
            if not responses_df.empty:
                st.markdown(get_pdf_download_link(
                    st.session_state.client_name,
                    responses_df,
                    "Pobierz raport PDF"
                ), unsafe_allow_html=True)
    else:
        st.info("Brak danych. Wypełnij swoją pierwszą ankietę, aby zobaczyć statystyki.")

    # Button to go back to questionnaire
    if st.button("Wypełnij nową ankietę"):
        st.session_state.show_client_dashboard = False
        st.rerun()

    # Logout button
    if st.button("Wyloguj"):
        st.session_state.logged_in = False
        st.session_state.show_client_dashboard = False
        st.rerun()


def show_trainer_dashboard():
    """Shows the trainer dashboard for managing clients."""
    st.title("Panel Trenera")

    tab1, tab2 = st.tabs(["Lista Klientów", "Dodaj Nowego Klienta"])

    with tab1:
        st.header("Twoi Klienci")

        clients = get_all_clients()

        if clients:
            # Create a selection for client
            client_options = [f"{name} (Kod: {code})" for _, name, code in clients]
            selected_client = st.selectbox("Wybierz klienta:", client_options)

            # Extract client ID from selection
            selected_index = client_options.index(selected_client)
            selected_client_id, selected_client_name, _ = clients[selected_index]

            # Store selected client ID in session state
            st.session_state.selected_client_id_trainer = selected_client_id

            # Display client data
            st.subheader(f"Dane dla: {selected_client_name}")

            responses_df = get_responses_by_client(selected_client_id)
            total_responses = count_responses_by_client(selected_client_id)

            if not responses_df.empty:
                st.write(f"Liczba ankiet: {total_responses}")

                # Latest values
                latest_values = responses_df.iloc[-1]
                latest_date = responses_df.index[-1].strftime("%Y-%m-%d %H:%M")
                st.write(f"Ostatnia ankieta: {latest_date}")

                # Display metrics
                cols = st.columns(3)
                for i, metric in enumerate(
                        ['Zmęczenie', 'Jakość Snu', 'Ból Mięśni', 'Stres', 'Trudność Treningu', 'Godziny Snu']):
                    with cols[i % 3]:
                        is_reversed = metric not in ['Jakość Snu', 'Godziny Snu']
                        fig = create_gauge_chart(latest_values[metric], metric, is_reversed=is_reversed)
                        st.plotly_chart(fig, use_container_width=True)

                # Trend chart
                st.subheader("Trendy")

                # Chart settings
                col1, col2 = st.columns([1, 3])
                with col1:
                    filter_days = st.slider(
                        "Ilość dni:",
                        min_value=7,
                        max_value=90,
                        value=st.session_state.filter_days,
                        key="trainer_days"
                    )

                    metrics = st.multiselect(
                        "Wybierz metryki:",
                        options=['Zmęczenie', 'Jakość Snu', 'Godziny Snu', 'Ból Mięśni', 'Stres', 'Trudność Treningu'],
                        default=st.session_state.selected_metrics,
                        key="trainer_metrics"
                    )

                with col2:
                    trend_chart = create_trend_chart(responses_df, metrics, filter_days)
                    if trend_chart:
                        st.plotly_chart(trend_chart, use_container_width=True)
                    else:
                        st.info("Niewystarczająca ilość danych.")

                # Notes from client
                if 'Notatka' in responses_df.columns:
                    notes_df = responses_df[responses_df['Notatka'].str.strip() != ''][['Notatka']]
                    if not notes_df.empty:
                        st.subheader("Notatki Klienta")
                        for idx, row in notes_df.iterrows():
                            st.markdown(f"**{idx.strftime('%Y-%m-%d %H:%M')}:** {row['Notatka']}")
                    else:
                        st.info("Klient nie dodał żadnych notatek.")

                # Download links
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(get_download_link(
                        responses_df,
                        f"dane_{selected_client_name}.csv",
                        "Pobierz dane jako CSV"
                    ), unsafe_allow_html=True)

                with col2:
                    st.markdown(get_pdf_download_link(
                        selected_client_name,
                        responses_df,
                        "Pobierz raport PDF"
                    ), unsafe_allow_html=True)
            else:
                st.info("Ten klient nie ma jeszcze żadnych danych.")

            # Trainer notes section
            st.subheader("Dodaj Notatkę dla Klienta")
            with st.form("trainer_note_form"):
                note_text = st.text_area("Twoja notatka:", height=100)
                note_date = st.date_input("Data:", datetime.now())

                submit_note = st.form_submit_button("Dodaj Notatkę")

                if submit_note and note_text:
                    note_datetime = datetime.combine(note_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                    success = add_trainer_note(selected_client_id, note_text, note_datetime)
                    if success:
                        st.success("✅ Notatka dodana!")
                        st.rerun()
                    else:
                        st.error("❌ Wystąpił błąd podczas dodawania notatki.")

            # Display existing trainer notes
            st.subheader("Twoje Poprzednie Notatki")
            trainer_notes = get_trainer_notes(selected_client_id)

            if not trainer_notes.empty:
                st.dataframe(trainer_notes, use_container_width=True)
            else:
                st.info("Nie masz jeszcze żadnych notatek dla tego klienta.")
        else:
            st.info("Brak klientów. Dodaj nowego klienta, aby rozpocząć.")

    with tab2:
        st.header("Dodaj Nowego Klienta")

        with st.form("add_client_form"):
            client_name = st.text_input("Imię i nazwisko klienta:")
            submit_button = st.form_submit_button("Dodaj Klienta")

            if submit_button and client_name:
                unique_code = add_client(client_name)
                if unique_code:
                    st.success(f"✅ Klient dodany! Kod dostępu: **{unique_code}**")
                    st.info("Przekaż ten kod klientowi, aby mógł się zalogować.")
                else:
                    st.error("❌ Wystąpił błąd podczas dodawania klienta.")

    # Logout button
    if st.button("Wyloguj", key="trainer_logout"):
        st.session_state.trainer_logged_in = False
        st.rerun()


def main():
    """Main function to control the app flow."""
    # Initialize session state
    init_session_state()

    # Determine which page to show
    if st.session_state.trainer_logged_in:
        show_trainer_dashboard()
    elif st.session_state.logged_in:
        if st.session_state.show_client_dashboard:
            show_client_dashboard()
        else:
            show_client_questionnaire()
    else:
        show_login_screen()


# Run the app
if __name__ == "__main__":
    main()