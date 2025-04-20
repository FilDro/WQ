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
    try:
        # Fetch credentials from Streamlit secrets
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"❌ Brak poświadczeń Supabase w Streamlit Secrets: {e}")
        st.stop()  # Stop execution if secrets are missing
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji klienta Supabase: {e}")
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
            df.columns = ['Zmęczenie', 'Jakość Snu', 'Godziny Snu', 'Ból Mięśni', 'Stres', 'Trudność Treningu',
                          'Notatka']
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
        'Jakość Snu': 8 - latest['Jakość Snu'],
        'Godziny Snu': 8 - latest['Godziny Snu'],
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