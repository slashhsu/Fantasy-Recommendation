import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import xml.etree.ElementTree as ET
from yahoo_oauth import OAuth2
import os
from dotenv import load_dotenv
import time
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Fantasy Basketball Free Agent Recommender",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
STAT_CATEGORIES = [
    'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO'
]

PERIODS = {
    'Last Week': 'lastweek',
    'Last Month': 'lastmonth',
    'Season': 'season'
}

STAT_TYPES = ['total', 'average']

# Hardcoded Yahoo stat ID to stat name mapping for standard 9-cat
STAT_ID_TO_NAME = {
    '0': 'GP',
    '3': 'FGM',
    '4': 'FGA',
    '5': 'FG%',
    '6': 'FTM',
    '7': 'FTA',
    '8': 'FT%',
    '10': '3PTM',
    '12': 'PTS',
    '13': 'REB',
    '14': 'AST',
    '15': 'ST',
    '16': 'BLK',
    '17': 'TO',
}

RANK_COLS = [
    'FGM', 'FTM', 'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO'
]

# Initialize session state
if 'oauth' not in st.session_state:
    st.session_state.oauth = None

def load_oauth():
    """Load OAuth credentials and initialize OAuth2 object"""
    try:
        oauth = OAuth2(None, None, from_file="oauth.json")
        # Force token refresh
        oauth.refresh_access_token()
        return oauth
    except Exception as e:
        st.error(f"Error loading OAuth credentials: {str(e)}")
        return None

def parse_xml_response(xml_content):
    """Parse XML response from Yahoo Fantasy API"""
    try:
        root = ET.fromstring(xml_content)
        # Define the namespace
        ns = {'yahoo': 'http://www.yahooapis.com/v1/base.rng',
              'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
        
        leagues = []
        # Find all game elements
        for game in root.findall('.//fantasy:game', ns):
            code = game.find('fantasy:code', ns)
            if code is not None and code.text == 'nba':
                game_key = game.find('fantasy:game_key', ns)
                season = game.find('fantasy:season', ns)
                if game_key is not None and season is not None:
                    leagues.append({
                        'key': game_key.text,
                        'name': f"NBA {season.text}"
                    })
        return leagues
    except ET.ParseError as e:
        st.error(f"Error parsing XML: {str(e)}")
        return []

def get_user_leagues(oauth):
    """Fetch user's fantasy leagues"""
    if not oauth:
        return []
    
    url = "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games"
    try:
        response = oauth.session.get(url)
        
        # Debug information
        st.write("Response Status Code:", response.status_code)
        
        # Check if the response is successful
        if response.status_code != 200:
            st.error(f"API request failed with status code: {response.status_code}")
            st.write("Response content:", response.text)
            return []
        
        # Parse XML response
        try:
            root = ET.fromstring(response.text)
            # Define the namespace
            ns = {'yahoo': 'http://www.yahooapis.com/v1/base.rng',
                  'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
            
            leagues = []
            # Find all game elements
            for game in root.findall('.//fantasy:game', ns):
                code = game.find('fantasy:code', ns)
                if code is not None and code.text == 'nba':
                    game_key = game.find('fantasy:game_key', ns)
                    season = game.find('fantasy:season', ns)
                    if game_key is not None and season is not None:
                        # Get leagues for this game
                        leagues_url = f"https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games;game_keys={game_key.text}/leagues"
                        leagues_response = oauth.session.get(leagues_url)
                        if leagues_response.status_code == 200:
                            leagues_root = ET.fromstring(leagues_response.text)
                            for league in leagues_root.findall('.//fantasy:league', ns):
                                league_key = league.find('fantasy:league_key', ns)
                                league_name = league.find('fantasy:name', ns)
                                if league_key is not None and league_name is not None:
                                    leagues.append({
                                        'key': league_key.text,
                                        'name': f"{league_name.text} ({season.text})"
                                    })
            
            if not leagues:
                st.warning("No NBA leagues found in the response")
            return leagues
            
        except ET.ParseError as e:
            st.error(f"Error parsing XML: {str(e)}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error occurred: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return []

def get_league_players(oauth, league_key, stat_periods_types, max_players=150):
    """Fetch all free agents, fetch stats for each period/type, average or sum as appropriate, return top 50 by weighted rank."""
    if not oauth:
        return pd.DataFrame()
    stat_id_to_name = st.session_state.get('stat_id_to_name', {})
    st.write('DEBUG: stat_id_to_name mapping:', stat_id_to_name)
    # Step 1: Get the list of true free agent player_keys
    fa_url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;status=FA"
    fa_keys = set()
    try:
        fa_resp = oauth.session.get(fa_url)
        if fa_resp.status_code == 200:
            fa_root = ET.fromstring(fa_resp.text)
            ns = {'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
            for player in fa_root.findall('.//fantasy:player', ns):
                player_key = player.find('fantasy:player_key', ns)
                if player_key is not None:
                    fa_keys.add(player_key.text)
        else:
            st.warning('Could not fetch free agent list, showing all players.')
    except Exception as e:
        st.warning(f'Error fetching free agent list: {e}')
    # Step 2: Fetch all player info
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players"
    try:
        response = oauth.session.get(url)
        st.write("Response Status Code:", response.status_code)
        if response.status_code != 200:
            st.error(f"API request failed with status code: {response.status_code}")
            st.write("Response content:", response.text)
            return pd.DataFrame()
        root = ET.fromstring(response.text)
        ns = {'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
        player_infos = []
        for player in root.findall('.//fantasy:player', ns):
            player_data = {}
            player_key = player.find('fantasy:player_key', ns)
            if player_key is not None:
                player_data['player_key'] = player_key.text
            else:
                continue
            full_name = player.find('.//fantasy:full', ns)
            if full_name is not None:
                player_data['name'] = full_name.text
            else:
                player_data['name'] = 'Unknown'
            status = player.find('.//fantasy:status', ns)
            if status is not None:
                player_data['status'] = status.text
            else:
                player_data['status'] = ''
            player_infos.append(player_data)
        # Only keep players whose player_key is in the FA list
        if fa_keys:
            fa_infos = [p for p in player_infos if p['player_key'] in fa_keys]
        else:
            fa_infos = player_infos  # fallback: show all
        if not fa_infos:
            st.warning('No free agents found in league.')
            return pd.DataFrame()
        st.info(f"Fetching stats for up to {max_players} free agents. This may take a while...")
        fa_infos = fa_infos[:max_players]
        # For each free agent, fetch their stats for each period/type
        stats_list = []
        progress = st.progress(0)
        for i, pinfo in enumerate(fa_infos):
            player_key = pinfo['player_key']
            player_stats = {'player_key': player_key}
            for period, stat_type in stat_periods_types:
                stats_url = f"https://fantasysports.yahooapis.com/fantasy/v2/player/{player_key}/stats;type={period}"
                stats_resp = oauth.session.get(stats_url)
                if stats_resp.status_code == 200:
                    try:
                        sroot = ET.fromstring(stats_resp.text)
                        if i == 0 and period == stat_periods_types[0][0]:
                            st.write('DEBUG: Raw XML for first player:', stats_resp.text)
                        found_stat = False
                        for stat in sroot.findall('.//fantasy:stat', ns):
                            stat_id = stat.find('fantasy:stat_id', ns)
                            stat_value = stat.find('fantasy:value', ns)
                            if stat_id is not None and stat_value is not None:
                                found_stat = True
                                stat_name = stat_id_to_name.get(stat_id.text, stat_id.text)
                                try:
                                    val = float(stat_value.text)
                                except (ValueError, TypeError):
                                    val = 0.0
                                key = f"{stat_name}__{period}__{stat_type}"
                                player_stats[key] = val
                        if not found_stat:
                            st.warning(f'No stats found in XML for player {player_key} [{period}]')
                    except Exception as e:
                        st.write(f"Error parsing stats for {player_key}: {e}")
                else:
                    st.write(f"Failed to fetch stats for {player_key} [{period}]")
            stats_list.append(player_stats)
            progress.progress((i+1)/len(fa_infos))
            time.sleep(0.2)
        progress.empty()
        info_df = pd.DataFrame(fa_infos)
        stats_df = pd.DataFrame(stats_list)
        if stats_df.empty:
            st.warning('No stats found for free agents.')
            return info_df
        merged = pd.merge(info_df, stats_df, on='player_key', how='left')
        return merged
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()

def recommend_players(players_df, stat_weights, stat_periods_types):
    df = players_df.copy()
    df = df.fillna(0)
    stat_names = st.session_state.get('stat_names', [])
    # For each stat, aggregate across all selected periods/types
    for stat in stat_names:
        cols = [f"{stat}__{period}__{stat_type}" for period, stat_type in stat_periods_types if f"{stat}__{period}__{stat_type}" in df.columns]
        if not cols:
            continue
        # If stat_type is 'average', take mean; if 'total', take sum; else mean
        if any('average' in c for c in cols):
            df[stat + '_agg'] = df[cols].mean(axis=1)
        else:
            df[stat + '_agg'] = df[cols].sum(axis=1)
    # Rank each stat
    for stat in stat_names:
        if stat + '_agg' in df.columns:
            if 'Turnover' in stat or stat == 'TO':  # lower is better
                df[stat + '_rank'] = df[stat + '_agg'].rank(ascending=True, method='min')
            else:
                df[stat + '_rank'] = df[stat + '_agg'].rank(ascending=False, method='min')
    # Apply weights and sum
    for stat in stat_names:
        if stat + '_rank' in df.columns:
            df[stat + '_weighted'] = df.get(stat + '_rank', 0) * stat_weights.get(stat, 1)
    weighted_cols = [stat + '_weighted' for stat in stat_names if stat + '_weighted' in df.columns]
    df['Weighted_Performance'] = df[weighted_cols].sum(axis=1)
    df = df.sort_values('Weighted_Performance', ascending=True)
    # Include raw stat columns (aggregated) for radar plot
    raw_stat_cols = [stat + '_agg' for stat in stat_names if stat + '_agg' in df.columns]
    display_cols = ['name', 'Weighted_Performance'] + weighted_cols + raw_stat_cols
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in df.columns]
    return df[display_cols].head(50)

def get_league_stat_categories(oauth, league_key):
    """Fetch stat categories for the league and return (stat_names, stat_id_to_name)"""
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings"
    try:
        response = oauth.session.get(url)
        if response.status_code != 200:
            st.warning(f"Could not fetch league stat categories: {response.status_code}")
            return [], {}
        root = ET.fromstring(response.text)
        ns = {'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
        stat_names = []
        stat_id_to_name = {}
        for stat in root.findall('.//fantasy:stat', ns):
            stat_id_elem = stat.find('fantasy:stat_id', ns)
            stat_name_elem = stat.find('fantasy:name', ns)
            if stat_id_elem is not None and stat_name_elem is not None:
                stat_id = stat_id_elem.text
                stat_name = stat_name_elem.text
                stat_names.append(stat_name)
                stat_id_to_name[stat_id] = stat_name
        return stat_names, stat_id_to_name
    except Exception as e:
        st.warning(f"Error fetching league stat categories: {e}")
        return [], {}

def get_latest_match_stats(oauth, league_key):
    """Fetch the latest match stats for the selected league and return as a DataFrame."""
    if not oauth or not league_key:
        return pd.DataFrame()
    # Get the current week (scoreboard endpoint without week param returns latest)
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/scoreboard"
    try:
        response = oauth.session.get(url)
        if response.status_code != 200:
            st.warning(f"Could not fetch latest match stats: {response.status_code}")
            return pd.DataFrame()
        root = ET.fromstring(response.text)
        ns = {'yahoo': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
        # Get stat id to name mapping
        stat_names, stat_id_to_name = get_league_stat_categories(oauth, league_key)
        stat_id_map = {v: k for k, v in stat_id_to_name.items()}
        # Parse matchups
        data = []
        for matchup in root.findall('.//yahoo:matchup', ns):
            for team in matchup.findall('.//yahoo:team', ns):
                team_name = team.find('yahoo:name', ns)
                stats = team.findall('.//yahoo:stat', ns)
                team_stats = {'Team Name': team_name.text if team_name is not None else 'Unknown'}
                for stat in stats:
                    stat_id = stat.find('yahoo:stat_id', ns)
                    stat_value = stat.find('yahoo:value', ns)
                    if stat_id is not None and stat_value is not None:
                        stat_name = stat_id_to_name.get(stat_id.text, f"Stat {stat_id.text}")
                        team_stats[stat_name] = stat_value.text
                data.append(team_stats)
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error fetching latest match stats: {e}")
        return pd.DataFrame()

def get_my_team_name(oauth, league_key):
    """Fetch the user's team name for the selected league."""
    if not oauth or not league_key:
        return None
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/teams"
    try:
        response = oauth.session.get(url)
        if response.status_code != 200:
            return None
        root = ET.fromstring(response.text)
        ns = {'fantasy': 'http://fantasysports.yahooapis.com/fantasy/v2/base.rng'}
        for team in root.findall('.//fantasy:team', ns):
            is_owned = team.find('fantasy:is_owned_by_current_login', ns)
            if is_owned is not None and is_owned.text == '1':
                name_elem = team.find('fantasy:name', ns)
                if name_elem is not None:
                    return name_elem.text
        return None
    except Exception:
        return None

def highlight_winner(val, winner):
    if winner == 'me':
        return 'background-color: #b6fcb6; font-weight: bold;'
    elif winner == 'opponent':
        return 'background-color: #ffb3b3;'
    else:
        return ''

def get_vertical_matchup_table(match_stats_df, my_team_name):
    """Return a vertical comparison DataFrame (stat as row, columns: Me, Opponent, Winner)"""
    if match_stats_df is None or match_stats_df.empty or not my_team_name:
        return None, None
    # Find the row for my team
    my_row = match_stats_df[match_stats_df['Team Name'] == my_team_name]
    if my_row.empty:
        return None, None
    # Find the row for the opponent (same matchup, i.e., same index +/-1)
    idx = my_row.index[0]
    if idx % 2 == 0:
        opp_idx = idx + 1
    else:
        opp_idx = idx - 1
    if opp_idx < 0 or opp_idx >= len(match_stats_df):
        return None, None
    opp_row = match_stats_df.iloc[[opp_idx]]
    # Build vertical table
    stats_cols = [col for col in match_stats_df.columns if col != 'Team Name']
    data = []
    winners = []
    for col in stats_cols:
        my_val = my_row.iloc[0][col]
        opp_val = opp_row.iloc[0][col]
        # Try to compare as float, else as string
        def parse_val(val):
            if isinstance(val, str) and '/' in val:
                return float(val.split('/')[0])
            try:
                return float(str(val).replace('%','').replace('.','0.') if str(val).startswith('.') else val)
            except Exception:
                return None
        my_v = parse_val(my_val)
        opp_v = parse_val(opp_val)
        if my_v is not None and opp_v is not None:
            if col.lower() in ['turnovers', 'to']:
                # Lower is better
                if my_v < opp_v:
                    winner = 'me'
                elif my_v > opp_v:
                    winner = 'opponent'
                else:
                    winner = 'tie'
            else:
                # Higher is better
                if my_v > opp_v:
                    winner = 'me'
                elif my_v < opp_v:
                    winner = 'opponent'
                else:
                    winner = 'tie'
        else:
            winner = 'tie'
        data.append([col, my_val, opp_val, winner])
        winners.append(winner)
    vert_df = pd.DataFrame(data, columns=['Stat', 'Me', 'Opponent', 'Winner'])
    return vert_df, winners

def main():
    # Header with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #2E4053; padding: 1rem;'>
            🏀 Fantasy Basketball Free Agent Recommender
        </h1>
    """, unsafe_allow_html=True)
    
    # Initialize OAuth
    if st.session_state.oauth is None:
        st.session_state.oauth = load_oauth()
    
    if st.session_state.oauth is None:
        st.error("Please ensure you have valid OAuth credentials in oauth.json")
        return
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 League Selection")
        # League Selection with custom styling
        leagues = get_user_leagues(st.session_state.oauth)
        if not leagues:
            st.error("No leagues found. Please check your Yahoo Fantasy account.")
            return
        
        selected_league = st.selectbox(
            "Select your league",
            options=leagues,
            format_func=lambda x: x['name']
        )
        
        # Fetch league stat categories
        stat_names, stat_id_to_name = get_league_stat_categories(st.session_state.oauth, selected_league['key'])
        if not stat_names:
            st.error("Could not fetch stat categories for this league.")
            return
        st.session_state.stat_id_to_name = stat_id_to_name
        st.session_state.stat_names = stat_names

        st.markdown("### ⚖️ Stat Weights")
        # Stat Category Selection with sliders in a container
        with st.container():
            stat_weights = {}
            for stat in stat_names:
                weight = st.slider(
                    f"Weight for {stat}",
                    0.0, 2.0, 1.0, 0.1,
                    help=f"Adjust the importance of {stat} in player recommendations"
                )
                stat_weights[stat] = weight

    with col2:
        st.markdown("### 📈 Time Period & Stat Type")
        # Period and Stat Type Selection in a container
        with st.container():
            col3, col4 = st.columns(2)
            with col3:
                selected_periods = st.multiselect(
                    "Select Time Period(s)",
                    options=list(PERIODS.keys()),
                    default=[list(PERIODS.keys())[0]]
                )
            with col4:
                selected_stat_types = st.multiselect(
                    "Select Stat Type(s)",
                    options=STAT_TYPES,
                    default=[STAT_TYPES[0]]
                )
            stat_periods_types = [(PERIODS[p], t) for p in selected_periods for t in selected_stat_types]

        st.markdown("### 🎯 Match Result Weights")
        # Match Result Weights in a container
        with st.container():
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("#### Win Weights")
                win_weights = {
                    'Win 1': st.number_input("Win by 1", value=1.3, min_value=0.0, max_value=5.0, step=0.1),
                    'Win 2': st.number_input("Win by 2", value=1.5, min_value=0.0, max_value=5.0, step=0.1),
                    'Win 3+': st.number_input("Win by 3+", value=1.7, min_value=0.0, max_value=5.0, step=0.1)
                }
            with col6:
                st.markdown("#### Loss Weights")
                loss_weights = {
                    'Lose 1': st.number_input("Lose by 1", value=-1.3, min_value=-5.0, max_value=0.0, step=0.1),
                    'Lose 2': st.number_input("Lose by 2", value=-1.5, min_value=-5.0, max_value=0.0, step=0.1),
                    'Lose 3+': st.number_input("Lose by 3+", value=-1.7, min_value=-5.0, max_value=0.0, step=0.1)
                }
        # Compute stat win summary and vertical table here, before any chart/table
        match_stats_df = get_latest_match_stats(st.session_state.oauth, selected_league['key'])
        my_team_name = get_my_team_name(st.session_state.oauth, selected_league['key'])
        vert_df, winners = get_vertical_matchup_table(match_stats_df, my_team_name) if not match_stats_df.empty else (None, None)
        if vert_df is not None:
            win_count = sum(1 for w in vert_df['Winner'] if w == 'me')
            lose_count = sum(1 for w in vert_df['Winner'] if w == 'opponent')
            tie_count = sum(1 for w in vert_df['Winner'] if w == 'tie')
            my_row = match_stats_df[match_stats_df['Team Name'] == my_team_name]
            idx = my_row.index[0]
            if idx % 2 == 0:
                opp_idx = idx + 1
            else:
                opp_idx = idx - 1
            opponent_name = match_stats_df.iloc[opp_idx]['Team Name']
            st.markdown(f"<div style='margin:1em 0 1em 0; font-size:1.1em;'>(Latest Matchup) Stat Results: <b style='color:#2E4053'>You won {win_count}</b>, <b style='color:#B22222'>Opponent won {lose_count}</b>, <b style='color:#888'>{tie_count} ties</b></div>", unsafe_allow_html=True)
            vert_df_display = vert_df.drop(columns=['Winner'])
            st.markdown(f"#### My Matchup (Vertical Comparison)<br><b style='color:#2E4053'>Me: {my_team_name}</b> <b style='color:#B22222; margin-left:2em;'>Opponent: {opponent_name}</b>", unsafe_allow_html=True)
            def vert_highlight(row):
                idx = row.name
                winner = vert_df.iloc[idx]['Winner']
                if winner == 'me':
                    return ['','background-color: #b6fcb6; font-weight: bold;','']
                elif winner == 'opponent':
                    return ['','background-color: #ffb3b3;','']
                else:
                    return ['','','']
            st.dataframe(vert_df_display.style.apply(vert_highlight, axis=1), use_container_width=True)

    # Show latest match stats above recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Latest Match Stats")
    if not match_stats_df.empty:
        # Only apply gradient to columns that are fully numeric
        def is_numeric_column(series):
            try:
                # Try to convert all values to float (ignore NaN)
                for v in series.dropna():
                    # If value has a slash, it's not numeric
                    if isinstance(v, str) and '/' in v:
                        return False
                    float(str(v).replace('%','').replace('.','0.') if str(v).startswith('.') else v)
                return True
            except Exception:
                return False
        stat_cols = [col for col in match_stats_df.columns if col != 'Team Name' and is_numeric_column(match_stats_df[col])]
        st.dataframe(
            match_stats_df.style.background_gradient(cmap='RdYlGn', subset=stat_cols),
            use_container_width=True
        )
    else:
        st.info("No latest match stats available.")

    # Generate Recommendations button centered
    col7, col8, col9 = st.columns([1,2,1])
    with col8:
        if st.button("🎯 Generate Recommendations", use_container_width=True):
            with st.spinner("Analyzing free agents..."):
                players_df = get_league_players(st.session_state.oauth, selected_league['key'], stat_periods_types, max_players=150)
                if not players_df.empty:
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["📊 Recommendations", "📈 Raw Data"])
                    
                    with tab1:
                        rec_df = recommend_players(players_df, stat_weights, stat_periods_types)
                        
                        # Always show all available raw stat columns (ending with '_agg') on the radar plot
                        raw_stat_cols = [col for col in rec_df.columns if col.endswith('_agg')]
                        top_5_players = rec_df.head(5)
                        fig = go.Figure()
                        for _, player in top_5_players.iterrows():
                            stats = [player.get(stat, 0) for stat in raw_stat_cols]
                            fig.add_trace(go.Scatterpolar(
                                r=stats,
                                theta=[col.replace('_agg', '') for col in raw_stat_cols],
                                name=player['name'],
                                fill='toself'
                            ))
                        # Find max value for each stat to set axis range
                        max_values = [
                            max([top_5_players.iloc[i].get(stat, 0) for i in range(len(top_5_players))])
                            for stat in raw_stat_cols
                        ]
                        max_value = max(max_values) if max_values else 1
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max_value]
                                )
                            ),
                            showlegend=True,
                            title="Top 5 Players Performance Comparison (Raw Stats)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display recommendations in a styled table
                        st.dataframe(
                            rec_df.style.background_gradient(cmap='RdYlGn_r', subset=['Weighted_Performance']),
                            use_container_width=True
                        )
                    
                    with tab2:
                        st.dataframe(
                            players_df.style.background_gradient(cmap='RdYlGn'),
                            use_container_width=True
                        )
                else:
                    st.error("No players found in the league")

if __name__ == "__main__":
    main() 