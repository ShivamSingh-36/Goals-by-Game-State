import matplotlib
matplotlib.use('Agg')
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plot import plot_pitch
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')
import zipfile
import os

@st.cache_data
def load_data():
    try:
        
        if not os.path.exists('Final.csv'):
            
            zip_file = None
            if os.path.exists('Final.zip'):
                zip_file = 'Final.zip'
            elif os.path.exists('Final.csv.zip'):
                zip_file = 'Final.csv.zip'
            
            if zip_file:
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')
            else:
                st.error("❌ No data file found. Please ensure Final.zip exists.")
                return pd.DataFrame()
        
        if not os.path.exists('Final.csv'):
            st.error("❌ Error: Data file could not be loaded.")
            return pd.DataFrame()
            
        df = pd.read_csv('Final.csv', encoding='utf-8-sig')
        
        def clean_player_id(pid):
            pid_str = str(pid)
            if '(' in pid_str and ')' in pid_str:
                match = re.search(r'\((\d+)\)', pid_str)
                if match:
                    return match.group(1)
            return pid_str
        
        df['Player ID'] = df['Player ID'].apply(clean_player_id)
        
        df['Season'] = df['Season'].astype(int)
        df['Match ID'] = df['Match ID'].astype(str)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Error: 'Final.csv' file not found. Please ensure the data file exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()
    
df = load_data()

@st.cache_data
def pre_format_data():
    if df.empty:
        return pd.DataFrame()
    df_copy = df.copy()

    if 'Shot Result' in df_copy.columns:
        df_copy['isGoal'] = (df_copy['Shot Result'] == 'Goal').astype(int)
    elif 'isGoal' in df_copy.columns:
        df_copy['isGoal'] = df_copy['isGoal'].apply(lambda x: 1 if x == 1 else 0)
    
    return df_copy

df_fil = pre_format_data()

@st.cache_data
def calculate_advanced_metrics(player_df):
    """Calculate advanced shooting metrics for a player"""
    if len(player_df) == 0:
        return {}
    
    try:
        metrics = {}
        
        required_cols = ['isGoal', 'Shot xG']
        if not all(col in player_df.columns for col in required_cols):
            return {}

        total_shots = len(player_df)
        total_goals = float(player_df['isGoal'].sum())
        total_xg = float(player_df['Shot xG'].sum())
        
        metrics['conversion_rate'] = (total_goals / total_shots * 100) if total_shots > 0 else 0
        metrics['xg_per_shot'] = player_df['Shot xG'].mean() if total_shots > 0 else 0
        metrics['avg_xg_per_shot'] = total_xg / total_shots if total_shots > 0 else 0

        metrics['xg_overperformance'] = total_goals - total_xg
        metrics['xg_efficiency'] = (total_goals / total_xg * 100) if total_xg > 0 else 0

        metrics['total_shots'] = int(total_shots)
        metrics['total_goals'] = int(total_goals)
        metrics['total_xg'] = total_xg

        if len(player_df) > 1:
            xg_std = player_df['Shot xG'].std()
            metrics['shot_consistency'] = 1 / (1 + xg_std) if xg_std > 0 else 1
        else:
            metrics['shot_consistency'] = 0
        
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

@st.cache_data
def analyze_shot_zones(player_df):
    """Analyze shooting patterns by pitch zones"""
    if len(player_df) == 0:
        return pd.DataFrame()
    
    try:
        
        if 'x' not in player_df.columns or 'y' not in player_df.columns:
            return pd.DataFrame()
        
        def get_zone(row):
            x, y = row['x'] * 100, row['y'] * 100
            
            if x > 88:
                zone_x = "6-yard box"
            elif x > 83:
                zone_x = "Penalty area"
            elif x > 66:
                zone_x = "Edge of box"
            else:
                zone_x = "Long range"

            if 40 <= y <= 60:
                zone_y = "Central"
            elif (y < 40 and y >= 20) or (y > 60 and y <= 80):
                zone_y = "Half-space"
            else:
                zone_y = "Wide"
            
            return f"{zone_y} - {zone_x}"
        
        player_df = player_df.copy()
        player_df['Zone'] = player_df.apply(get_zone, axis=1)
        
        zone_stats = player_df.groupby('Zone').agg({
            'Shot xG': ['sum', 'mean', 'count'],
            'isGoal': 'sum'
        }).round(3)
        
        zone_stats.columns = ['Total xG', 'Avg xG', 'Shots', 'Goals']
        zone_stats['Conversion %'] = (zone_stats['Goals'] / zone_stats['Shots'] * 100).round(1)
        zone_stats['xG Diff'] = (zone_stats['Goals'] - zone_stats['Total xG']).round(2)
        
        return zone_stats.sort_values('Shots', ascending=False)
    except Exception as e:
        st.error(f"Error analyzing zones: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def seasonal_performance(player_df):
    """Analyze performance trends across seasons"""
    if len(player_df) == 0:
        return pd.DataFrame()
    
    try:
        if 'Season' not in player_df.columns:
            return pd.DataFrame()
        
        season_stats = player_df.groupby('Season').agg({
            'Shot xG': ['sum', 'mean', 'count'],
            'isGoal': 'sum'
        }).round(3)
        
        season_stats.columns = ['Total xG', 'xG/Shot', 'Shots', 'Goals']
        season_stats['Conversion %'] = (season_stats['Goals'] / season_stats['Shots'] * 100).round(1)
        season_stats['xG Overperformance'] = (season_stats['Goals'] - season_stats['Total xG']).round(2)
        
        return season_stats
    except Exception as e:
        st.error(f"Error analyzing seasonal performance: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_situation_breakdown(player_df):
    """Breakdown by shot situation (open play, counter, etc)"""
    if len(player_df) == 0:
        return pd.DataFrame()
    
    try:
        if 'Situation' not in player_df.columns:
            return pd.DataFrame()
        
        situation_stats = player_df.groupby('Situation').agg({
            'Shot xG': ['sum', 'mean', 'count'],
            'isGoal': 'sum'
        }).round(3)
        
        situation_stats.columns = ['Total xG', 'xG/Shot', 'Shots', 'Goals']
        situation_stats['Conversion %'] = (situation_stats['Goals'] / situation_stats['Shots'] * 100).round(1)
        
        return situation_stats.sort_values('Shots', ascending=False)
    except Exception as e:
        st.error(f"Error analyzing situations: {str(e)}")
        return pd.DataFrame()

def find_distance(x, y):
    """Calculate distance from goal"""
    try:
        return np.sqrt(np.sum(np.square(np.array([100, 50])-np.array([x*100, y*100]))))
    except:
        return 0

if df.empty:
    st.error("❌ Failed to load data. Please check that 'Final.csv' exists and is properly formatted.")
    st.stop()

st.title("State of Play")
st.text("An app for finding Understat player by player shot data based on their shooting\nand finishing performance in different gamestates")
st.caption("Made by Shivam Singh.", unsafe_allow_html=False)
st.sidebar.header("PLAYER INPUTS")

try:
    player_options = df[['Player', 'Player ID']].drop_duplicates()
    player_options['display'] = player_options['Player'].str.strip() + ' (' + player_options['Player ID'].astype(str) + ')'
    player_dict = dict(zip(player_options['display'], player_options['Player ID'].astype(str)))
    sorted_options = sorted(player_dict.keys())
    
    if len(sorted_options) == 0:
        st.error("No players found in the dataset.")
        st.stop()
    
    selected_display = st.sidebar.selectbox(
        "Input Understat Player Name and ID:", 
        options=sorted_options,
        index=0
    )
    player_id = player_dict[selected_display]
except Exception as e:
    st.error(f"Error loading player list: {str(e)}")
    st.stop()

try:
    available_seasons = sorted(df['Season'].unique())
    min_season = min(available_seasons)
    max_season = max(available_seasons)
    
    start_year, end_year = st.sidebar.select_slider(
         "Select Range of Seasons:",
         options=available_seasons,
         value=(min_season, max_season)
    )
except Exception as e:
    st.error(f"Error with season selection: {str(e)}")
    start_year, end_year = 2014, 2024

try:
    if 'Filter Game_State' in df.columns:
        available_game_states = df['Filter Game_State'].dropna().unique().tolist()
    else:
        available_game_states = []
    
    if len(available_game_states) > 0:
        filter_gs = st.sidebar.multiselect(
            "Filter for Game States:",
            options=available_game_states,
            default=available_game_states
        )
    else:
        filter_gs = []
        st.sidebar.warning("No game state data available")
except Exception as e:
    st.error(f"Error loading game states: {str(e)}")
    filter_gs = []

player_col = st.sidebar.color_picker('Pick A Color:', '#EA2304')
theme = st.sidebar.radio(
     "Visualisation Theme:",
     ('dark', 'light')
)
goal_alpha = st.sidebar.slider(
    'Change Transparency of Goal Scatter Points:',
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.05
)
only_goals = st.sidebar.checkbox('Show Only Goals')
add_pens = st.sidebar.checkbox('Keep Penalties', value=True)
only_changed_gs = st.sidebar.checkbox('Show Only Game State Altering Moments')

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Advanced Analytics")
show_advanced = st.sidebar.checkbox('Enable Advanced Analytics', value=True)

show_zones = False
show_seasonal = False
show_situations = False
show_metrics_card = False

if show_advanced:
    show_zones = st.sidebar.checkbox('Shot Zone Analysis', value=True)
    show_seasonal = st.sidebar.checkbox('Seasonal Trends', value=True)
    show_situations = st.sidebar.checkbox('Situation Breakdown', value=True)
    show_metrics_card = st.sidebar.checkbox('Performance Dashboard', value=True)

df_fil = df_fil.copy()
df_fil = df_fil[df_fil['Player ID'].astype(str) == str(player_id)]
df_fil = df_fil[df_fil['Season'].between(start_year, end_year)]

if filter_gs:
    if 'Filter Game_State' in df_fil.columns:
        df_fil = df_fil[df_fil['Filter Game_State'].isin(filter_gs)]

if only_goals:
    if 'Shot Result' in df_fil.columns:
        df_fil = df_fil[df_fil['Shot Result'] == 'Goal']

if not add_pens:
    if 'Situation' in df_fil.columns:
        df_fil = df_fil[df_fil['Situation'] != 'Penalty']

if only_changed_gs:
    if 'Changed Game State' in df_fil.columns:
        df_fil = df_fil[df_fil['Changed Game State'] == True]

def generate_gb_df():
    if len(df_fil) == 0:
        return pd.DataFrame()
    
    try:
        required_cols = ['Player ID', 'x', 'y', 'Shot xG', 'isGoal', 'Filter Game_State']
        if not all(col in df_fil.columns for col in required_cols):
            return pd.DataFrame()
        
        df_gb = df_fil[required_cols].groupby("Filter Game_State").agg(['sum', 'count'])
        
        if len(df_gb) == 0:
            return pd.DataFrame()
        
        df_gb['Distance'] = df_gb.apply(
            lambda x: find_distance(x['x']['sum']/x['x']['count'], x['y']['sum']/x['y']['count']), 
            axis=1
        )
        
        df_gb = df_gb.drop([('Player ID', 'sum'), ('Shot xG', 'count'), ('isGoal', 'count'),
                        ('x', 'count'), ('x', 'sum'), ('y', 'count'), ('y', 'sum')], axis=1)
        df_gb.columns = ['Total Shots', 'Total xG', 'Total Goals', 'Distance From Goal']
        df_gb['xG per Shot'] = df_gb['Total xG']/df_gb['Total Shots']
        df_gb['xG Overperformance'] = df_gb['Total Goals'] - df_gb['Total xG']
        df_gb['xG O/P per Shot'] = df_gb['xG Overperformance']/df_gb['Total Shots']
        
        return df_gb
    except Exception as e:
        st.error(f"Error generating game state breakdown: {str(e)}")
        return pd.DataFrame()

def convert_df(df):
    return df.to_csv().encode('utf-8-sig')

if len(df_fil) > 0:

    player_name = df_fil['Player'].iloc[0] if 'Player' in df_fil.columns else "Unknown Player"
    st.header(f"\n\nPlayer: {player_name}")

    if show_advanced and show_metrics_card:
        metrics = calculate_advanced_metrics(df_fil)
        
        if metrics:
            st.subheader("📊 PERFORMANCE DASHBOARD")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Shots", metrics['total_shots'])
                st.metric("Conversion Rate", f"{metrics['conversion_rate']:.1f}%")
            
            with col2:
                st.metric("Goals Scored", metrics['total_goals'])
                st.metric("xG per Shot", f"{metrics['xg_per_shot']:.3f}")
            
            with col3:
                st.metric("Total xG", f"{metrics['total_xg']:.2f}")
                xg_eff_ratio = metrics['xg_efficiency'] / 100
                st.metric("xG Efficiency", f"{xg_eff_ratio:.2f}x")
            
            with col4:
                st.metric("xG Overperformance", f"{metrics['xg_overperformance']:+.2f}")
                st.metric("Shot Consistency", f"{metrics['shot_consistency']:.3f}")
            
            with st.expander("📈 Performance Insights", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Finishing Analysis:**")
                    if metrics['conversion_rate'] > 15:
                        st.success("🔥 Elite conversion rate")
                    elif metrics['conversion_rate'] > 10:
                        st.info("✅ Strong finishing")
                    else:
                        st.warning("⚠️ Room for improvement")
                    
                    st.write("**Shot Selection:**")
                    if metrics['xg_per_shot'] > 0.15:
                        st.success("🎯 Excellent shot quality")
                    elif metrics['xg_per_shot'] > 0.10:
                        st.info("✅ Good positioning")
                    else:
                        st.warning("📍 Lower quality chances")
                
                with col_b:
                    st.write("**xG Performance:**")
                    if metrics['xg_overperformance'] > 2:
                        st.success("⭐ Significantly outperforming xG")
                    elif metrics['xg_overperformance'] > 0:
                        st.info("📈 Performing above expected")
                    elif metrics['xg_overperformance'] > -2:
                        st.warning("📉 Slightly underperforming")
                    else:
                        st.error("⚠️ Well below expected goals")
            
            st.markdown("---")
    st.subheader("\nPLAYER SHOT BY GAMESTATE COMPARISON")
    
    gb_df = generate_gb_df()
    if not gb_df.empty:
        st.table(gb_df)
    else:
        st.info("No game state data available for current filters")
    if show_advanced and show_zones:
        st.subheader("🎯 SHOT ZONE ANALYSIS")
        zone_df = analyze_shot_zones(df_fil)
        
        if not zone_df.empty and len(zone_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(zone_df, use_container_width=True)
            
            with col2:
                st.write("**Zone Insights:**")
                
                try:
                    best_zone_idx = zone_df['xG Diff'].idxmax()
                    worst_zone_idx = zone_df['xG Diff'].idxmin()
                    most_shots_idx = zone_df['Shots'].idxmax()
                    
                    st.write(f"🔥 **Best finishing:** {best_zone_idx}")
                    st.write(f"   • +{zone_df.loc[best_zone_idx, 'xG Diff']:.2f} vs xG")
                    st.write(f"📍 **Most active:** {most_shots_idx}")
                    st.write(f"   • {int(zone_df.loc[most_shots_idx, 'Shots'])} shots")
                    
                    if zone_df.loc[worst_zone_idx, 'xG Diff'] < -0.5:
                        st.write(f"⚠️ **Needs work:** {worst_zone_idx}")
                        st.write(f"   • {zone_df.loc[worst_zone_idx, 'xG Diff']:.2f} vs xG")
                except Exception as e:
                    st.warning("Unable to generate zone insights")
        else:
            st.info("Not enough data for zone analysis")
        
        st.markdown("---")
    if show_advanced and show_seasonal and (end_year - start_year > 0):
        st.subheader("📅 SEASONAL PERFORMANCE TRENDS")
        seasonal_df = seasonal_performance(df_fil)
        
        if not seasonal_df.empty and len(seasonal_df) > 1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                    
                    seasons = seasonal_df.index.tolist()
                    ax1.plot(seasons, seasonal_df['Goals'].values, marker='o', label='Goals', 
                            linewidth=2, color='#EA2304')
                    ax1.plot(seasons, seasonal_df['Total xG'].values, marker='s', label='xG', 
                            linewidth=2, linestyle='--', color='#4A90E2')
                    ax1.set_ylabel('Count', fontsize=10)
                    ax1.set_title('Goals vs Expected Goals', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(seasons, seasonal_df['Conversion %'].values, marker='o', 
                            linewidth=2, color='#2ECC71')
                    ax2.set_xlabel('Season', fontsize=10)
                    ax2.set_ylabel('Conversion %', fontsize=10)
                    ax2.set_title('Conversion Rate Trend', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error creating seasonal chart: {str(e)}")
            
            with col2:
                st.write("**Trend Summary:**")
                
                try:
                    if len(seasonal_df) >= 3:
                        recent_conv = seasonal_df['Conversion %'].iloc[-2:].mean()
                        earlier_conv = seasonal_df['Conversion %'].iloc[:2].mean()
                        
                        if recent_conv > earlier_conv + 2:
                            st.success("📈 Improving finishing")
                        elif recent_conv < earlier_conv - 2:
                            st.warning("📉 Declining efficiency")
                        else:
                            st.info("➡️ Consistent performance")
                    
                    best_season_idx = seasonal_df['Goals'].idxmax()
                    best_season = seasonal_df.loc[best_season_idx]
                    st.write(f"**Best season:** {best_season_idx}")
                    st.write(f"• {int(best_season['Goals'])} goals")
                    st.write(f"• {best_season['Conversion %']:.1f}% conversion")
                except Exception as e:
                    st.warning("Unable to generate trend summary")
            
            with st.expander("📊 View Detailed Seasonal Stats"):
                st.dataframe(seasonal_df, use_container_width=True)
        else:
            st.info("Select multiple seasons to view trends")
        
        st.markdown("---")
    if show_advanced and show_situations:
        st.subheader("⚽ SHOT SITUATION BREAKDOWN")
        situation_df = get_situation_breakdown(df_fil)
        
        if not situation_df.empty and len(situation_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(situation_df, use_container_width=True)

            with col2:
                try:

                    pie_data = pd.DataFrame({
                        'Situation': situation_df.index.tolist(),
                        'Shots': situation_df['Shots'].values
                    })
                    
                    fig = px.pie(
                        pie_data, 
                        values='Shots', 
                        names='Situation',
                        title='Shots by Situation',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    # Customize appearance
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont_size=10
                    )
                    
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.02
                        ),
                        height=400,
                        width=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error creating pie chart: {str(e)}")
                    
            st.write("**Situation Insights:**")
            try:
                if len(situation_df) > 0:
                    best_situation_idx = situation_df['Conversion %'].idxmax()
                    best_conv = situation_df.loc[best_situation_idx, 'Conversion %']
                    st.write(f"🎯 Most clinical in **{best_situation_idx}** situations ({best_conv:.1f}% conversion)")
            except Exception as e:
                st.warning("Unable to generate situation insights")
        else:
            st.info("Situation data not available")
    
    if st.button('🎨 Generate Visualization', type='primary', use_container_width=True):
        st.header("PLAYER SHOT DATAFRAME")

        cols_to_drop = ['x','y', 'h_a', 'isGoal', 'Player ID', 'Filter Game_State']
        existing_cols = [col for col in cols_to_drop if col in df_fil.columns]
        display_df = df_fil.drop(existing_cols, axis=1)
        st.dataframe(display_df.reset_index(drop=True), width=2000, height=250)
        
        df_fil_save = convert_df(df_fil)
        
        st.download_button(
            label="📥 Download Shot DataFrame as CSV",
            data=df_fil_save,
            file_name='PlayerShots.csv',
            mime='text/csv',
        )

        theme_dict = {'light':'white', 'dark':'black'}
        st.header("\nPLAYER SHOT MAP")

        with st.spinner('⏳ Generating plot...'):
            try:
                fig, ax = plot_pitch(df_fil, theme=theme, player_col=player_col, alpha=goal_alpha)
                st.pyplot(fig)

                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, facecolor=theme_dict[theme], bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                
                st.download_button(
                    label="📥 Download Shot Map (PNG)",
                    data=buf,
                    file_name="PlayerShots.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating shot map: {str(e)}")

else:
    st.error("Sorry no entries with these inputs, please change your inputs")
    st.info("Try selecting a different player or adjusting the season range")

st.markdown("""
### GUIDE TO THE WIDGET ELEMENTS

1. Player ID – Player name and Understat ID  
2. Range of Season – Select relevant seasons  
3. Filter Game State – Filter shots by match state  
4. Show Only Goals – Display goals only  
5. Keep Penalties – Include or exclude penalties  
6. Game State Altering Moments – Show goals that changed the game state  
7. Pick a Colour – Choose goal marker colour  
8. Visualisation Theme – Light or dark theme  
9. Change Transparency – Adjust goal marker transparency  

---

### GAME STATE DEFINITIONS

- **Winning / Losing:** Team is ahead or behind, but goal difference is unspecified.  
- **Winning by 1 / Losing by 1:** Exactly one-goal margin.  
- **Winning by More Than 1 / Losing by More Than 1:** Two or more goals ahead or behind.  

⚠️ *Generic states are analytically distinct from margin-specific states and should not be directly compared.*

---

### 🆕 ADVANCED ANALYTICS

**Performance Dashboard:** Conversion rate, xG efficiency, and performance insights  

**Shot Zone Analysis:** Performance by shot location and pitch zone  

**Seasonal Trends:** Performance changes across seasons  

**Situation Breakdown:** Analysis by shot situation (open play, set-piece, counter, etc.)
""")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
