import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="⚽ Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    try:
        with open('football_predictor_with_odds.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Make sure 'football_predictor_with_odds.pkl' is uploaded.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('football_data_with_odds.csv', low_memory=False)
        df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Make sure 'football_data_with_odds.csv' is uploaded.")
        st.stop()

model_data = load_model()
model = model_data['model']
home_encoder = model_data['home_encoder']
away_encoder = model_data['away_encoder']
feature_columns = model_data['feature_columns']
accuracy = model_data.get('accuracy', 0)

df = load_data()
available_teams = sorted(list(home_encoder.classes_))

# Helper functions
def get_team_stats(team_name):
    team_matches = df[
        (df['HomeTeam'] == team_name) | 
        (df['AwayTeam'] == team_name)
    ].tail(10)
    
    if len(team_matches) == 0:
        return {'avg_goals': 1.5, 'avg_conceded': 1.5, 'win_rate': 0.33, 'matches': 0}
    
    goals, conceded, wins = [], [], 0
    for _, m in team_matches.iterrows():
        if m['HomeTeam'] == team_name:
            g, c = m['FTHG'], m['FTAG']
            if m['FTR'] == 'H': wins += 1
        else:
            g, c = m['FTAG'], m['FTHG']
            if m['FTR'] == 'A': wins += 1
        goals.append(g)
        conceded.append(c)
    
    return {
        'avg_goals': np.mean(goals),
        'avg_conceded': np.mean(conceded),
        'win_rate': wins / len(team_matches),
        'matches': len(team_matches)
    }

def predict_match(home_team, away_team, odds_home=None, odds_draw=None, odds_away=None):
    if home_team not in available_teams or away_team not in available_teams:
        return None
    
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    
    features = {
        'HomeTeam_encoded': home_encoder.transform([home_team])[0],
        'AwayTeam_encoded': away_encoder.transform([away_team])[0],
        'home_goals': home_stats['avg_goals'],
        'home_conceded': home_stats['avg_conceded'],
        'home_win_rate': home_stats['win_rate'],
        'away_goals': away_stats['avg_goals'],
        'away_conceded': away_stats['avg_conceded'],
        'away_win_rate': away_stats['win_rate']
    }
    
    using_odds = False
    if odds_home and odds_draw and odds_away:
        using_odds = True
        prob_h = 1 / odds_home
        prob_d = 1 / odds_draw
        prob_a = 1 / odds_away
        total = prob_h + prob_d + prob_a
        
        features['prob_home_norm'] = prob_h / total
        features['prob_draw_norm'] = prob_d / total
        features['prob_away_norm'] = prob_a / total
        
        max_prob = max(prob_h/total, prob_d/total, prob_a/total)
        features['favorite'] = 1 if prob_h/total == max_prob else (2 if prob_a/total > prob_d/total else 0)
        
        probs = sorted([prob_h/total, prob_d/total, prob_a/total], reverse=True)
        features['market_confidence'] = probs[0] - probs[1]
    else:
        features.update({
            'prob_home_norm': 0.45, 'prob_draw_norm': 0.27,
            'prob_away_norm': 0.28, 'favorite': 1, 'market_confidence': 0.18
        })
    
    features['shots_ratio'] = 1.0
    features['total_shots'] = 20.0
    
    X = pd.DataFrame([features], columns=feature_columns)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    
    return {
        'prediction': prediction,
        'home_prob': prob_dict.get('H', 0) * 100,
        'draw_prob': prob_dict.get('D', 0) * 100,
        'away_prob': prob_dict.get('A', 0) * 100,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'using_odds': using_odds
    }

# Main App
st.markdown('<div class="main-header">⚽ Football Match Predictor</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.info(f"🎯 **Model Accuracy: {accuracy:.1%}** | 🤖 Professional AI Predictions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"📊 **Teams Available:** {len(available_teams)}")
    st.write(f"📈 **Training Matches:** 5,320")
    st.write(f"🗓️ **Data Period:** 2000-2017")
    
    st.markdown("---")
    st.subheader("💡 Tips")
    st.write("• Add betting odds for better accuracy")
    st.write("• Check both teams are spelled correctly")
    st.write("• High confidence = 55%+ probability")
    
    st.markdown("---")
    st.subheader("📞 About")
    st.write("Professional football prediction using machine learning trained on historical match data and betting odds.")

# Main prediction interface
st.header("🎮 Make Your Prediction")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Home Team",
        options=available_teams,
        index=0
    )

with col2:
    away_team = st.selectbox(
        "✈️ Away Team",
        options=available_teams,
        index=1 if len(available_teams) > 1 else 0
    )

# Odds input
st.subheader("💰 Betting Odds (Optional - Improves Accuracy)")
use_odds = st.checkbox("I have current betting odds")

odds_h, odds_d, odds_a = None, None, None

if use_odds:
    col1, col2, col3 = st.columns(3)
    with col1:
        odds_h = st.number_input("Home Win Odds", min_value=1.01, value=2.10, step=0.1)
    with col2:
        odds_d = st.number_input("Draw Odds", min_value=1.01, value=3.40, step=0.1)
    with col3:
        odds_a = st.number_input("Away Win Odds", min_value=1.01, value=3.60, step=0.1)

# Predict button
if st.button("🔮 Predict Match", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("❌ Please select different teams!")
    else:
        with st.spinner("🔍 Analyzing match..."):
            result = predict_match(home_team, away_team, odds_h, odds_d, odds_a)
            
            if result:
                # Prediction result
                result_map = {'H': f'🏠 {home_team} Win', 'D': '🤝 Draw', 'A': f'✈️ {away_team} Win'}
                
                st.markdown(f'<div class="prediction-box">🎯 {result_map[result["prediction"]]}</div>', 
                           unsafe_allow_html=True)
                
                # Probabilities chart
                st.subheader("📊 Win Probabilities")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[home_team, 'Draw', away_team],
                        y=[result['home_prob'], result['draw_prob'], result['away_prob']],
                        marker_color=['#1f77b4', '#7f7f7f', '#ff7f0e'],
                        text=[f"{result['home_prob']:.1f}%", 
                              f"{result['draw_prob']:.1f}%", 
                              f"{result['away_prob']:.1f}%"],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Team statistics
                st.subheader("📈 Team Form (Last 10 Matches)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 🏠 {home_team}")
                    st.markdown(f"""
                    <div class="stat-box">
                    <b>⚽ Goals per match:</b> {result['home_stats']['avg_goals']:.2f}<br>
                    <b>🥅 Conceded per match:</b> {result['home_stats']['avg_conceded']:.2f}<br>
                    <b>📊 Win rate:</b> {result['home_stats']['win_rate']*100:.0f}%<br>
                    <b>🎮 Recent matches:</b> {result['home_stats']['matches']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"### ✈️ {away_team}")
                    st.markdown(f"""
                    <div class="stat-box">
                    <b>⚽ Goals per match:</b> {result['away_stats']['avg_goals']:.2f}<br>
                    <b>🥅 Conceded per match:</b> {result['away_stats']['avg_conceded']:.2f}<br>
                    <b>📊 Win rate:</b> {result['away_stats']['win_rate']*100:.0f}%<br>
                    <b>🎮 Recent matches:</b> {result['away_stats']['matches']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence indicator
                max_prob = max(result['home_prob'], result['draw_prob'], result['away_prob'])
                
                if max_prob >= 55:
                    confidence = "HIGH 🔥"
                    color = "green"
                    advice = "Strong prediction - High confidence"
                elif max_prob >= 45:
                    confidence = "MEDIUM ⚖️"
                    color = "orange"
                    advice = "Moderate confidence - Proceed with caution"
                else:
                    confidence = "LOW ⚠️"
                    color = "red"
                    advice = "Low confidence - Too close to call"
                
                st.markdown("---")
                st.subheader("💡 Recommendation")
                
                if result['using_odds']:
                    st.success(f"✅ Using betting odds - Enhanced accuracy!")
                else:
                    st.info(f"💡 Add betting odds for better accuracy (+5-7%)")
                
                st.markdown(f"""
                <div style="padding: 1.5rem; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color};">
                <h3 style="color: {color}; margin-top: 0;">Confidence: {confidence}</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Error making prediction. Please check team names.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>⚽ Football Match Predictor</b> | Powered by Machine Learning</p>
    <p>Model trained on 5,320+ matches with 57.4% accuracy</p>
    <p style="font-size: 0.9rem;">For entertainment purposes only. Past performance doesn't guarantee future results.</p>
</div>
""", unsafe_allow_html=True)