import os
import pickle
import random
import time
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import shutil
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Paths & Resources
# -----------------------------
VECTOR_PATH = "vectorizer.pkl"
MODEL_PATH = "fake_news_model.pkl"

# Check model files exist
if not os.path.exists(VECTOR_PATH) or not os.path.exists(MODEL_PATH):
    st.error("🚨 Model files not found! Please ensure `vectorizer.pkl` and `fake_news_model.pkl` are in the application directory.")
    st.stop()

@st.cache_resource
def load_model():
    with open(VECTOR_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# -----------------------------
# Variables
# -----------------------------
CLASS_LABELS = {0: "FAKE", 1: "REAL"}
COLOR_MAP = {"FAKE": "#ff4b4b", "REAL": "#00d26a"}

# Headline pools
EASY_HEADLINES = [
    "Breaking: You won't believe what happened in the USA!!!",
    "India announces new AI innovation.",
    "Shocking: Alien life discovered on Mars!",
    "Germany economy steady amid challenges.",
    "Unbelievable: China develops invisible drones.",
    "Scientists confirm water found on Moon.",
    "Experts reveal AI can write novels indistinguishable from humans.",
    "Unbelievable: Person claims to time travel using dreams."
]

MEDIUM_HEADLINES = [
    "Government announces new policy on digital privacy.",
    "Stock market reaches all-time high amid economic recovery.",
    "New study shows coffee reduces risk of heart disease.",
    "Celebrity couple announces surprise divorce.",
    "Local hero saves child from burning building.",
    "Tech giant unveils revolutionary smartphone.",
    "Election results expected later tonight.",
    "Hurricane warning issued for coastal regions."
]

HARD_HEADLINES = [
    "Researchers discover new species in Amazon rainforest.",
    "Controversial law passes by narrow margin.",
    "International summit ends with historic agreement.",
    "Company recalls popular product due to safety concerns.",
    "Archaeologists find ancient tomb in Egypt.",
    "Space mission successfully lands on Mars.",
    "Economic experts predict recession next year.",
    "Health officials warn of new virus variant."
]

EXPERT_HEADLINES = [
    "Study finds no link between vaccines and autism, yet debate continues.",
    "Federal reserve hints at interest rate hike in Q3.",
    "Satirical news site misleads readers with fake headline.",
    "Deepfake video of politician circulates online.",
    "Misleading headline uses out-of-context quote.",
    "Article uses sensational language to describe routine event.",
    "Headline contradicts content of the article.",
    "Fake expert quoted in health advice column."
]

ALL_HEADLINES = EASY_HEADLINES + MEDIUM_HEADLINES + HARD_HEADLINES + EXPERT_HEADLINES

HINTS = [
    "🔍 Check unusual words!",
    "🎯 Pattern seems suspicious!",
    "🤖 ML model signals anomaly!",
    "⚠️ Heuristic detects clickbait!"
]

LEADERBOARD_FILE = "leaderboard.json"
ACHIEVEMENTS_FILE = "achievements.json"
BACKUP_DIR = "backups"

os.makedirs(BACKUP_DIR, exist_ok=True)

# -----------------------------
# Achievements List (115 total)
# -----------------------------
ACHIEVEMENTS = []

# 1–10: Novice to Guru
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"correct_{i*10}",
        "name": f"{i*10} Correct Answers",
        "desc": f"Correctly identify {i*10} headlines.",
        "icon": "✅",
        "max_progress": i*10
    })

# 11–20: Streak master
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"streak_{i*5}",
        "name": f"Streak of {i*5}",
        "desc": f"Get {i*5} correct answers in a row.",
        "icon": "🔥",
        "max_progress": i*5
    })

# 21–30: Speed demon
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"speed_{i}",
        "name": f"Speed Level {i}",
        "desc": f"Answer {i*5} headlines in under 3 seconds each.",
        "icon": "⚡",
        "max_progress": i*5
    })

# 31–40: Monster slayer
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"monster_{i}",
        "name": f"Monster Slayer {i}",
        "desc": f"Survive {i} monster rounds.",
        "icon": "👹",
        "max_progress": i
    })

# 41–50: Perfect scores
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"perfect_{i}",
        "name": f"Perfect Round {i}",
        "desc": f"Score 100% on a game {i} times.",
        "icon": "🎯",
        "max_progress": i
    })

# 51–60: Game master
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"games_{i}",
        "name": f"Game Master {i}",
        "desc": f"Play {i*10} games.",
        "icon": "🎮",
        "max_progress": i*10
    })

# 61–70: Category expert
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"category_{i}",
        "name": f"Category Expert {i}",
        "desc": f"Correctly identify {i*10} headlines in a single category.",
        "icon": "📚",
        "max_progress": i*10
    })

# 71–80: Comeback kid
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"comeback_{i}",
        "name": f"Comeback Kid {i}",
        "desc": f"Get {i*5} correct after a wrong answer.",
        "icon": "🔄",
        "max_progress": i*5
    })

# 81–90: Accuracy ace
for i in range(1, 11):
    ACHIEVEMENTS.append({
        "id": f"accuracy_{i}",
        "name": f"Accuracy Ace {i}",
        "desc": f"Achieve {i*10}% accuracy over 20+ headlines.",
        "icon": "📊",
        "max_progress": i*10
    })

# 91–100: Ultra rare
rare_names = ["Legend", "Myth", "Immortal", "Unstoppable", "Omniscient",
              "Fact Checker Pro", "Truth Seeker", "Fake Buster", "News Wizard", "AI Whisperer"]
for i, name in enumerate(rare_names, 1):
    ACHIEVEMENTS.append({
        "id": f"rare_{i}",
        "name": name,
        "desc": f"Unlock the {name} achievement by doing something legendary!",
        "icon": "🏆",
        "max_progress": 1
    })

# 15 New Player-Status Achievements
ACHIEVEMENTS.extend([
    {
        "id": "collector",
        "name": "Collector",
        "desc": "Unlock 10 achievements.",
        "icon": "🏷️",
        "max_progress": 10
    },
    {
        "id": "completionist",
        "name": "Completionist",
        "desc": "Unlock all achievements.",
        "icon": "🎯",
        "max_progress": len(ACHIEVEMENTS) + 15
    },
    {
        "id": "speedrunner",
        "name": "Speedrunner",
        "desc": "Finish a game in under 2 minutes.",
        "icon": "⏱️",
        "max_progress": 1
    },
    {
        "id": "perfectionist",
        "name": "Perfectionist",
        "desc": "Achieve a perfect score (100%) in any game mode.",
        "icon": "🎯",
        "max_progress": 1
    },
    {
        "id": "grinder",
        "name": "Grinder",
        "desc": "Play 100 games.",
        "icon": "⚙️",
        "max_progress": 100
    },
    {
        "id": "casual",
        "name": "Casual",
        "desc": "Play fewer than 10 games (status, not an achievement).",
        "icon": "🛋️",
        "max_progress": 1,
        "hidden": True
    },
    {
        "id": "hardcore",
        "name": "Hardcore",
        "desc": "Play 10 games on hard mode.",
        "icon": "🔥",
        "max_progress": 10
    },
    {
        "id": "newbie",
        "name": "Newbie",
        "desc": "Play your first game.",
        "icon": "🐣",
        "max_progress": 1
    },
    {
        "id": "veteran",
        "name": "Veteran",
        "desc": "Play 500 games.",
        "icon": "🧓",
        "max_progress": 500
    },
    {
        "id": "legend",
        "name": "Legend",
        "desc": "Reach rank 1 on the leaderboard.",
        "icon": "🏆",
        "max_progress": 1
    },
    {
        "id": "myth",
        "name": "Myth",
        "desc": "Unlock all achievements (including these).",
        "icon": "🧙",
        "max_progress": len(ACHIEVEMENTS) + 15
    },
    {
        "id": "immortal",
        "name": "Immortal",
        "desc": "Complete every game mode without a single wrong answer.",
        "icon": "🧛",
        "max_progress": 1
    },
    {
        "id": "unstoppable",
        "name": "Unstoppable",
        "desc": "Achieve a 100% win rate over 20 games.",
        "icon": "🦸",
        "max_progress": 1
    },
    {
        "id": "omniscient",
        "name": "Omniscient",
        "desc": "Predict AI confidence within 5% 10 times.",
        "icon": "🔮",
        "max_progress": 10
    },
    {
        "id": "ai_whisperer",
        "name": "AI Whisperer",
        "desc": "Predict AI confidence exactly 5 times.",
        "icon": "🤖",
        "max_progress": 5
    }
])

# -----------------------------
# Modern Styles (CSS) – responsive & robust
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    .big-title {
        font-size: 3.5em;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.3em;
        font-weight: 300;
        margin-bottom: 30px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        font-size: 1.1em;
        padding: 15px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .prediction-box {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    .prediction-box.fake { border-left-color: #ff4b4b; background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%); }
    .prediction-box.real { border-left-color: #00d26a; background: linear-gradient(135deg, #f0fff4 0%, #d4f4dd 100%); }
    .prediction-label { font-size: 2em; font-weight: 700; margin-bottom: 10px; }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: #f0f0f0;
        overflow: hidden;
        margin: 15px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 1s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.9em;
    }
    .confidence-fill.fake { background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%); }
    .confidence-fill.real { background: linear-gradient(90deg, #00d26a 0%, #00f280 100%); }
    span.suspicious {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        cursor: help;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(255, 75, 75, 0.3);
    }
    span.suspicious:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.5);
    }
    .reasoning-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #667eea;
    }
    .reasoning-item {
        padding: 10px;
        margin: 8px 0;
        background: white;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
    }
    .reasoning-item:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    @keyframes monster-pulse {
        0%, 100% { box-shadow: 0 0 20px #ff0000, 0 0 40px #ff0000; border-color: #ff0000; }
        50% { box-shadow: 0 0 40px #ff0000, 0 0 80px #ff0000; border-color: #ff3333; }
    }
    .monster-active {
        border: 4px solid #ff0000;
        padding: 25px;
        border-radius: 20px;
        animation: monster-pulse 1.5s infinite;
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.1) 0%, rgba(255, 50, 50, 0.1) 100%);
        position: relative;
    }
    .monster-badge {
        position: absolute;
        top: -15px;
        right: 20px;
        background: linear-gradient(135deg, #ff0000 0%, #ff3333 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9em;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.4);
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .score-number { font-size: 3em; font-weight: 800; margin: 10px 0; }
    .score-label { font-size: 1em; opacity: 0.9; font-weight: 300; }
    .timer-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    .timer-number { font-size: 3em; font-weight: 800; margin: 10px 0; }
    .leaderboard-item {
        background: white;
        padding: 20px;
        margin: 10px 0;
        border-radius: 12px;
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
    .leaderboard-item:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .leaderboard-rank {
        font-size: 2em;
        font-weight: 800;
        margin-right: 20px;
        width: 60px;
        text-align: center;
    }
    .leaderboard-rank.gold { color: #FFD700; }
    .leaderboard-rank.silver { color: #C0C0C0; }
    .leaderboard-rank.bronze { color: #CD7F32; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
    }
    .dataframe { border-radius: 12px; overflow: hidden; }
    .stAlert { border-radius: 12px; border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .stMetric { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    
    /* Responsive design for mobile */
    @media only screen and (max-width: 600px) {
        .stButton > button {
            padding: 8px 16px;
            font-size: 0.9em;
        }
        .main-card {
            padding: 15px;
        }
        .big-title {
            font-size: 2.5em;
        }
        .subtitle {
            font-size: 1.1em;
        }
        .leaderboard-rank {
            font-size: 1.5em;
            margin-right: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Improved Helper Functions
# -----------------------------

@st.cache_data(ttl=3600)
def cached_analyze(text):
    """Cached version for repeated headlines (e.g., game pools)."""
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob >= 0.5 else 0
    return CLASS_LABELS[pred], prob

def analyze_text(text):
    """Robust analysis with user-friendly error handling."""
    try:
        text = str(text).strip()
        if not text:
            st.warning("⚠️ Please enter some text to analyze.")
            return None, None
        if len(text) < 5:
            st.warning("⚠️ Text is too short. Please enter at least 5 characters.")
            return None, None
        if len(text) > 1000:
            text = text[:1000]
            st.info("ℹ️ Text truncated to 1000 characters for performance.")
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]
        pred = 1 if prob >= 0.5 else 0
        return CLASS_LABELS[pred], prob
    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        return None, None

def explain_fake(text, top_n=5):
    try:
        X = vectorizer.transform([text])
        coef = model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        indices = X.nonzero()[1]
        word_scores = {feature_names[i]: coef[i]*X[0,i] for i in indices}
        top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return [w for w,s in top_words if s<0]
    except:
        return []

def highlight_suspicious(text):
    ml_words = explain_fake(text)
    def repl(match):
        word = match.group(0)
        if word.lower() in [w.lower() for w in ml_words]:
            return f"<span class='suspicious' title='ML signal: contributes to FAKE'>{word}</span>"
        return word
    return re.sub(r'\b\w+\b', repl, text, flags=re.IGNORECASE)

def explain_reasoning(text, top_n=5):
    reasons = []
    try:
        X = vectorizer.transform([text])
        if hasattr(model,"coef_"):
            coef = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            indices = X.nonzero()[1]
            word_scores = {feature_names[i]: coef[i]*X[0,i] for i in indices}
            top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            for word, score in top_words:
                if score < 0:
                    reasons.append(f"🔴 ML indicates '{word}' contributes to FAKE")
                else:
                    reasons.append(f"🟢 ML indicates '{word}' contributes to REAL")
    except:
        pass
    if "!!!" in text or text.isupper():
        reasons.append("⚠️ Heuristic: Excessive punctuation or all-caps detected")
    clickbait_words = ["shocking","unbelievable","you won't believe"]
    for w in clickbait_words:
        if w.lower() in text.lower():
            reasons.append(f"🎯 Heuristic: Clickbait word detected '{w}'")
    return reasons

# -----------------------------
# JSON backup functions
# -----------------------------
def backup_json(file_path):
    """Create a timestamped backup of a JSON file."""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = os.path.join(BACKUP_DIR, f"{os.path.basename(file_path)}.{timestamp}.bak")
        shutil.copy(file_path, backup_name)
        # Keep only last 5 backups to avoid clutter
        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith(os.path.basename(file_path))])
        for old in backups[:-5]:
            os.remove(os.path.join(BACKUP_DIR, old))

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE,"r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_leaderboard(board):
    backup_json(LEADERBOARD_FILE)
    with open(LEADERBOARD_FILE,"w") as f:
        json.dump(board, f, indent=2)

# ==================== FIXED load_achievements FUNCTION ====================
def load_achievements(player_name):
    """Load achievements for a player, ensuring all achievements exist."""
    if os.path.exists(ACHIEVEMENTS_FILE):
        with open(ACHIEVEMENTS_FILE, "r") as f:
            all_achievements = json.load(f)
    else:
        all_achievements = {}
    
    # Ensure player entry exists
    if player_name not in all_achievements:
        all_achievements[player_name] = {}
    
    player_data = all_achievements[player_name]
    modified = False
    
    # Add any missing achievements from the master list
    for ach in ACHIEVEMENTS:
        ach_id = ach["id"]
        if ach_id not in player_data:
            player_data[ach_id] = {
                "unlocked": False,
                "progress": 0,
                "max": ach["max_progress"],
                "unlocked_date": None
            }
            modified = True
        else:
            # Optional: update max value if it changed (rare)
            if player_data[ach_id]["max"] != ach["max_progress"]:
                player_data[ach_id]["max"] = ach["max_progress"]
                modified = True
    
    if modified:
        save_achievements(all_achievements)
    
    return player_data
# ===========================================================================

def save_achievements(all_achievements):
    backup_json(ACHIEVEMENTS_FILE)
    with open(ACHIEVEMENTS_FILE, "w") as f:
        json.dump(all_achievements, f, indent=2)

def update_achievement(player_name, ach_id, increment=1, force_progress=None, skip_collective=False):
    """Update a single achievement and optionally trigger collective checks."""
    if os.path.exists(ACHIEVEMENTS_FILE):
        with open(ACHIEVEMENTS_FILE, "r") as f:
            all_achs = json.load(f)
    else:
        all_achs = {}
    if player_name not in all_achs:
        # If player missing, we'll use load_achievements to populate properly
        load_achievements(player_name)  # this will create entry and save
        with open(ACHIEVEMENTS_FILE, "r") as f:
            all_achs = json.load(f)
    if ach_id in all_achs[player_name]:
        ach_data = all_achs[player_name][ach_id]
        if not ach_data["unlocked"]:
            old_progress = ach_data["progress"]
            if force_progress is not None:
                ach_data["progress"] = force_progress
            else:
                ach_data["progress"] += increment
            # Only save and check collective if progress actually changed
            if ach_data["progress"] != old_progress:
                if ach_data["progress"] >= ach_data["max"]:
                    ach_data["unlocked"] = True
                    ach_data["unlocked_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                save_achievements(all_achs)
                if not skip_collective:
                    check_collective_achievements(player_name, all_achs)
    else:
        # Achievement ID not found – ignore (shouldn't happen)
        pass

def check_collective_achievements(player_name, all_achs=None):
    """Check and update collector, completionist, myth achievements."""
    if all_achs is None:
        with open(ACHIEVEMENTS_FILE, "r") as f:
            all_achs = json.load(f)
    player_achs = all_achs[player_name]
    unlocked_count = sum(1 for a in player_achs.values() if a["unlocked"])
    total_achievements = len(ACHIEVEMENTS)
    # Collector (unlock 10 achievements)
    if not player_achs["collector"]["unlocked"]:
        new_progress = min(unlocked_count, player_achs["collector"]["max"])
        if new_progress > player_achs["collector"]["progress"]:
            update_achievement(player_name, "collector", force_progress=new_progress, skip_collective=True)
    # Completionist (unlock all achievements)
    if not player_achs["completionist"]["unlocked"] and unlocked_count >= total_achievements:
        update_achievement(player_name, "completionist", force_progress=total_achievements, skip_collective=True)
    # Myth (unlock all achievements including these)
    if not player_achs["myth"]["unlocked"] and unlocked_count >= total_achievements:
        update_achievement(player_name, "myth", force_progress=total_achievements, skip_collective=True)

# -----------------------------
# Global helper for correct‑answer achievements
# -----------------------------
def on_correct_answer(player_name):
    """Call this whenever a player answers correctly."""
    for ach in ACHIEVEMENTS:
        if ach["id"].startswith("correct_"):
            update_achievement(player_name, ach["id"], increment=1, skip_collective=False)

# -----------------------------
# Game UI Helpers (refactored)
# -----------------------------
def render_game_header(score, lives=None, time_left=None, headline_num=None, total=None):
    """Common header for game modes."""
    cols = st.columns([1,2,1] if lives is None else [1,1,1])
    with cols[0]:
        st.markdown(f"**Score:** {score}")
    if lives is not None:
        with cols[1]:
            st.markdown(f"**Lives:** {'❤️' * lives}")
    if time_left is not None:
        with cols[2]:
            st.markdown(f"**Time:** {time_left}s")
    if headline_num is not None and total is not None:
        st.progress((headline_num) / total, text=f"Headline {headline_num+1}/{total}")

# -----------------------------
# Session State Reset
# -----------------------------
def reset_game_mode(mode):
    """Clear session state variables for a given game mode."""
    mode_keys = {
        "Mind-Game (Timed)": ["mind_index", "mind_score", "timer_start"],
        "⚡ Speed Round": ["speed_index", "speed_score", "speed_timer_start", "speed_streak"],
        "💀 Survival Mode": ["survival_index", "survival_score", "survival_wrong", "survival_headlines"],
        "🧠 Expert Mode": ["expert_index", "expert_score"],
        "🔄 Swap Mode (62)": ["swap_index", "swap_score", "swap_headlines"],
        "🔍 Zoom In (53)": ["zoom_index", "zoom_score", "zoom_start_time", "zoom_headline", "zoom_pred"],
        "⚔️ Fact-Check Battle (65)": ["battle_index", "battle_player_score", "battle_ai_score", "battle_round", "battle_headlines"],
        "📚 Training Mode (9)": ["training_index", "training_score", "training_headlines", "training_explanation"]
    }
    for key in mode_keys.get(mode, []):
        if key in st.session_state:
            del st.session_state[key]

# -----------------------------
# Session State
# -----------------------------
if "game_mode" not in st.session_state:
    st.session_state.game_mode = "Mind-Game (Timed)"
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""
if "total_correct" not in st.session_state:
    st.session_state.total_correct = 0
if "games_played" not in st.session_state:
    st.session_state.games_played = 0
if "hard_mode_games" not in st.session_state:
    st.session_state.hard_mode_games = 0
if "fastest_game_time" not in st.session_state:
    st.session_state.fastest_game_time = float('inf')
if "perfect_scores" not in st.session_state:
    st.session_state.perfect_scores = 0
if "win_streak" not in st.session_state:
    st.session_state.win_streak = 0
if "total_games_played" not in st.session_state:
    st.session_state.total_games_played = 0
if "player_name" not in st.session_state:
    st.session_state.player_name = "Player"

# Original Mind-Game
if "mind_index" not in st.session_state:
    st.session_state.mind_index = 0
if "mind_score" not in st.session_state:
    st.session_state.mind_score = 0
if "timer_start" not in st.session_state:
    st.session_state.timer_start = time.time()

# Speed Round
if "speed_index" not in st.session_state:
    st.session_state.speed_index = 0
if "speed_score" not in st.session_state:
    st.session_state.speed_score = 0
if "speed_timer_start" not in st.session_state:
    st.session_state.speed_timer_start = time.time()
if "speed_streak" not in st.session_state:
    st.session_state.speed_streak = 0

# Survival
if "survival_index" not in st.session_state:
    st.session_state.survival_index = 0
if "survival_score" not in st.session_state:
    st.session_state.survival_score = 0
if "survival_wrong" not in st.session_state:
    st.session_state.survival_wrong = 0
if "survival_headlines" not in st.session_state:
    st.session_state.survival_headlines = []

# Expert
if "expert_index" not in st.session_state:
    st.session_state.expert_index = 0
if "expert_score" not in st.session_state:
    st.session_state.expert_score = 0

# Swap Mode (62)
if "swap_index" not in st.session_state:
    st.session_state.swap_index = 0
if "swap_score" not in st.session_state:
    st.session_state.swap_score = 0
if "swap_headlines" not in st.session_state:
    st.session_state.swap_headlines = []

# Zoom In Mode (53)
if "zoom_index" not in st.session_state:
    st.session_state.zoom_index = 0
if "zoom_score" not in st.session_state:
    st.session_state.zoom_score = 0
if "zoom_start_time" not in st.session_state:
    st.session_state.zoom_start_time = time.time()
if "zoom_headline" not in st.session_state:
    st.session_state.zoom_headline = ""
if "zoom_pred" not in st.session_state:
    st.session_state.zoom_pred = None

# Fact-Check Battle (65)
if "battle_index" not in st.session_state:
    st.session_state.battle_index = 0
if "battle_player_score" not in st.session_state:
    st.session_state.battle_player_score = 0
if "battle_ai_score" not in st.session_state:
    st.session_state.battle_ai_score = 0
if "battle_round" not in st.session_state:
    st.session_state.battle_round = 0
if "battle_headlines" not in st.session_state:
    st.session_state.battle_headlines = []

# Training Mode (9)
if "training_index" not in st.session_state:
    st.session_state.training_index = 0
if "training_score" not in st.session_state:
    st.session_state.training_score = 0
if "training_headlines" not in st.session_state:
    st.session_state.training_headlines = []
if "training_explanation" not in st.session_state:
    st.session_state.training_explanation = ""

# Accuracy Challenge
if "accuracy_index" not in st.session_state:
    st.session_state.accuracy_index = 0
if "accuracy_score" not in st.session_state:
    st.session_state.accuracy_score = 0
if "accuracy_started" not in st.session_state:
    st.session_state.accuracy_started = False

# Auto Booth
if "auto_index" not in st.session_state:
    st.session_state.auto_index = 0
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False
if "auto_speed" not in st.session_state:
    st.session_state.auto_speed = 3

# -----------------------------
# Header & Sidebar
# -----------------------------
st.markdown("""
<div style='background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px; margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
    <h1 style='color: #667eea; font-size: 3.5em; font-weight: 800; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
        🔍 Fake News Detector AI
    </h1>
    <p style='color: #764ba2; font-size: 1.3em; margin-top: 10px; font-weight: 500;'>
        Powered by Machine Learning • Detect Misinformation in Real-Time
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("**Algorithm:** Logistic Regression\n\n**Features:** TF-IDF Vectorization\n\n**Accuracy:** Trained on thousands of articles")
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    board = load_leaderboard()
    if board:
        top_player = max(board.items(), key=lambda x: x[1]["score"])
        st.success(f"**Top Player**\n\n{top_player[0]}\n\n{top_player[1]['score']} points")
    else:
        st.warning("No records yet!")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("This AI-powered tool uses machine learning to detect fake news by analyzing linguistic patterns, clickbait indicators, and content authenticity markers.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Single News", "📊 CSV/Batch", "🤖 Auto Booth", "🎮 Mind-Game", "🏆 Achievements", "🎯 Accuracy Challenge"])

# -----------------------------
# Single News
# -----------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("### 📰 Analyze News Article")
        news_text = st.text_area("Paste your news headline or article here:", height=200, placeholder="Enter the news text you want to verify...")
        analyze_btn = st.button("🔍 Analyze Now", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Tips")
        st.info("**Look for:**\n- Excessive punctuation (!!!)\n- ALL CAPS words\n- Clickbait phrases\n- Unrealistic claims\n- Emotional language")
        st.markdown("</div>", unsafe_allow_html=True)
    if analyze_btn and news_text.strip():
        with st.spinner("AI is analyzing..."):
            pred, prob = analyze_text(news_text)
            time.sleep(0.5)
        if pred is None:
            st.stop()
        result_class = "fake" if pred == "FAKE" else "real"
        st.markdown(f"""
        <div class='prediction-box {result_class}'>
            <div class='prediction-label' style='color: {COLOR_MAP[pred]};'>
                {'🚫 FAKE NEWS' if pred == 'FAKE' else '✅ REAL NEWS'}
            </div>
            <div style='font-size: 1.2em; margin: 10px 0;'>
                Confidence Level: <strong>{prob*100:.1f}%</strong>
            </div>
            <div class='confidence-bar'>
                <div class='confidence-fill {result_class}' style='width: {prob*100}%;'>
                    {prob*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if pred == "FAKE":
            st.markdown("### 🔍 Suspicious Words Detected")
            st.markdown("<div class='main-card'>", unsafe_allow_html=True)
            st.markdown(highlight_suspicious(news_text), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        reasons = explain_reasoning(news_text)
        if reasons:
            st.markdown("### 🧠 AI Analysis Reasoning")
            st.markdown("<div class='reasoning-box'>", unsafe_allow_html=True)
            for r in reasons:
                st.markdown(f"<div class='reasoning-item'>{r}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CSV/Batch
# -----------------------------
with tab2:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Batch Analysis")
    st.info("Upload a CSV file with a 'text' column containing news articles to analyze multiple items at once.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("❌ CSV must have a 'text' column!")
        else:
            with st.spinner("Analyzing articles..."):
                results = []
                progress_bar = st.progress(0)
                for idx, row in df.iterrows():
                    pred, prob = analyze_text(row['text'])
                    if pred is None:
                        pred = "ERROR"
                        prob = 0.0
                    results.append({
                        "text": row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                        "prediction": pred,
                        "confidence": f"{prob*100:.1f}%" if prob is not None else "N/A"
                    })
                    progress_bar.progress((idx + 1) / len(df))
                df_result = pd.DataFrame(results)
                col1, col2, col3 = st.columns(3)
                fake_count = len(df_result[df_result['prediction'] == 'FAKE'])
                real_count = len(df_result[df_result['prediction'] == 'REAL'])
                error_count = len(df_result[df_result['prediction'] == 'ERROR'])
                with col1:
                    st.metric("Total Articles", len(df_result))
                with col2:
                    st.metric("Fake News", fake_count, delta=None, delta_color="inverse")
                with col3:
                    st.metric("Real News", real_count, delta=None)
                if error_count > 0:
                    st.warning(f"{error_count} articles could not be analyzed.")
                st.markdown("### 📋 Results")
                st.dataframe(df_result, use_container_width=True, height=400)
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results",
                    csv_data,
                    "fake_news_results.csv",
                    "text/csv",
                    use_container_width=True
                )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Auto Booth
# -----------------------------
with tab3:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Automatic News Analysis Demo")
    st.info("Watch the AI automatically analyze pre-loaded headlines in real-time!")
    col1, col2 = st.columns([3, 1])
    with col1:
        speed = st.slider("⚡ Cycle Speed (seconds)", 1, 10, 3)
        st.session_state.auto_speed = speed
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.auto_running:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.auto_running = True
                st.rerun()
        else:
            if st.button("⏸️ Stop", use_container_width=True):
                st.session_state.auto_running = False
                st.rerun()
    if st.session_state.auto_running:
        headline = ALL_HEADLINES[st.session_state.auto_index % len(ALL_HEADLINES)]
        pred, prob = cached_analyze(headline)  # use cached version
        result_class = "fake" if pred == "FAKE" else "real"
        st.markdown(f"""
        <div class='prediction-box {result_class}'>
            <h3>📰 Current Headline:</h3>
            <p style='font-size: 1.2em; margin: 15px 0;'>{headline}</p>
            <div class='prediction-label' style='color: {COLOR_MAP[pred]};'>
                {'🚫 FAKE' if pred == 'FAKE' else '✅ REAL'}
            </div>
            <div class='confidence-bar'>
                <div class='confidence-fill {result_class}' style='width: {prob*100}%;'>
                    {prob*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if pred == "FAKE":
            st.markdown(highlight_suspicious(headline), unsafe_allow_html=True)
        reasons = explain_reasoning(headline)
        if reasons:
            st.markdown("**🧠 Analysis:**")
            for r in reasons:
                st.markdown(f"- {r}")
        time.sleep(speed)
        st.session_state.auto_index += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Mind-Game (with all modes)
# -----------------------------
with tab4:
    if not st.session_state.game_started:
        st.markdown("<div class='main-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("### 🎮 Mind-Game Challenge")
        
        # How to play expander
        with st.expander("📖 How to Play"):
            st.markdown("""
            - **Mind-Game (Timed):** 10 seconds per headline. Click REAL or FAKE before time runs out.
            - **⚡ Speed Round:** 20 headlines, 60 seconds total. Streak bonus every 5 correct.
            - **💀 Survival Mode:** Endless headlines, 3 lives. Difficulty increases.
            - **🧠 Expert Mode:** Subtle headlines – no obvious clickbait.
            - **🔄 Swap Mode (62):** See AI's prediction first. Agree or disagree. Points for catching AI mistakes.
            - **🔍 Zoom In (53):** Headline gradually unblurs. Faster recognition = more points.
            - **⚔️ Fact-Check Battle (65):** Debate an AI opponent. Audience votes decide the winner.
            - **📚 Training Mode (9):** No timer, detailed explanations after every answer.
            """)
        
        st.markdown("Choose your game mode:")
        mode = st.radio(
            "Select Mode",
            [
                "Mind-Game (Timed)",
                "⚡ Speed Round",
                "💀 Survival Mode",
                "🧠 Expert Mode",
                "🔄 Swap Mode (62)",
                "🔍 Zoom In (53)",
                "⚔️ Fact-Check Battle (65)",
                "📚 Training Mode (9)"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.game_mode = mode
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            player_name = st.text_input("Enter your name:", value="Player", placeholder="Your name here...")
            if st.button("🚀 Start Game", use_container_width=True, type="primary"):
                reset_game_mode(mode)  # ensure clean state
                st.session_state.game_started = True
                st.session_state.show_feedback = False
                st.session_state.player_name = player_name
                st.session_state.total_games_played += 1
                # Update achievements: games played
                for ach in ACHIEVEMENTS:
                    if ach["id"].startswith("games_"):
                        update_achievement(player_name, ach["id"], increment=1, skip_collective=False)
                if st.session_state.total_games_played == 1:
                    update_achievement(player_name, "newbie", force_progress=1, skip_collective=False)
                # Other game-start achievements (e.g., hard mode, etc.) can be added later
                if mode == "Mind-Game (Timed)":
                    st.session_state.mind_index = 0
                    st.session_state.mind_score = 0
                    st.session_state.timer_start = time.time()
                elif mode == "⚡ Speed Round":
                    st.session_state.speed_index = 0
                    st.session_state.speed_score = 0
                    st.session_state.speed_timer_start = time.time()
                    st.session_state.speed_streak = 0
                elif mode == "💀 Survival Mode":
                    st.session_state.survival_index = 0
                    st.session_state.survival_score = 0
                    st.session_state.survival_wrong = 0
                    st.session_state.survival_headlines = random.sample(ALL_HEADLINES, min(50, len(ALL_HEADLINES)))
                elif mode == "🧠 Expert Mode":
                    st.session_state.expert_index = 0
                    st.session_state.expert_score = 0
                elif mode == "🔄 Swap Mode (62)":
                    st.session_state.swap_index = 0
                    st.session_state.swap_score = 0
                    st.session_state.swap_headlines = random.sample(ALL_HEADLINES, 10)
                elif mode == "🔍 Zoom In (53)":
                    st.session_state.zoom_index = 0
                    st.session_state.zoom_score = 0
                    st.session_state.zoom_start_time = time.time()
                    st.session_state.zoom_headline = random.choice(ALL_HEADLINES)
                    st.session_state.zoom_pred, _ = cached_analyze(st.session_state.zoom_headline)
                elif mode == "⚔️ Fact-Check Battle (65)":
                    st.session_state.battle_index = 0
                    st.session_state.battle_player_score = 0
                    st.session_state.battle_ai_score = 0
                    st.session_state.battle_round = 0
                    st.session_state.battle_headlines = random.sample(ALL_HEADLINES, 5)
                elif mode == "📚 Training Mode (9)":
                    st.session_state.training_index = 0
                    st.session_state.training_score = 0
                    st.session_state.training_headlines = random.sample(ALL_HEADLINES, 10)
                    st.session_state.training_explanation = ""
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        # Show leaderboard
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Current Leaderboard")
        board = load_leaderboard()
        if board:
            sorted_board = sorted(board.items(), key=lambda x: x[1]["score"], reverse=True)[:10]
            for i, (name, data) in enumerate(sorted_board, 1):
                rank_class = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"
                st.markdown(f"""
                <div class='leaderboard-item'>
                    <div class='leaderboard-rank {rank_class}'>{medal}</div>
                    <div style='flex: 1;'>
                        <div style='font-size: 1.2em; font-weight: 600;'>{name}</div>
                        <div style='font-size: 0.9em; opacity: 0.7;'>{data.get('date', 'Unknown')}</div>
                    </div>
                    <div style='font-size: 1.5em; font-weight: 700; color: #667eea;'>{data['score']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No scores yet. Be the first to play!")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        mode = st.session_state.game_mode
        player_name = st.session_state.player_name

        # ---------- Mind-Game (Timed) ----------
        if mode == "Mind-Game (Timed)":
            @st.fragment(run_every=0.5)
            def game_loop_timed():
                if st.session_state.mind_index < len(EASY_HEADLINES):
                    idx = st.session_state.mind_index
                    current = EASY_HEADLINES[idx % len(EASY_HEADLINES)]
                    pred, prob = cached_analyze(current)
                    time_elapsed = time.time() - st.session_state.timer_start
                    time_left = max(0, 10 - time_elapsed)
                    render_game_header(st.session_state.mind_score, time_left=int(time_left), 
                                       headline_num=idx, total=len(EASY_HEADLINES))
                    st.markdown(f"### {current}")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"timed_real_{idx}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"timed_fake_{idx}"):
                            action = "FAKE"
                    if action or time_left <= 0:
                        if action == pred:
                            st.session_state.mind_score += 1
                            on_correct_answer(player_name)
                            st.success("Correct!")
                        elif action:
                            st.error(f"Wrong! It was {pred}")
                        else:
                            st.warning("Time's up!")
                        st.session_state.mind_index += 1
                        st.session_state.timer_start = time.time()
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.mind_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.mind_score:
                        board[player_name] = {"score": st.session_state.mind_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_timed()

        # ---------- Speed Round ----------
        elif mode == "⚡ Speed Round":
            @st.fragment(run_every=0.5)
            def game_loop_speed():
                total_time = 60
                elapsed = time.time() - st.session_state.speed_timer_start
                time_left = max(0, total_time - elapsed)
                if st.session_state.speed_index < 20 and time_left > 0:
                    idx = st.session_state.speed_index
                    headline = EASY_HEADLINES[idx % len(EASY_HEADLINES)]
                    pred, prob = cached_analyze(headline)
                    render_game_header(st.session_state.speed_score, time_left=int(time_left), 
                                       headline_num=idx, total=20)
                    st.markdown(f"**Streak:** {st.session_state.speed_streak}")
                    st.markdown(f"#### {headline}")
                    if random.random() < 0.3:
                        st.info(random.choice(HINTS))
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"speed_real_{idx}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"speed_fake_{idx}"):
                            action = "FAKE"
                    if action:
                        if action == pred:
                            st.session_state.speed_score += 1
                            st.session_state.speed_streak += 1
                            on_correct_answer(player_name)
                            if st.session_state.speed_streak % 5 == 0:
                                st.session_state.speed_score += 2
                                st.success(f"🔥 Streak bonus! +2 points")
                                # Update streak achievements
                                for ach in ACHIEVEMENTS:
                                    if ach["id"].startswith("streak_") and st.session_state.speed_streak >= ach["max_progress"]:
                                        update_achievement(player_name, ach["id"], force_progress=ach["max_progress"], skip_collective=False)
                            else:
                                st.success("Correct!")
                        else:
                            st.session_state.speed_streak = 0
                            st.error(f"Wrong! It was {pred}")
                        st.session_state.speed_index += 1
                        time.sleep(0.3)
                        st.rerun()
                else:
                    if time_left <= 0:
                        st.warning("⏰ Time's up!")
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.speed_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.speed_score:
                        board[player_name] = {"score": st.session_state.speed_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_speed()

        # ---------- Survival Mode ----------
        elif mode == "💀 Survival Mode":
            @st.fragment(run_every=0.5)
            def game_loop_survival():
                if st.session_state.survival_wrong < 3 and st.session_state.survival_index < len(st.session_state.survival_headlines):
                    idx = st.session_state.survival_index
                    headline = st.session_state.survival_headlines[idx]
                    pred, prob = cached_analyze(headline)
                    lives_left = 3 - st.session_state.survival_wrong
                    render_game_header(st.session_state.survival_score, lives=lives_left, 
                                       headline_num=idx, total=len(st.session_state.survival_headlines))
                    st.markdown(f"#### {headline}")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"surv_real_{idx}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"surv_fake_{idx}"):
                            action = "FAKE"
                    if action:
                        if action == pred:
                            st.session_state.survival_score += 1
                            on_correct_answer(player_name)
                            st.success("Correct!")
                        else:
                            st.session_state.survival_wrong += 1
                            st.error(f"Wrong! It was {pred}. Lives left: {3 - st.session_state.survival_wrong}")
                        st.session_state.survival_index += 1
                        time.sleep(0.5)
                        st.rerun()
                else:
                    if st.session_state.survival_wrong >= 3:
                        st.warning("💀 Game Over – you lost all lives.")
                    else:
                        st.info("You've completed all headlines!")
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.survival_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.survival_score:
                        board[player_name] = {"score": st.session_state.survival_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_survival()

        # ---------- Expert Mode ----------
        elif mode == "🧠 Expert Mode":
            @st.fragment(run_every=0.5)
            def game_loop_expert():
                if st.session_state.expert_index < len(EXPERT_HEADLINES):
                    idx = st.session_state.expert_index
                    headline = EXPERT_HEADLINES[idx]
                    pred, prob = cached_analyze(headline)
                    render_game_header(st.session_state.expert_score, headline_num=idx, total=len(EXPERT_HEADLINES))
                    st.markdown(f"#### {headline}")
                    st.caption("Expert Mode – subtle headlines, no clickbait!")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"exp_real_{idx}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"exp_fake_{idx}"):
                            action = "FAKE"
                    if action:
                        if action == pred:
                            st.session_state.expert_score += 1
                            on_correct_answer(player_name)
                            st.success("Correct!")
                        else:
                            st.error(f"Wrong! It was {pred}")
                        st.session_state.expert_index += 1
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.expert_score} / {len(EXPERT_HEADLINES)}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.expert_score:
                        board[player_name] = {"score": st.session_state.expert_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_expert()

        # ---------- Swap Mode (62) ----------
        elif mode == "🔄 Swap Mode (62)":
            @st.fragment(run_every=0.5)
            def game_loop_swap():
                if st.session_state.swap_index < len(st.session_state.swap_headlines):
                    idx = st.session_state.swap_index
                    headline = st.session_state.swap_headlines[idx]
                    pred, prob = cached_analyze(headline)
                    render_game_header(st.session_state.swap_score, headline_num=idx, total=len(st.session_state.swap_headlines))
                    st.markdown(f"#### {headline}")
                    st.info(f"🤖 AI predicts this headline is **{pred}** with {prob*100:.1f}% confidence.")
                    st.markdown("**Do you agree with the AI?**")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ AGREE", key=f"swap_agree_{idx}"):
                            action = "agree"
                    with col_b2:
                        if st.button("❌ DISAGREE", key=f"swap_disagree_{idx}"):
                            action = "disagree"
                    if action:
                        ai_wrong = random.random() < 0.3
                        if action == "agree":
                            if not ai_wrong:
                                st.session_state.swap_score += 1
                                st.success("You correctly agreed with the AI! +1 point")
                                on_correct_answer(player_name)  # considered a correct meta-judgment
                            else:
                                st.error("The AI was wrong, and you agreed with it. No points.")
                        else:
                            if ai_wrong:
                                st.session_state.swap_score += 2
                                st.success("You caught the AI's mistake! +2 points")
                                on_correct_answer(player_name)  # also correct judgment
                            else:
                                st.error("The AI was correct, but you disagreed. No points.")
                        st.session_state.swap_index += 1
                        time.sleep(1)
                        st.rerun()
                else:
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.swap_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.swap_score:
                        board[player_name] = {"score": st.session_state.swap_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_swap()

        # ---------- Zoom In (53) ----------
        elif mode == "🔍 Zoom In (53)":
            @st.fragment(run_every=0.5)
            def game_loop_zoom():
                if st.session_state.zoom_index < 10:
                    if st.session_state.zoom_headline == "":
                        st.session_state.zoom_headline = random.choice(ALL_HEADLINES)
                        st.session_state.zoom_pred, _ = cached_analyze(st.session_state.zoom_headline)
                        st.session_state.zoom_start_time = time.time()
                    headline = st.session_state.zoom_headline
                    elapsed = time.time() - st.session_state.zoom_start_time
                    reveal_ratio = min(1.0, elapsed / 10)
                    visible_length = int(len(headline) * reveal_ratio)
                    if visible_length < len(headline):
                        displayed = headline[:visible_length] + "..." + ("█" * (len(headline) - visible_length))
                    else:
                        displayed = headline
                    render_game_header(st.session_state.zoom_score, headline_num=st.session_state.zoom_index, total=10)
                    st.markdown(f"#### {displayed}")
                    if reveal_ratio < 1:
                        st.caption(f"Revealing... {int(reveal_ratio*100)}% complete")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"zoom_real_{st.session_state.zoom_index}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"zoom_fake_{st.session_state.zoom_index}"):
                            action = "FAKE"
                    if action:
                        if action == st.session_state.zoom_pred:
                            if reveal_ratio < 0.3:
                                points = 5
                            elif reveal_ratio < 0.6:
                                points = 3
                            elif reveal_ratio < 0.9:
                                points = 2
                            else:
                                points = 1
                            st.session_state.zoom_score += points
                            on_correct_answer(player_name)
                            st.success(f"Correct! +{points} points")
                        else:
                            st.error(f"Wrong! It was {st.session_state.zoom_pred}")
                        st.session_state.zoom_index += 1
                        st.session_state.zoom_headline = ""
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.balloons()
                    st.markdown(f"## Final Score: {st.session_state.zoom_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.zoom_score:
                        board[player_name] = {"score": st.session_state.zoom_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_zoom()

        # ---------- Fact-Check Battle (65) ----------
        elif mode == "⚔️ Fact-Check Battle (65)":
            @st.fragment(run_every=0.5)
            def game_loop_battle():
                if st.session_state.battle_round < len(st.session_state.battle_headlines):
                    headline = st.session_state.battle_headlines[st.session_state.battle_round]
                    pred, prob = cached_analyze(headline)
                    st.markdown(f"### Round {st.session_state.battle_round+1} of {len(st.session_state.battle_headlines)}")
                    st.markdown(f"#### {headline}")
                    st.markdown(f"**Player Score:** {st.session_state.battle_player_score}  |  **AI Score:** {st.session_state.battle_ai_score}")
                    st.markdown("**Your verdict:**")
                    col1, col2 = st.columns(2)
                    player_choice = None
                    with col1:
                        if st.button("✅ REAL", key=f"battle_real_{st.session_state.battle_round}"):
                            player_choice = "REAL"
                    with col2:
                        if st.button("🚫 FAKE", key=f"battle_fake_{st.session_state.battle_round}"):
                            player_choice = "FAKE"
                    if player_choice:
                        ai_choice = pred
                        if player_choice == ai_choice:
                            st.session_state.battle_player_score += 1
                            st.session_state.battle_ai_score += 1
                            on_correct_answer(player_name)  # correct judgment
                            st.info("You and the AI agree. The audience is split – each gets 1 point.")
                        else:
                            if prob > 0.7 or prob < 0.3:
                                st.session_state.battle_ai_score += 2
                                st.warning("The AI is confident, and the audience agrees with the AI. AI gets 2 points.")
                            else:
                                if random.random() < 0.5:
                                    st.session_state.battle_player_score += 2
                                    st.success("You made a compelling argument! The audience gives you 2 points.")
                                    on_correct_answer(player_name)  # still correct in the sense of winning audience?
                                else:
                                    st.session_state.battle_ai_score += 2
                                    st.warning("The AI's argument was more convincing. AI gets 2 points.")
                        st.session_state.battle_round += 1
                        time.sleep(1.5)
                        st.rerun()
                else:
                    st.balloons()
                    if st.session_state.battle_player_score > st.session_state.battle_ai_score:
                        st.markdown(f"## 🏆 You win! Final: You {st.session_state.battle_player_score} – AI {st.session_state.battle_ai_score}")
                    elif st.session_state.battle_player_score < st.session_state.battle_ai_score:
                        st.markdown(f"## 🤖 AI wins! Final: You {st.session_state.battle_player_score} – AI {st.session_state.battle_ai_score}")
                    else:
                        st.markdown(f"## 🤝 It's a tie! Final: You {st.session_state.battle_player_score} – AI {st.session_state.battle_ai_score}")
                    board = load_leaderboard()
                    if player_name not in board or board[player_name]["score"] < st.session_state.battle_player_score:
                        board[player_name] = {"score": st.session_state.battle_player_score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        save_leaderboard(board)
                        st.success("New high score saved!")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_battle()

        # ---------- Training Mode (9) ----------
        elif mode == "📚 Training Mode (9)":
            @st.fragment(run_every=0.5)
            def game_loop_training():
                if st.session_state.training_index < len(st.session_state.training_headlines):
                    idx = st.session_state.training_index
                    headline = st.session_state.training_headlines[idx]
                    pred, prob = cached_analyze(headline)
                    render_game_header(st.session_state.training_score, headline_num=idx, total=len(st.session_state.training_headlines))
                    st.markdown(f"#### {headline}")
                    col_b1, col_b2 = st.columns(2)
                    action = None
                    with col_b1:
                        if st.button("✅ REAL", key=f"train_real_{idx}"):
                            action = "REAL"
                    with col_b2:
                        if st.button("🚫 FAKE", key=f"train_fake_{idx}"):
                            action = "FAKE"
                    if action:
                        if action == pred:
                            st.session_state.training_score += 1
                            on_correct_answer(player_name)
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Wrong! It was {pred}.")
                        st.markdown("### 📖 Explanation")
                        reasons = explain_reasoning(headline)
                        for r in reasons:
                            st.markdown(f"- {r}")
                        if pred == "FAKE":
                            st.markdown("**Suspicious words:**")
                            st.markdown(highlight_suspicious(headline), unsafe_allow_html=True)
                        st.session_state.training_index += 1
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.balloons()
                    st.markdown(f"## Training Complete! You got {st.session_state.training_score}/{len(st.session_state.training_headlines)} correct.")
                    if st.button("Play Again", use_container_width=True):
                        reset_game_mode(mode)
                        st.session_state.game_started = False
                        st.rerun()
            game_loop_training()

# -----------------------------
# Achievements Tab
# -----------------------------
with tab5:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🏆 Your Achievements")
    player_name = st.session_state.get("player_name", "Player")
    if os.path.exists(ACHIEVEMENTS_FILE):
        with open(ACHIEVEMENTS_FILE, "r") as f:
            all_players_data = json.load(f)
        all_players = list(all_players_data.keys())
    else:
        all_players = []
    selected_player = st.selectbox("Select player:", [player_name] + [p for p in all_players if p != player_name])
    if selected_player != player_name:
        player_name = selected_player
        st.session_state.player_name = player_name
        st.rerun()
    player_achs = load_achievements(player_name)
    cols = st.columns(3)
    for i, ach in enumerate(ACHIEVEMENTS):
        if ach.get("hidden", False):
            continue
        col = cols[i % 3]
        ach_data = player_achs.get(ach["id"], {"unlocked": False, "progress": 0, "max": ach["max_progress"]})
        unlocked = ach_data["unlocked"]
        progress = ach_data["progress"]
        max_prog = ach_data["max"]
        with col:
            if unlocked:
                st.markdown(f"""
                <div style="background: #e8f5e8; border-radius: 10px; padding: 10px; margin: 5px 0; border-left: 5px solid #00d26a;">
                    <span style="font-size: 1.5em;">{ach['icon']}</span>
                    <strong style="color: #00a86b;">{ach['name']}</strong><br>
                    <small>{ach['desc']}</small><br>
                    <span style="color: green;">✔ Unlocked {ach_data.get('unlocked_date','')}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if max_prog > 1:
                    percent = int(progress / max_prog * 100)
                    st.markdown(f"""
                    <div style="background: #f0f0f0; border-radius: 10px; padding: 10px; margin: 5px 0;">
                        <span style="font-size: 1.5em;">{ach['icon']}</span>
                        <strong>{ach['name']}</strong><br>
                        <small>{ach['desc']}</small><br>
                        <div style="background: #ddd; height: 8px; border-radius: 4px; margin: 5px 0;">
                            <div style="background: #667eea; height: 8px; border-radius: 4px; width: {percent}%;"></div>
                        </div>
                        <span style="font-size: 0.9em;">{progress}/{max_prog}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f0f0f0; border-radius: 10px; padding: 10px; margin: 5px 0; opacity: 0.7;">
                        <span style="font-size: 1.5em;">{ach['icon']}</span>
                        <strong>{ach['name']}</strong><br>
                        <small>{ach['desc']}</small><br>
                        <span style="color: #888;">🔒 Locked</span>
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Accuracy Challenge
# -----------------------------
with tab6:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Accuracy Challenge")
    st.markdown("No timer – just pure accuracy. Get all 10 right for a perfect 100%!")
    if not st.session_state.accuracy_started:
        player_name = st.text_input("Your name:", value="Player", key="acc_name_input")
        if st.button("Start Challenge", use_container_width=True):
            st.session_state.accuracy_index = 0
            st.session_state.accuracy_score = 0
            st.session_state.accuracy_started = True
            st.session_state.player_name = player_name  # use unified player_name
            st.session_state.total_games_played += 1
            if st.session_state.total_games_played == 1:
                update_achievement(player_name, "newbie", force_progress=1, skip_collective=False)
            st.rerun()
    else:
        player_name = st.session_state.player_name
        if st.session_state.accuracy_index < len(EASY_HEADLINES):
            idx = st.session_state.accuracy_index
            headline = EASY_HEADLINES[idx]
            pred, prob = analyze_text(headline)
            if pred is None:
                st.stop()
            st.progress((idx) / len(EASY_HEADLINES), text=f"Headline {idx+1} of {len(EASY_HEADLINES)}")
            st.markdown(f"**Current Score:** {st.session_state.accuracy_score} / {idx} correct")
            st.markdown(f"### 📰 {headline}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ REAL", key=f"acc_real_{idx}"):
                    if pred == "REAL":
                        st.session_state.accuracy_score += 1
                        st.success("Correct!")
                        on_correct_answer(player_name)
                    else:
                        st.error(f"Wrong! It was {pred}.")
                    st.session_state.accuracy_index += 1
                    st.rerun()
            with col2:
                if st.button("🚫 FAKE", key=f"acc_fake_{idx}"):
                    if pred == "FAKE":
                        st.session_state.accuracy_score += 1
                        st.success("Correct!")
                        on_correct_answer(player_name)
                    else:
                        st.error(f"Wrong! It was {pred}.")
                    st.session_state.accuracy_index += 1
                    st.rerun()
        else:
            accuracy_pct = (st.session_state.accuracy_score / len(EASY_HEADLINES)) * 100
            st.balloons()
            st.markdown(f"## 🎉 You scored **{accuracy_pct:.1f}%**")
            if accuracy_pct == 100:
                st.markdown("### Perfect! 🏆")
                update_achievement(player_name, "perfectionist", force_progress=1, skip_collective=False)
                st.session_state.perfect_scores += 1
            if st.button("Play Again", use_container_width=True):
                st.session_state.accuracy_started = False
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: white; opacity: 0.7;'>Made with ❤️ by Jaivardhan • Powered by Machine Learning and AI</p>", unsafe_allow_html=True)