# 🏀 Fantasy Basketball Free Agent Recommender

A powerful Streamlit-based web app that integrates with **Yahoo Fantasy Basketball** to help you identify and recommend the best **free agents** based on **custom scoring preferences**, recent performance, and your latest matchup data.

## 🔥 Features

- ✅ **Yahoo OAuth Integration**: Secure login with your Yahoo Fantasy account
- 🏆 **Custom League Selection**: View and analyze your leagues automatically
- ⚖️ **Stat Weight Configuration**: Adjust importance of each category (e.g., PTS, REB, AST, TO)
- 🗓️ **Flexible Time Periods**: Choose from "Last Week", "Last Month", or "Season"
- 📈 **Average or Total Stats**: Select between average or total performance metrics
- 🎯 **Matchup-Based Tuning**: Factor in your recent match results to tweak recommendations
- 📊 **Radar Charts & Data Tables**: Visualize performance comparison of top picks
- 🔒 **Secure Credentials Handling** via `oauth.json` and `.env`

## 🚀 Setup Instructions

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/fantasy-basketball-recommender.git
cd fantasy-basketball-recommender


### 2. Install Dependencies

pip install -r requirements.txt
3. Yahoo OAuth Setup
  *Visit the Yahoo Developer Console
  
  *Create a new project and app
  
  *Download the credentials file
  
 * Save it in your project root as oauth.json

###4. Run the App

streamlit run app.py
