import json
import plotly.express as px
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load AI configurations to get their types
def load_ai_configs():
    try:
        with open("ai_configs.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Create a mapping of AI names to their types
def create_ai_type_mapping():
    configs = load_ai_configs()
    ai_types = {}
    for config in configs:
        ai_types[config["name"]] = config["type"]
    return ai_types

try:
    with open("game_stats.json", "r") as f:
        stats_data = json.load(f)
        # Access the "history" key where the actual game data is stored
        stats = stats_data["history"]
        print("Stats loaded successfully")
except FileNotFoundError:
    stats = []

def plot_stats():
    if not stats:
        print("No stats available to plot")
        return
    
    # Get AI type information
    ai_types = create_ai_type_mapping()
    
    # Convert stats to DataFrame for easier analysis
    df = pd.DataFrame(stats)
    
    # Calculate win stats
    win_stats = defaultdict(lambda: {"wins": 0, "games": 0, "turns": [], "type": "Unknown"})
    
    for game in stats:
        player1 = game["player1"]
        player2 = game["player2"]
        winner_idx = game["winner"]
        turns = game["turns"]
        
        # Update games played
        win_stats[player1]["games"] += 1
        win_stats[player2]["games"] += 1
        
        # Track turns
        win_stats[player1]["turns"].append(turns)
        win_stats[player2]["turns"].append(turns)
        
        # Set AI type if available
        if player1 in ai_types:
            win_stats[player1]["type"] = ai_types[player1]
        if player2 in ai_types:
            win_stats[player2]["type"] = ai_types[player2]
        
        # Update wins
        if winner_idx == 0:
            win_stats[player1]["wins"] += 1
        elif winner_idx == 1:
            win_stats[player2]["wins"] += 1
    
    # Create figures
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("Win Rate by Player", "Average Turns per Player", "Total Games Played per Player"),
                        vertical_spacing=0.1)
    
    # Prepare data for plotting
    players = []
    player_labels = []  # Will include AI type
    win_rates = []
    avg_turns = []
    total_games = []
    colors = []  # For coloring by AI type
    
    # Create color mapping for AI types
    ai_type_set = set(data["type"] for player, data in win_stats.items())
    color_map = {ai_type: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                 for i, ai_type in enumerate(ai_type_set)}
    
    for player, data in win_stats.items():
        if data["games"] > 0:  # Avoid division by zero
            players.append(player)
            player_labels.append(f"{player} ({data['type']})")
            win_rates.append(data["wins"] / data["games"] * 100)
            avg_turns.append(sum(data["turns"]) / len(data["turns"]))
            total_games.append(data["games"])
            colors.append(color_map[data["type"]])
    
    # Add win rate bar chart
    fig.add_trace(
        go.Bar(x=player_labels, y=win_rates, text=[f"{wr:.1f}%" for wr in win_rates], 
               textposition='auto', marker_color=colors),
        row=1, col=1
    )
    
    # Add average turns bar chart
    fig.add_trace(
        go.Bar(x=player_labels, y=avg_turns, text=[f"{at:.1f}" for at in avg_turns], 
               textposition='auto', marker_color=colors),
        row=2, col=1
    )
    
    # Add total games played bar chart
    fig.add_trace(
        go.Bar(x=player_labels, y=total_games, text=[f"{g}" for g in total_games], 
               textposition='auto', marker_color=colors),
        row=3, col=1
    )
    
    # Create a "dummy" trace for each AI type to create a legend
    for ai_type, color in color_map.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], name=ai_type, marker_color=color, showlegend=True),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Mancala Game Statistics",
        height=1000,
        barmode='group',
        legend_title_text="AI Type"
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Turns", row=2, col=1)
    fig.update_yaxes(title_text="Total Games", row=3, col=1)
    
    # Save the figure to a file
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.write_html("plots/mancala_stats.html")
    print("Stats visualization saved to plots/mancala_stats.html")
    
    # Also try to show it in the browser
    try:
        fig.show()
    except Exception as e:
        print(f"Could not show interactive plot: {e}")
        print("You can open the saved HTML file in your browser instead.")

if __name__ == "__main__":
    plot_stats()