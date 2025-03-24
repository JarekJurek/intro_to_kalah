# Mancala (Kalah) Game Implementation

## Introduction

#### Group 22, DTU Introduction to AI

This project implements the traditional board game Mancala (also known as Kalah), featuring multiple game modes, customizable AI opponents with different strategies, and comprehensive statistics tracking. Developed as part of DTU's Introduction to AI course by Group 22, this implementation serves as both an interactive game and a platform for AI algorithm experimentation.

Mancala is one of the oldest known board games, dating back to ancient times. The game involves strategic movement of stones across a board with the objective of capturing more stones than your opponent. Our implementation follows the standard Kalah ruleset with 6 pits per player and one store (Kalah) for each player.

## Installation

This project requires Python 3.8+ and can be installed using either UV (recommended) or pip.

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver.

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/JarekJurek/intro_to_kalah.git
   cd intro_to_kalah
   ```

3. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies Using requirements.txt:**

   ```bash
   uv pip install -r requirements.txt
   ```

   **Or using the lock file:**
   ```bash
   uv pip sync
   ```

### Option 2: Using pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JarekJurek/intro_to_kalah.git
   cd intro_to_kalah
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game

After installation, you can run the Mancala game with:

```bash
python main.py
```

For visualization of game statistics:

```bash
python stats_reader.py
```

## Requirements

The project requires the following Python packages:
- numpy
- pandas
- plotly
- plotly-express
- statsmodels

All dependencies are listed in the 

requirements.txt

 file and will be installed automatically with the above instructions.

## Game Modes

### 1. Player vs Player

Challenge a friend to a game of Mancala on the same computer.

**How to play:**
1. Select option "1" from the main menu
2. Players take turns selecting pits (1-6)
3. The game displays the board state after each move
4. Game ends when one side is empty or one player has more than half the stones
5. The player with more stones in their store wins

### 2. Player vs AI

Test your skills against various AI strategies.

**How to play:**
1. Select option "2" from the main menu
2. Choose an AI configuration from the displayed list
3. Make your moves by selecting pits (1-6)
4. The AI will compute and display its chosen moves
5. Game continues until an end condition is reached

**AI Difficulty:**
The AI configurations range from basic to advanced strategies. Each AI uses specified parameters that affect its decision-making process, including search depth and evaluation weights.

### 3. AI vs AI (Single Match)

Watch two AI configurations battle against each other.

**How to play:**
1. Select option "3" from the main menu
2. Choose two AI configurations from the list
3. Enter the number of games you want them to play
4. Watch as the AIs compete against each other
5. Results are summarized at the end of the matches

This mode is excellent for comparing different AI strategies and seeing which performs better.

### 4. AIs vs AIs (Tournament)

Run multiple matches between randomly selected AI configurations.

**How to play:**
1. Select option "4" from the main menu
2. Specify the number of matches to run
3. Specify the number of games per match
4. Choose whether to show each move or just the results
5. Watch as the tournament unfolds

This mode is ideal for extensive AI strategy testing and gathering statistics across many different matchups.

### 5. View Statistics

Review and visualize game statistics.

**How to use:**
1. Select option "5" from the main menu
2. View text-based statistics in the console
3. Optionally generate and open interactive visualizations

## Visualizing Statistics

The game includes a comprehensive statistics tracking system that records:
- Game outcomes
- Player/AI performance
- Number of turns per game
- Timestamps

To visualize these statistics:
1. Play several games to accumulate data
2. Select option "5" (View Statistics) from the main menu
3. When prompted, type "y" to generate visualizations
4. An interactive HTML visualization will be created and opened in your browser

### Understanding the Visualizations

The visualization includes three main graphs:
1. **Win Rate by Player**: Shows the percentage of games won by each player/AI
2. **Average Turns per Player**: Shows the average game length for each player/AI
3. **Total Games Played**: Shows participation statistics

Each AI type is color-coded for easy identification. Hover over the bars to see exact values.

### Accessing Saved Visualizations

All visualizations are saved as HTML files in the `plots` directory:
```
plots/mancala_stats.html
```

You can open these files in any web browser at any time to review past visualizations.

## AI Configuration

AIs in the game are configurable through the `ai_configs.json` file. Each AI configuration includes:

```json
{
  "name": "Strategic AI",
  "type": "Minimax",
  "algorithm": "minimax",
  "depth": 5,
  "weights_early": [1, 3, 2],
  "weights_late": [3, 1, 2],
  "capture_weight": 1.5
}
```

### Parameters:

- **name**: Identifier for the AI
- **type**: Category of AI (for visualization grouping)
- "minimax" or "bfs" (Breadth-First Search)
- **depth**: Search depth (higher values make stronger but slower AI)
- **weights_early**: Importance of [score, board_control, extra_turns] in early game
- **weights_late**: Importance of [score, board_control, extra_turns] in late game
- **capture_weight**: How much the AI values potential captures

### Creating Custom AIs

To create your own AI configuration:
1. Open `ai_configs.json`
2. Add a new JSON object with your desired parameters
3. Save the file
4. Your new AI will appear in the selection lists

Example for adding an aggressive AI:
```json
{
  "name": "Aggressive AI",
  "type": "Capture-focused",
  "algorithm": "minimax",
  "depth": 4,
  "weights_early": [1, 0.5, 1],
  "weights_late": [2, 0.5, 1],
  "capture_weight": 3.0
}
```

## Game Rules

### Basic Rules
1. Players take turns moving stones
2. On your turn, select a non-empty pit on your side
3. Stones are distributed one by one counterclockwise
4. Your store is included in distribution, opponent's store is skipped

### Special Rules
1. **Extra Turn**: If your last stone lands in your store, you get another turn
2. **Capture**: If your last stone lands in an empty pit on your side, you capture that stone plus all stones in the opposite pit
3. **Game End**: The game ends when all pits on one side are empty, or a player has more than half the total stones in their store
4. **Final Capture**: When the game ends, remaining stones go to the player who owns that side

## Project Structure

- `main.py`: Main game implementation and user interface
- `stats_reader.py`: Statistics visualization module
- `ai_configs.json`: AI configuration file
- `game_stats.json`: Saved game statistics
- `plots/`: Directory containing visualization outputs

## Debugging and Troubleshooting

### Common Issues

1. **Missing dependencies**:
   ```
   ModuleNotFoundError: No module named 'plotly'
   ```
   Solution: Install required packages with `pip install plotly pandas`

2. **AI configuration file not found**:
   Solution: Create a basic `ai_configs.json` file with at least one configuration

3. **Visualization doesn't open**:
   Solution: Open the HTML file manually from the `plots` directory

### Advanced Debugging

To enable more detailed output during AI decision making, you can modify the `best_move` method in the `AI` class to print evaluation scores:

```python
def best_move(self, game):
    # Add print statements to see evaluation scores
    print(f"AI {self.name} evaluating moves...")
    # Rest of the method
```

