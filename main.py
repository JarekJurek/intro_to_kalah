"""
Mancala Game Implementation

This module provides a complete implementation of the Mancala (Kalah) board game with:
- Game logic and rules handling
- AI opponents with configurable strategies
- Multiple game modes (PvP, PvAI, AI vs AI)
- Game statistics tracking and persistence

Key components:
1. Mancala class: Implements the game board and rules
2. AI class: Implements minimax and BFS algorithms for computer players
3. GameStatistics: Tracks and saves game outcomes
4. Game mode functions: Manages different play scenarios
5. Main menu interface: User interaction and game setup

The implementation supports customizable AI configurations loaded from ai_configs.json,
with statistics tracked and saved to game_stats.json.
"""

from collections import defaultdict
import json
import os
import random
import time

import stats_reader

class Mancala:
    """A simple Mancala game implementation with Minimax and Alpha-Beta Pruning."""

    def __init__(self):
        """Initializes the game board with 6 pits per player and 1 store each."""
        self.pits = [4] * 6 + [0] + [4] * 6 + [0]  # 6 pits + 1 store per player
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.turns = 0

    def clone(self):
        """Creates a deep copy of the game state for AI simulation."""
        clone = Mancala()
        clone.pits = self.pits.copy()
        clone.current_player = self.current_player
        return clone

    def move(self, pit_index):
        """Executes a move for the current player from the given pit index."""
        if not (1 <= pit_index <= 6):
            raise ValueError("Invalid pit index. Choose between 1 and 6.")
        
        pit_index -= 1  # Convert to 0-based index

        player_offset = self.current_player * 7
        pit = player_offset + pit_index
        gems = self.pits[pit]
        if gems == 0:
            raise ValueError("Cannot move from an empty pit.")
        
        self.pits[pit] = 0
        index = pit
        
        while gems > 0:
            index = (index + 1) % 14
            if index == (6 if self.current_player == 1 else 13):  # Skip opponent's store
                continue
            self.pits[index] += 1
            gems -= 1
        
        # Extra turn condition
        if index == (6 if self.current_player == 0 else 13):
            return  # Player gets another turn
        
        # Capture rule
        if (self.pits[index] == 1 and index in range(player_offset, player_offset + 6)):
            opposite_index = 12 - index  # Mirrored position
            if self.pits[opposite_index] > 0:
                self.pits[player_offset + 6] += self.pits[opposite_index] + 1
                self.pits[index] = self.pits[opposite_index] = 0
        
        self.current_player = 1 - self.current_player  # Switch turn
        self.turns += 1
    
    def get_valid_moves(self):
        """Returns a list of valid moves for the current player."""
        player_offset = self.current_player * 7
        return [i+1 for i in range(6) if self.pits[player_offset + i] > 0]

    def is_game_over(self):
        """Checks if the game is over when one side is empty or if a player has more than half the gems."""
        total_gems = sum(self.pits)  # Count all gems
        # Game over if one side is empty
        if sum(self.pits[:6]) == 0 or sum(self.pits[7:13]) == 0:
            return True
        # Game over if one player has more than half the gems in their store
        if self.pits[6] > total_gems / 2 or self.pits[13] > total_gems / 2:
            return True
        return False

    def get_winner(self):
        """Determines the winner based on the final scores."""
        self.pits[6] += sum(self.pits[:6])
        self.pits[13] += sum(self.pits[7:13])
        self.pits[:6] = self.pits[7:13] = [0] * 6
        if self.pits[6] > self.pits[13]:
            return 0  # Player 1 wins
        elif self.pits[6] < self.pits[13]:
            return 1  # Player 2 wins
        return -1  # Draw

    def get_winner_message(self, player1="Player 1", player2="Player 2"):
        """Returns a message describing the winner."""
        winner = self.get_winner()
        if winner == 0:
            return f"Player 1 ({player1}) Wins in {self.turns} turns!"
        elif winner == 1:
            return f"Player 2 ({player2}) Wins! in {self.turns} turns!"
        return "It's a Draw!"

    def display(self):
        """Displays the current board state."""
        print("\n┌───────────── MANCALA BOARD ─────────────┐")
        print("│                                         │")
        print("│           PLAYER 2 (TOP)                │")
        print("│     6     5     4     3     2     1     │")
        print("│  ┌─────┬─────┬─────┬─────┬─────┬─────┐  │")
        print(f"│  │ {self.pits[12]:2d}  │ {self.pits[11]:2d}  │ {self.pits[10]:2d}  │ {self.pits[9]:2d}  │ {self.pits[8]:2d}  │ {self.pits[7]:2d}  │  │")
        print("│  └─────┴─────┴─────┴─────┴─────┴─────┘  │")
        print(f"│P2 [{self.pits[13]:2d}]                           [{self.pits[6]:2d}] P1│")
        print("│  ┌─────┬─────┬─────┬─────┬─────┬─────┐  │")
        print(f"│  │ {self.pits[0]:2d}  │ {self.pits[1]:2d}  │ {self.pits[2]:2d}  │ {self.pits[3]:2d}  │ {self.pits[4]:2d}  │ {self.pits[5]:2d}  │  │")
        print("│  └─────┴─────┴─────┴─────┴─────┴─────┘  │")
        print("│     1     2     3     4     5     6     │")
        print("│           PLAYER 1 (BOTTOM)             │")
        print("│                                         │")
        print("└─────────────────────────────────────────┘")
        print(f"\nCurrent Turn: {'PLAYER 1' if self.current_player == 0 else 'PLAYER 2'}")
        print(f"Store: Player 1 = {self.pits[6]}, Player 2 = {self.pits[13]}\n")

    def evaluate(self, config=None):
        """Evaluates the current board state for the current player."""
        if config is None:
            config = {
                "weights_early": [1, 1, 1],  
                "weights_late": [1, 1, 1],   
                "capture_weight": 1          
            }
        
        # Base evaluation: score difference based on current player
        if self.current_player == 0:  # Player 1
            score = self.pits[6] - self.pits[13]  # P1 - P2
            ai_control = sum(self.pits[:6])
            human_control = sum(self.pits[7:13])
        else:  # Player 2
            score = self.pits[13] - self.pits[6]  # P2 - P1
            ai_control = sum(self.pits[7:13])
            human_control = sum(self.pits[:6])
        
        control = ai_control - human_control

        # Extra turn potential:
        extra_turn_potential = 0
        store_index = 6 if self.current_player == 0 else 13
        pit_range = range(0, 6) if self.current_player == 0 else range(7, 13)
        
        for i in pit_range:
            pit_value = self.pits[i]
            if pit_value > 0:
                # If this pit's stones would end in the player's store
                landing_spot = (i + pit_value) % 14
                if landing_spot == store_index:
                    extra_turn_potential += 2
        
        # Capture potential
        capture_potential = 0
        for i in pit_range:
            pit_value = self.pits[i]
            if pit_value > 0:
                landing_spot = (i + pit_value) % 14
                if landing_spot in pit_range and self.pits[landing_spot] == 0:
                    opposite = 12 - landing_spot
                    if self.pits[opposite] > 0:
                        capture_potential += self.pits[opposite]

        # Weight the game state:
        total_gems = ai_control + human_control
        max_gems = 48
        game_progress = 1.0 - total_gems / max_gems

        # Early game strategy vs late game strategy
        if game_progress < 0.5:
            w = config["weights_early"]
        else:  # Late game: prioritize store
            w = config["weights_late"]
            
        return (w[0] * score + 
                w[1] * control + 
                w[2] * extra_turn_potential + 
                config["capture_weight"] * capture_potential * (1 + game_progress))
    
class AI:
    """An AI agent that plays Mancala using the Minimax algorithm with Alpha-Beta pruning."""

    # def __init__(self, name, depth=5, weights_early=[1, 1, 1], weights_late=[1, 1, 1]):
    def __init__(self, config):
        
        self.algorithm = config.get('algorithm', 'minimax').lower()  # Default to Minimax
        self.depth = config['depth']
        self.weights_early = config['weights_early']
        self.weights_late = config['weights_late']
        self.name = config['name']
        self.config = config
        
        if self.algorithm not in ['minimax', 'bfs']:
            raise ValueError(f"Invalid algorithm: {self.algorithm}.  Must be 'minimax' or 'bfs'.")

    def best_move(self, game):
            """Returns the best move according to the AI's chosen strategy."""
            if self.algorithm == 'minimax':
                alpha = float('-inf')
                beta = float('inf')
                best_move = game.get_valid_moves()[0]
                best_score = float('-inf')
                for move in game.get_valid_moves():
                    game_clone = game.clone()
                    extra_turn = game_clone.move(move)
                    if extra_turn:
                        score = self.minimax(game_clone, self.depth - 1, alpha, beta, True)
                    else:
                        score = self.minimax(game_clone, self.depth - 1, alpha, beta, False)
                    if score > best_score:
                        best_score = score
                        best_move = move
                    alpha = max(alpha, best_score)
                return best_move
            elif self.algorithm == 'bfs':
                return self.breadth_first_search(game)
            else:
                raise ValueError(f"Invalid algorithm: {self.algorithm}. Must be 'minimax' or 'bfs'.")

    def minimax(self, game, depth, alpha, beta, maximizing):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Best score for maximizing player
            beta: Best score for minimizing player
            maximizing: Boolean indicating if maximizing or minimizing
            
        Returns:
            Best evaluation score for the current player
        """
        if depth == 0 or game.is_game_over():
            return game.evaluate(self.config)
            
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            # No valid moves, switch player
            game.current_player = 1 - game.current_player
            return self.minimax(game, depth, alpha, beta, not maximizing)
            
        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_clone = game.clone()
                extra_turn = game_clone.move(move)
                
                if extra_turn:
                    eval = self.minimax(game_clone, depth - 1, alpha, beta, True)
                else:
                    eval = self.minimax(game_clone, depth - 1, alpha, beta, False)
                    
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_clone = game.clone()
                extra_turn = game_clone.move(move)
                
                if extra_turn:
                    eval = self.minimax(game_clone, depth - 1, alpha, beta, False)
                else:
                    eval = self.minimax(game_clone, depth - 1, alpha, beta, True)
                    
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def breadth_first_search(self, game):
        """Returns the best move according to Breadth-First Search."""
        original_game = game.clone()  # Store the original game state
        valid_moves = original_game.get_valid_moves()  # Get valid moves from the original state
        if not valid_moves:
            return None

        best_move = None
        best_score = float('-inf')
        queue = [(original_game.clone(), move, 0) for move in valid_moves]

        while queue:
            current_game, move, current_depth = queue.pop(0)
            try:
                current_game.move(move)
            except ValueError:
                continue

            if current_depth == self.depth - 1 or current_game.is_game_over():
                score = self.evaluate(current_game)
                if score > best_score:
                    best_score = score
                    best_move = move
            elif current_depth < self.depth - 1:
                next_moves = original_game.get_valid_moves() #error
                if next_moves:
                    for next_move in next_moves:
                        next_game = current_game.clone()
                        try:
                            next_game.move(next_move)
                            queue.append((next_game, next_move, current_depth + 1))
                        except ValueError:
                            continue

        if best_move is None:
            if valid_moves:
                best_move = valid_moves[0]
            else:
                return None
        return best_move

    def evaluate(self, game):
        """
        Evaluates the game state.  A simple heuristic for Mancala:
        Difference in stones in the player's store.
        """
        if game.current_player == 0:
            return game.pits[6] - game.pits[13]
        else:
            return game.pits[13] - game.pits[6]    
        
class GameStatistics:
    """Tracks game statistics."""
    
    def __init__(self):
        self.player_wins = defaultdict(int)
        self.total_games = 0
        self.draws = 0
        self.turns = 0
        self.history = []
    
    def record_game(self, player1, player2, winner, turns):
        """Record the results of a game."""
        self.total_games += 1
        result = {
            'player1': player1,
            'player2': player2,
            'winner': winner,
            'turns': turns,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(result)
        
        if winner == 0:
            self.player_wins[player1] += 1
        elif winner == 1:
            self.player_wins[player2] += 1
        else:
            self.draws += 1
    
    def display_stats(self):
        """Display the current statistics."""
        print("\n===== Game Statistics =====")
        print(f"Total games played: {self.total_games}")
        print(f"Draws: {self.draws}")
        print("\nWins by player:")
        for player, wins in self.player_wins.items():
            print(f"  {player}: {wins} wins ({(wins/self.total_games)*100:.1f}%)")
        print("\nRecent games:")
        for game in self.history[-5:]:
            winner_name = game['winner']
            if winner_name == 0:
                winner_name = game['player1']
            elif winner_name == 1:
                winner_name = game['player2']
            else:
                winner_name = "Draw"
            print(f"  {game['timestamp']}: {game['player1']} vs {game['player2']} - Winner: {winner_name}")
        
        self.save_stats()

    def save_stats(self):
        """Save the game statistics to a file."""
        try:
            if not os.path.exists("game_stats.json") or os.path.getsize("game_stats.json") == 0:
                with open("game_stats.json", "w") as f:
                    json.dump({"history": self.history}, f, indent=2)
            else:
                try:
                    with open("game_stats.json", "r") as f:
                        data = json.load(f)
                    # Merge the histories
                    if "history" not in data:
                        data["history"] = []
                    data["history"].extend(self.history)
                except json.JSONDecodeError:
                    # File exists but is corrupted
                    data = {"history": self.history}
                
                with open("game_stats.json", "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving game statistics: {e}")

        # self.history = []  # Clear history after saving


def load_ai_configs():
    """
    Load AI configurations from ai_configs.json file.
    
    Returns:
        list: A list of AI configuration dictionaries, or empty list if file not found
    """
    filename = "ai_configs.json"
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return json.load(f)
    

def player_vs_player(stats):
    """
    Run a Player vs Player game mode.
    
    Args:
        stats: GameStatistics object to record results
        
    Returns:
        int: Winner of the game (0 for player 1, 1 for player 2, -1 for draw)
    """
    game = Mancala()
    while not game.is_game_over():
        game.display()
        try:
            player_name = "Player 1" if game.current_player == 0 else "Player 2"
            pit_choice = int(input(f"{player_name}, choose a pit (1-6): "))
            game.move(pit_choice)
            print('--------------------------------------------------------------------')
        except ValueError as e:
            print(e)

    game.display()
    winner = game.get_winner()
    print(game.get_winner_message("Human 1", "Human 2"))
    stats.record_game("Human 1", "Human 2", winner, game.turns)
    return winner


def player_vs_ai(stats, con):
    """Human vs AI game mode."""
    game = Mancala()
    ai_configs = load_ai_configs()
    ai = AI(ai_configs[con])
    
    human_player = 0  # Human goes first as player 1
    
    while not game.is_game_over():
        game.display()
        
        if game.current_player == human_player:
            try:
                pit_choice = int(input("Your turn, choose a pit (1-6): "))
                game.move(pit_choice)
            except ValueError as e:
                print(e)
                input("Press Enter to continue...")
        else:
            print(f"{ai.name} is thinking...")
            time.sleep(1)  # Add a slight delay to make it feel more natural
            move = ai.best_move(game)
            print(f"{ai.name} chooses pit {move}")
            game.move(move)
            input("Press Enter to continue...")
    
    game.display()
    winner = game.get_winner()
    
    player_names = ["Human", ai.name] if human_player == 0 else [ai.name, "Human"]
    print(game.get_winner_message(player_names[0], player_names[1]))
    stats.record_game(player_names[0], player_names[1], winner, game.turns)
    return winner

def ai_vs_ai(stats, ai1_config, ai2_config, num_games=1, show_each_move=True):
    """
    AI vs AI game mode with support for multiple games.
    
    Args:
        stats: GameStatistics object to record results
        ai1_config: Configuration dictionary for first AI
        ai2_config: Configuration dictionary for second AI
        num_games: Number of games to play
        show_each_move: Whether to display each move or just the final results
        
    Returns:
        Dictionary with results of the matches
    """
    results = {ai1_config['name']: 0, ai2_config['name']: 0, 'draws': 0}
    total_turns = 0
    
    print(f"Running {num_games} games between {ai1_config['name']} and {ai2_config['name']}...")
    
    for game_num in range(num_games):
        game = Mancala()
        ai1 = AI(ai1_config)
        ai2 = AI(ai2_config)
        
        # Alternate starting player
        if game_num % 2 == 0:
            first_ai, second_ai = ai1, ai2
        else:
            first_ai, second_ai = ai2, ai1
            
        print(f"\nGame {game_num + 1}/{num_games}: {first_ai.name} starts")
        
        while not game.is_game_over():
            if show_each_move:
                game.display()
            
            current_ai = first_ai if game.current_player == 0 else second_ai
            
            if show_each_move:
                print(f"{current_ai.name} is thinking...")
                
            move = current_ai.best_move(game)
            
            if show_each_move:
                print(f"{current_ai.name} chooses pit {move}")
                
            game.move(move)

        
        # Game finished
        if show_each_move:
            game.display()
            
        winner = game.get_winner()
        total_turns += game.turns
        
        # Determine winner name based on the winner and starting positions
        if winner == 0:  # Player 1 wins
            winner_name = first_ai.name
        elif winner == 1:  # Player 2 wins
            winner_name = second_ai.name
        else:  # Draw
            winner_name = "Draw"
            
        # Update results
        if winner_name == ai1.name:
            results[ai1.name] += 1
        elif winner_name == ai2.name:
            results[ai2.name] += 1
        else:
            results['draws'] += 1
            
        # Record game stats
        stats.record_game(first_ai.name, second_ai.name, winner, game.turns)
        
        if show_each_move:
            print(game.get_winner_message(first_ai.name, second_ai.name))
    
    # Display summary after all games
    print("\n" + "="*50)
    print(f"Results of {num_games} games:")
    print(f"{ai1.name}: {results[ai1.name]} wins ({results[ai1.name]/num_games*100:.1f}%)")
    print(f"{ai2.name}: {results[ai2.name]} wins ({results[ai2.name]/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"Average turns per game: {total_turns/num_games:.1f}")
    print("="*50)
    
    return results

def ais_vs_ais(stats, num_games_per_match=1, num_matches=10, show_each_move=False):
    """
    Run multiple matches between randomly selected AI configurations.
    
    Args:
        stats: GameStatistics object to record results
        num_games_per_match: Number of games to play per AI matchup
        num_matches: Number of different AI matchups to run
        show_each_move: Whether to display each move or just the final results
    """
    ai_configs = load_ai_configs()
    
    if len(ai_configs) < 2:
        print("Not enough AI configurations available. Need at least 2.")
        return
    
    print(f"Running {num_matches} random AI matches with {num_games_per_match} games each...")
    
    for i in range(num_matches):
        # Select two different configurations randomly
        ai1_index = random.randint(0, len(ai_configs) - 1)
        ai2_index = random.randint(0, len(ai_configs) - 1)
        
        # Ensure we don't have the same AI playing itself
        while ai2_index == ai1_index:
            ai2_index = random.randint(0, len(ai_configs) - 1)
        
        ai1_config = ai_configs[ai1_index]
        ai2_config = ai_configs[ai2_index]
        
        print(f"\nMatch {i+1}/{num_matches}: {ai1_config['name']} vs {ai2_config['name']}")
        ai_vs_ai(stats, ai1_config, ai2_config, num_games_per_match, show_each_move)
    
    print("\nAll matches completed!")
    stats.display_stats()
        
def main():
    """
    Main function that runs the Mancala game interface.
    
    Presents a menu of game options and handles user interaction with the game.
    Initializes the statistics tracking and ensures stats are saved on exit.
    """
    stats = GameStatistics()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Welcome to Mancala!")
        print("Choose a game mode:")
        print("1. Player vs Player")
        print("2. Player vs AI")
        print("3. AI vs AI")
        print("4. AIs vs AIs")
        print("5. View Statistics")
        print("6. Exit")

        choice = input("Enter your choice: ")
        
        if choice == "1":
            player_vs_player(stats)
        
        elif choice == "2":
            configs = load_ai_configs()
            for i, config in enumerate(configs):
                print(f"{i}. {config}\n")
            con = int(input("Enter the AI configuration for your AI opponent: "))
            player_vs_ai(stats, con)
            # pass
            
        elif choice == "3":
            print("\n\nChoose AI configurations:")
            configs = load_ai_configs()
            for i, config in enumerate(configs):
                print(f"{i}. {config}\n")
            ai1 = int(input("Enter the AI configuration for AI 1: "))
            ai2 = int(input("Enter the AI configuration for AI 2: "))
            num_games = int(input("Enter the number of games to run: "))
            ai_vs_ai(stats, configs[ai1], configs[ai2], num_games, show_each_move=True)
            input("Press Enter to continue...")
        elif choice == "4":
            num_matches = int(input("Enter the number of matches to run: "))
            num_games_per_match = int(input("Enter the number of games per match: "))
            show_each_move = input("Show each move? (y/n): ").lower()
            if show_each_move == "y":
                show_each_move = True
            else:
                show_each_move = False
            ais_vs_ais(stats, num_games_per_match=num_games_per_match, num_matches=num_matches, show_each_move=show_each_move)
            input("Press Enter to continue...")

        elif choice == "5":
            stats.display_stats()
            plot = input("Do you want to plot the statistics? (y/n): ").lower()
            if plot == "y":
                stats_reader.plot_stats()
            else:
                pass
            input("Press Enter to continue...")

        elif choice == "6":
            print("Thanks for playing!")
            break
    stats.save_stats()

if __name__ == "__main__":
    main()

