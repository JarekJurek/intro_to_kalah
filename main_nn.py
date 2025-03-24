import matplotlib.pyplot as plt
import json
import os
import random
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def load_game_stats(filename="game_stats.json"):
    """Load game statistics from a JSON file. Handles file not found and JSON decode errors."""
    try:
        if not os.path.exists(filename):
            print(f"Warning: File '{filename}' not found. Returning empty data.")
            return {"history": []}  # Return empty data, not None
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is corrupted. Please check the file or delete it.")
        return {"history": []}  # Return empty data in case of error to avoid crashing
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"history": []}

def plot_stats(filename="game_stats.json"):
    """
    Plots the game statistics from a JSON file.  Plots:
    - Total games played over time.
    - Wins per player.
    - Average turns per game over time.
    """
    data = load_game_stats(filename)  # Load data using the function
    if not data or "history" not in data:  # Check if the data is empty or malformed.
        print("No data to plot or invalid data structure.")
        return

    history = data["history"]
    if not history:
        print("No game history available to plot.")
        return

    # Data extraction
    game_numbers = list(range(1, len(history) + 1))
    player_wins = {}
    turns_per_game = []

    for game in history:
        winner = game['winner']
        player1 = game['player1']
        player2 = game['player2']
        turns = game['turns']
        turns_per_game.append(turns)

        if winner == 0:
            player_wins[player1] = player_wins.get(player1, 0) + 1
        elif winner == 1:
            player_wins[player2] = player_wins.get(player2, 0) + 1
        else:
            player_wins['Draw'] = player_wins.get('Draw', 0) + 1

    # Create figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # Increased figure size for better readability
    fig.suptitle("Mancala Game Statistics", fontsize=16)  # Overall title for the plot

    # Plot 1: Total Games Played
    axs[0].plot(game_numbers, game_numbers, marker='o', linestyle='-', color='blue')
    axs[0].set_title("Total Games Played", fontsize=12)
    axs[0].set_xlabel("Game Number", fontsize=10)
    axs[0].set_ylabel("Total Games", fontsize=10)
    axs[0].grid(True)

    # Plot 2: Wins per Player (Bar Chart)
    if player_wins:  # Check if there are any wins to plot
        players = list(player_wins.keys())
        wins = list(player_wins.values())
        axs[1].bar(players, wins, color='green')
        axs[1].set_title("Wins per Player", fontsize=12)
        axs[1].set_xlabel("Player", fontsize=10)
        axs[1].set_ylabel("Number of Wins", fontsize=10)
        axs[1].grid(axis='y')
    else:
        axs[1].text(0.5, 0.5, "No wins to display (No games played yet).", ha='center', va='center', fontsize=10,
                    color='gray')
        axs[1].set_title("Wins per Player", fontsize=12)

    # Plot 3: Average Turns per Game
    axs[2].plot(game_numbers, turns_per_game, marker='s', linestyle='-', color='red')
    axs[2].set_title("Turns per Game", fontsize=12)
    axs[2].set_xlabel("Game Number", fontsize=10)
    axs[2].set_ylabel("Number of Turns", fontsize=10)
    axs[2].grid(True)

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add space for the title
    plt.show()


class Mancala:
    """A simple Mancala game implementation."""

    def __init__(self):
        """Initializes the game board with 6 pits per player and 1 store each."""
        self.pits = [4] * 6 + [0] + [4] * 6 + [0]  # 6 pits + 1 store per player
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.turns = 0

    def clone(self):
        """Creates a deep copy of the game state."""
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
        return [i + 1 for i in range(6) if self.pits[player_offset + i] > 0]

    def is_game_over(self):
        """Checks if the game is over."""
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
        print("│            PLAYER 2 (TOP)              │")
        print("│     6    5    4    3    2    1          │")
        print("│  ┌─────┬─────┬─────┬─────┬─────┬─────┐  │")
        print(f"│  │ {self.pits[12]:2d}  │ {self.pits[11]:2d}  │ {self.pits[10]:2d}  │ {self.pits[9]:2d}  │ {self.pits[8]:2d}  │ {self.pits[7]:2d}  │  │")
        print("│  └─────┴─────┴─────┴─────┴─────┴─────┘  │")
        print(
            f"│P2[{self.pits[13]:2d}]                                         [{self.pits[6]:2d}]P1│")
        print("│  ┌─────┬─────┬─────┬─────┬─────┬─────┐  │")
        print(f"│  │ {self.pits[0]:2d}  │ {self.pits[1]:2d}  │ {self.pits[2]:2d}  │ {self.pits[3]:2d}  │ {self.pits[4]:2d}  │ {self.pits[5]:2d}  │  │")
        print("│  └─────┴─────┴─────┴─────┴─────┴─────┘  │")
        print("│     1    2    3    4    5    6          │")
        print("│            PLAYER 1 (BOTTOM)           │")
        print("│                                         │")
        print("└─────────────────────────────────────────┘")
        print(f"\nCurrent Turn: {'PLAYER 1' if self.current_player == 0 else 'PLAYER 2'}")
        print(f"Store: Player 1 = {self.pits[6]}, Player 2 = {self.pits[13]}\n")

    def evaluate(self, config=None):
        """Evaluates the current board state for the current player.  This is a simple heuristic."""
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
    """An AI agent that plays Mancala using a Neural Network to select moves."""

    def __init__(self, config):
        """
        Initializes the AI agent with a neural network.
        """
        self.name = config['name']
        self.config = config
        self.model = self.build_model()
        self.training_data = []  # Store (state, action, reward) tuples for training

    def build_model(self):
        """
        Builds a neural network model using Keras to directly predict the best move.
        The input is the game state (14 pits), and the output is a probability distribution
        over the 6 possible moves.
        """
        model = Sequential()
        model.add(Dense(32, input_dim=14, activation='relu'))  # Increased complexity
        model.add(Dense(16, activation='relu'))
        model.add(Dense(6, activation='softmax'))  # 6 output units for the 6 possible moves
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])  # Use categorical crossentropy for multi-class classification
        return model

    def get_nn_move(self, game_state):
        """
        Gets the move suggested by the neural network for a given game state.
        """
        # Ensure the input is a numpy array with the correct shape
        game_state_array = np.array([game_state])
        predictions = self.model.predict(game_state_array)
        # print(f"Predictions: {predictions}")  # Debugging
        #  Returns the index of the move with the highest probability
        return np.argmax(predictions) + 1  # +1 to convert from 0-based index to 1-based

    def best_move(self, game):
        """
        Returns the best move according to the neural network.  Handles invalid moves.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        game_state = game.pits.copy()
        predicted_move = self.get_nn_move(game_state)

        if predicted_move in valid_moves:
            return predicted_move
        else:
            # If the neural network suggests an invalid move, choose a random valid move.
            print(
                f"Warning: Neural network suggested invalid move {predicted_move}. Choosing a random valid move.")
            return random.choice(valid_moves)

    def remember(self, state, action, reward):
        """
        Stores a game state, action, and reward for training the neural network.
        The action is stored as a one-hot encoded vector.
        """
        #  Convert action to one-hot encoding
        action_vector = to_categorical(action - 1, num_classes=6)  # -1 because actions are 1-based
        self.training_data.append((state, action_vector, reward))

    def train_model(self, batch_size=32, epochs=10):
        """
        Trains the neural network using the stored training data.
        """
        if not self.training_data:
            return

        states, actions, rewards = zip(*self.training_data)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        #  Instead of predicting the reward, we are predicting the action directly.
        #  The target is the one-hot encoded action.  We need to adjust the rewards
        #  to reflect the quality of the action taken.  For now, we can use the reward
        #  directly, but in a more sophisticated setup, we would use a value function
        #  or a Q-value.
        self.model.fit(states, actions, batch_size=batch_size, epochs=epochs, verbose=0)
        self.training_data = []  # Clear training data after training

class GameStatistics:
    """Tracks game statistics."""

    def __init__(self):
        self.player_wins = {}
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
            self.player_wins[player1] = self.player_wins.get(player1, 0) + 1
        elif winner == 1:
            self.player_wins[player2] = self.player_wins.get(player2, 0) + 1
        else:
            self.draws += 1

    def display_stats(self):
        """Display the current statistics."""
        print("\n===== Game Statistics =====")
        print(f"Total games played: {self.total_games}")
        print(f"Draws: {self.draws}")
        print("\nWins by player:")
        for player, wins in self.player_wins.items():
            print(f"  {player}: {wins} wins ({(wins / self.total_games) * 100:.1f}%)")
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



def load_ai_configs():
    """Load AI configurations from a file."""
    filename = "ai_configs.json"
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return json.load(f)



def player_vs_player(stats):
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
            time.sleep(1)  # Add a slight delay
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
    """AI vs AI game mode with support for multiple games."""
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

            move = current_ai.best_move(game)  # Get move from NN

            if show_each_move:
                print(f"{current_ai.name} chooses pit {move}")

            # Get the current state before applying the move
            state = game.pits.copy()
            game.move(move)
            
            #  Get the reward.  For simplicity, we give a reward of 1 for winning,
            #  0 for a draw, and -1 for losing.  This could be made more sophisticated.if game.is_game_over():
            winner = game.get_winner()
            if winner == 0:
                reward = 1 if game.current_player == 0 else -1
            elif winner == 1:
                reward = 1 if game.current_player == 1 else -1
                reward = 0
            else:
                reward = 0

            current_ai.remember(state, move, reward)  # Store state, action, reward

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

        # Train the models after each game.
        ai1.train_model(batch_size=32, epochs=10)
        ai2.train_model(batch_size=32, epochs=10)

    # Display summary after all games
    print("\n" + "=" * 50)
    print(f"Results of {num_games} games:")
    print(f"{ai1.name}: {results[ai1.name]} wins ({results[ai1.name] / num_games * 100:.1f}%)")
    print(f"{ai2.name}: {results[ai2.name]} wins ({results[ai2.name] / num_games * 100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws'] / num_games * 100:.1f}%)")
    print(f"Average turns per game: {total_turns / num_games:.1f}")
    print("=" * 50)

    return results


def ais_vs_ais(stats, num_games_per_match=1, num_matches=10, show_each_move=False):
    """Run multiple matches between randomly selected AI configurations."""
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

        print(f"\nMatch {i + 1}/{num_matches}: {ai1_config['name']} vs {ai2_config['name']}")
        ai_vs_ai(stats, ai1_config, ai2_config, num_games_per_match, show_each_move)

    print("\nAll matches completed!")
    stats.display_stats()




def main():
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
            ais_vs_ais(stats, num_games_per_match=num_games_per_match, num_matches=num_matches,
                        show_each_move=show_each_move)
            input("Press Enter to continue...")

        elif choice == "5":
            stats.display_stats()
            plot = input("Do you want to plot the statistics? (y/n): ").lower()
            if plot == "y":
                plot_stats()
            else:
                pass
            input("Press Enter to continue...")

        elif choice == "6":
            print("Thanks for playing!")
            break
        stats.save_stats()



if __name__ == "__main__":
    main()