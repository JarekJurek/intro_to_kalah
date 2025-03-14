import os

class Mancala:
    """A simple Mancala game implementation with Minimax and Alpha-Beta Pruning."""

    def __init__(self):
        """Initializes the game board with 6 pits per player and 1 store each."""
        self.pits = [4] * 6 + [0] + [4] * 6 + [0]  # 6 pits + 1 store per player
        self.current_player = 0  # 0 for player 1, 1 for player 2

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
            if index == (7 if self.current_player == 1 else 14):  # Skip opponent's store
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

    def is_game_over(self):
        """Checks if the game is over when one side is empty."""
        return sum(self.pits[:6]) == 0 or sum(self.pits[7:13]) == 0

    def get_winner(self):
        """Determines the winner based on the final scores."""
        self.pits[6] += sum(self.pits[:6])
        self.pits[13] += sum(self.pits[7:13])
        self.pits[:6] = self.pits[7:13] = [0] * 6
        if self.pits[6] > self.pits[13]:
            return "Player 1 Wins!"
        elif self.pits[6] < self.pits[13]:
            return "Player 2 Wins!"
        return "It's a Draw!"

    def display(self):
        """Clears the screen and displays the current board state."""
        print(" ", self.pits[12:6:-1])
        print(self.pits[13], "                  ", self.pits[6])
        print(" ", self.pits[:6])
        print(f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}")

    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with Alpha-Beta pruning."""
        if depth == 0 or self.is_game_over():
            return self.pits[6] - self.pits[13]
        
        if maximizing_player:
            max_eval = float('-inf')
            for i in range(6):
                try:
                    new_game = Mancala()
                    new_game.pits = self.pits[:]
                    new_game.current_player = self.current_player
                    new_game.move(i + 1)
                    eval = new_game.minimax(depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                except ValueError:
                    continue
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(6):
                try:
                    new_game = Mancala()
                    new_game.pits = self.pits[:]
                    new_game.current_player = self.current_player
                    new_game.move(i + 1)
                    eval = new_game.minimax(depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                except ValueError:
                    continue
            return min_eval

    def best_move(self):
        """Finds the best move using the Minimax algorithm."""
        best_value = float('-inf')
        best_move = None
        
        for i in range(6):
            try:
                new_game = Mancala()
                new_game.pits = self.pits[:]
                new_game.current_player = self.current_player
                new_game.move(i + 1)
                move_value = new_game.minimax(5, float('-inf'), float('inf'), False)
                if move_value > best_value:
                    best_value = move_value
                    best_move = i + 1
            except ValueError:
                continue
        return best_move


def main():
    game = Mancala()
    while not game.is_game_over():
        game.display()
        try:
            if game.current_player == 0:
                pit_choice = int(input("Choose a pit (1-6): "))
            else:
                pit_choice = game.best_move()
                print(f"AI chooses pit {pit_choice}")
            game.move(pit_choice)
            print('--------------------------------------------------------------------')
        except ValueError as e:
            print(e)

    game.display()
    print(game.get_winner())


if __name__ == "__main__":
    main()

