class Board:
    def __init__(self, pits=6, seeds=4):
        """Initialize a Mancala board with specified pits per player and seeds per pit."""
        self.pits = pits
        self.seeds = seeds
        
        # Initialize the board: [player 1's pits, player 1's store, player 2's pits, player 2's store]
        self.board = [seeds] * pits + [0] + [seeds] * pits + [0]
        
        # Player 1 goes first (0-indexed)
        self.current_player = 0
        self.game_over = False
        
    def get_player_pits(self, player):
        """Return the indices of the specified player's pits."""
        if player == 0:  # Player 1
            return list(range(0, self.pits))
        else:  # Player 2
            return list(range(self.pits + 1, 2 * self.pits + 1))
            
    def get_store(self, player):
        """Return the index of the specified player's store."""
        if player == 0:  # Player 1
            return self.pits
        else:  # Player 2
            return 2 * self.pits + 1
            
    def is_valid_move(self, pit_idx):
        """Check if a move is valid."""
        player_pits = self.get_player_pits(self.current_player)
        return pit_idx in player_pits and self.board[pit_idx] > 0
        
    def make_move(self, pit_idx):
        """Make a move from the specified pit."""
        if not self.is_valid_move(pit_idx):
            return False
            
        # Pick up seeds
        seeds_to_sow = self.board[pit_idx]
        self.board[pit_idx] = 0
        
        # Sow seeds
        current_pit = pit_idx
        opponent_store = self.get_store(1 - self.current_player)
        
        while seeds_to_sow > 0:
            current_pit = (current_pit + 1) % len(self.board)
            
            # Skip opponent's store
            if current_pit == opponent_store:
                continue
                
            # Place one seed
            self.board[current_pit] += 1
            seeds_to_sow -= 1
        
        # Check if last seed was placed in player's store
        player_store = self.get_store(self.current_player)
        if current_pit == player_store:
            # Player gets another turn
            pass
        else:
            # Switch player
            self.current_player = 1 - self.current_player
        
        # Check if game is over
        self.check_game_over()
        
        return True
    
    def check_game_over(self):
        """Check if the game is over, and finalize scores if it is."""
        player1_empty = all(self.board[i] == 0 for i in self.get_player_pits(0))
        player2_empty = all(self.board[i] == 0 for i in self.get_player_pits(1))
        
        if player1_empty or player2_empty:
            self.game_over = True
            
            # Collect remaining seeds
            for player in [0, 1]:
                player_store = self.get_store(player)
                for pit in self.get_player_pits(player):
                    self.board[player_store] += self.board[pit]
                    self.board[pit] = 0
    
    def get_winner(self):
        """Return the winner (0 or 1) or None for a tie."""
        if not self.game_over:
            return None
            
        score1 = self.board[self.get_store(0)]
        score2 = self.board[self.get_store(1)]
        
        if score1 > score2:
            return 0
        elif score2 > score1:
            return 1
        else:
            return None  # Tie
    
    def __str__(self):
        """String representation of the board."""
        p2_pits = self.board[self.pits+1:2*self.pits+1]
        p1_pits = self.board[:self.pits]
        
        result = f"Player 2: {p2_pits}\n"
        result += f"{self.board[2*self.pits+1]}{'':^{len(str(p1_pits))+len(str(p2_pits))-8}}{self.board[self.pits]}\n"
        result += f"Player 1: {p1_pits}\n"
        result += f"Current player: {self.current_player+1}"
        
        return result
