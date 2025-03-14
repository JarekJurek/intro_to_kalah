from board import Board

def display_board(board):
    """Display the current state of the Mancala board."""
    # Get the board state
    state = board.board
    pits = board.pits
    
    # Display player 2's pits (reversed for better visualization)
    print("\nPlayer 2's pits:")
    print("  ", end="")
    for i in range(2*pits, pits, -1):
        print(f"{state[i]:2d} ", end="")
    print(f"\nP2 store: {state[2*pits+1]}                 P1 store: {state[pits]}")
    
    # Display player 1's pits
    print("  ", end="")
    for i in range(pits):
        print(f"{state[i]:2d} ", end="")
    print("\nPlayer 1's pits:")
    
    # Show current player
    print(f"\nCurrent player: Player {board.current_player + 1}")

def get_player_move(board):
    """Get a valid move from the current player."""
    player_pits = board.get_player_pits(board.current_player)
    pit_mapping = {}
    
    # Map input numbers to actual pit indices
    if board.current_player == 0:  # Player 1
        pit_mapping = {i+1: pit for i, pit in enumerate(player_pits)}
    else:  # Player 2
        pit_mapping = {i+1: pit for i, pit in enumerate(player_pits)}
    
    # Show valid moves
    valid_moves = [key for key, idx in pit_mapping.items() if board.board[idx] > 0]
    print(f"Valid moves: {valid_moves}")
    
    while True:
        try:
            move = int(input(f"Player {board.current_player + 1}, choose a pit (1-{board.pits}): "))
            if move not in pit_mapping:
                print(f"Please enter a number between 1 and {board.pits}.")
                continue
                
            pit_idx = pit_mapping[move]
            if board.board[pit_idx] == 0:
                print("That pit is empty. Choose another pit.")
                continue
                
            return pit_idx
        except ValueError:
            print("Please enter a valid number.")

def main():
    """Main game loop."""
    # Initialize the game
    pits = 6  # Standard number of pits
    seeds = 4  # Standard number of seeds
    
    # Allow custom board setup
    print("Mancala (Kalah) Game")
    print("--------------------")
    try:
        custom = input("Use custom board? (y/n): ").lower().startswith('y')
        if custom:
            pits = int(input("Enter number of pits per player: "))
            seeds = int(input("Enter number of seeds per pit: "))
    except ValueError:
        print("Invalid input, using default board (6 pits, 4 seeds).")
        pits, seeds = 6, 4
    
    # Create and start the game
    board = Board(pits, seeds)
    
    # Game loop
    while not board.game_over:
        display_board(board)
        pit_idx = get_player_move(board)
        board.make_move(pit_idx)
    
    # Game over
    display_board(board)
    
    # Determine winner
    p1_store = board.board[board.get_store(0)]
    p2_store = board.board[board.get_store(1)]
    
    print("\nGame Over!")
    if p1_store > p2_store:
        print(f"Player 1 wins! Score: {p1_store}-{p2_store}")
    elif p2_store > p1_store:
        print(f"Player 2 wins! Score: {p2_store}-{p1_store}")
    else:
        print(f"It's a tie! Score: {p1_store}-{p2_store}")

if __name__ == "__main__":
    main()