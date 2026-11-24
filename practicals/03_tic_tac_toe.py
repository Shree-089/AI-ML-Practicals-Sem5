"""
Practical 03: Tic-Tac-Toe game.

The human plays X and the computer plays O. The computer uses the
Minimax algorithm to choose an optimal move.
"""


def print_board(board):
    for row in range(3):
        cells = [board[row * 3 + col] if board[row * 3 + col] != " " else str(row * 3 + col + 1)
                 for col in range(3)]
        print(" " + " | ".join(cells))
        if row < 2:
            print("---+---+---")


def check_winner(board, player):
    winning_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    return any(all(board[index] == player for index in line) for line in winning_lines)


def is_draw(board):
    return " " not in board and not check_winner(board, "X") and not check_winner(board, "O")


def minimax(board, is_maximizing):
    """Return a score from the computer's (O) point of view."""
    if check_winner(board, "O"):
        return 1
    if check_winner(board, "X"):
        return -1
    if is_draw(board):
        return 0

    if is_maximizing:
        best_score = float("-inf")
        for index in range(9):
            if board[index] == " ":
                board[index] = "O"
                best_score = max(best_score, minimax(board, False))
                board[index] = " "
        return best_score

    best_score = float("inf")
    for index in range(9):
        if board[index] == " ":
            board[index] = "X"
            best_score = min(best_score, minimax(board, True))
            board[index] = " "
    return best_score


def computer_move(board):
    best_score = float("-inf")
    best_index = None

    for index in range(9):
        if board[index] == " ":
            board[index] = "O"
            score = minimax(board, False)
            board[index] = " "
            if score > best_score:
                best_score = score
                best_index = index

    return best_index


def main():
    board = [" "] * 9
    print("Tic-Tac-Toe — You are X, computer is O")
    print("Enter a number from 1 to 9 to select a cell.")

    while True:
        print_board(board)

        try:
            move = int(input("Your move (1-9): ")) - 1
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if move not in range(9) or board[move] != " ":
            print("That cell is unavailable. Choose an empty cell.")
            continue

        board[move] = "X"
        if check_winner(board, "X"):
            print_board(board)
            print("You win!")
            break
        if is_draw(board):
            print_board(board)
            print("The game is a draw.")
            break

        move = computer_move(board)
        if move is not None:
            board[move] = "O"
            print(f"Computer chooses cell {move + 1}.")

        if check_winner(board, "O"):
            print_board(board)
            print("Computer wins!")
            break
        if is_draw(board):
            print_board(board)
            print("The game is a draw.")
            break


if __name__ == "__main__":
    main()
