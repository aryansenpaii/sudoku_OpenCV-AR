#SUDOKU SOLVING FUNCTIONALITY
def find_empty(board):
    """checkes where is an empty or unsolved block"""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None


def valid(board, num, pos):
    #checks row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    #checks column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    #checks box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if (board[i][j] == num and (i, j) != pos):
                return False

    return True


def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find
        #puts numbers from 1 to 9
    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve(board):
                return True
            board[row][col] = 0
    return False


def get_board(bo):
    if solve(bo):
        return bo
    else:
        raise ValueError
