import argparse
import copy
import sys
import time

cache = {} # you can use this to implement state caching!
# A dictionary of the form {id: (evaluated utility, perspective)}
LARGE_NUM = 1000000000

class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    # turn : a string that represents whose turn it is. 'r' for red, 'b' for black
    def __init__(self, board, turn):

        self.board = board
        self.turn = turn
        self.width = 8
        self.height = 8
        self.id = hash(str(board))

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")


    def winner(self):
        """
        Returns the state's winner on the given turn if there is one, 
        otherwise returns None
        """

        # If the enemy has no more pieces, or if the enemy cannot make 
        # any more moves, then we win

        # Count the pieces
        red_count = 0
        black_count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] in ['r', 'R']:
                    red_count += 1
                elif self.board[i][j] in ['b', 'B']:
                    black_count += 1

        # Check if the party on this turn can make any moves
        if len(self.generate_successors()) == 0:
            return get_next_turn(self.turn)

        elif red_count == 0:
            return 'b'
        elif black_count == 0:
            return 'r'
        
        return None
    
    
    def generate_successors(self):
        """
        Generate the successors for the current state, note we must jump 
        if we have the option to do so
        """
        suc = []
        jump_possible = False # check if we can make jumps for ANY one of our pieces
        for i in range(self.height):
            for j in range(self.width):
                piece = self.board[i][j]
                # If we are on a moveable piece
                if piece.lower() == self.turn:
                    jumped, locations = jump_helper(self.board, i, j, piece, self.height, self.width) # try jumping
                    if jumped: # If we jumped
                        for b in locations:
                            s = State(b, get_next_turn(self.turn))
                            suc.append(s)
                        jump_possible = True

        if not jump_possible: # if no jump available (Note: if we have the option to jump, we MUST jump)
            for i in range(self.height):
                for j in range(self.width):
                    piece = self.board[i][j]
                    if piece.lower() == self.turn: # if piece moveable
                        if piece.islower():
                            move_dir = ((-1, 1), (-1, -1)) if piece == 'r' else ((1, 1), (1, -1))
                            edge = 0 if piece == 'r' else self.height-1
                        else:
                            move_dir = ((-1, 1), (-1, -1), (1, 1), (1, -1))
                        
                        for d in move_dir:
                            new_i, new_j = i+d[0], j+d[1]
                            if 0 <= new_i < self.height and 0 <= new_j < self.width:
                                if self.board[new_i][new_j] == '.': # If empty space, move there
                                    new_board = copy.deepcopy(self.board)
                                    new_board[i][j] = '.'
                                    if piece.islower() and new_i == edge: # Turn to king
                                        piece = piece.upper()
                                    new_board[new_i][new_j] = piece
                                    s = State(new_board, get_next_turn(self.turn))
                                    suc.append(s)
        return suc
    


    def util(self, perspective, depth):
        """
        Return the evaluated utility of this board in a party's perspective, 
        using the formula f_self - f_opp
        where each f = 4*king + 2*advanced_pawn + 1*pawn
        """
        if self.id in cache:
            multiplier = 1 if cache[self.id][1] == perspective else -1
            if cache[self.id][0] in [LARGE_NUM, -LARGE_NUM]: 
                return multiplier * cache[self.id][0] // (depth+1)
            else:
                return multiplier * cache[self.id][0]

        vic = self.winner()
        if vic == perspective: # if we win
            cache[self.id] = (LARGE_NUM, perspective)
            return LARGE_NUM // (depth+1) # return a depth-adjusted large_num to encourage fastest win
        elif vic == get_next_turn(perspective): # if enemy wins
            cache[self.id] = (-LARGE_NUM, perspective)
            return -LARGE_NUM // (depth+1) # return depth-adjusted small num to encourage longest lose

        f_self = 0
        f_opp = 0
        midline = self.height // 2
        r = self.height % 2 # test even / odd
        for i in range(self.height):
            for j in range(self.width):
                piece = self.board[i][j]
                if piece.lower() == perspective: # If it's our own piece
                    if piece.isupper(): # if piece is king
                        f_self += 4
                    else: # if piece is pawn, check if it is advanced
                        if (piece == 'r' and i<midline) or (piece == 'b' and i>=(midline+r)):
                            f_self += 2
                        else:
                            f_self += 1
                elif piece.lower() == get_next_turn(perspective): # If it is an enemy piece
                    if piece.isupper(): # if piece is king
                        f_opp += 4
                    else: # if piece is pawn, check if it is advanced
                        if (piece == 'r' and i<midline) or (piece == 'b' and i>=(midline+r)):
                            f_opp += 2
                        else:
                            f_opp += 1
        f = f_self - f_opp
        cache[self.id] = (f, perspective)
        return f
                          
    
def jump_helper(board, i, j, piece, height, width) -> tuple[bool, list[list[list[str]]]]:
    """
    On the given board with particular coordinates pointing to a piece,
    try to make jumps. return whether we made any jumps and the list of 
    boards if jumps are made.
    """
    is_pawn = False
    jumped = False
    result = []

    if piece.islower():
        jump_dir = ((-2, 2), (-2, -2)) if piece == 'r' else ((2, 2), (2, -2))
        edge = 0 if piece == 'r' else height - 1 # The edge that makes pawn a king
        is_pawn = True
    else:
        jump_dir = ((-2, 2), (-2, -2), (2, 2), (2, -2))

    for d in jump_dir:
        new_i, new_j = i+d[0], j+d[1]
        if 0 <= new_i < height and 0 <= new_j < width:
            # Check if there's a piece to jump over
            jumped_i, jumped_j = i+d[0]//2, j+d[1]//2
            if board[jumped_i][jumped_j] in get_opp_char(piece) and board[new_i][new_j] == '.':
                # Make the jump
                jumped = True
                new_board = copy.deepcopy(board)
                new_board[i][j] = '.'
                new_board[jumped_i][jumped_j] = '.'
                if is_pawn and new_i == edge: # pawn becomes king
                    piece = piece.upper()
                new_board[new_i][new_j] = piece 
                temp1, temp2 = jump_helper(new_board, new_i, new_j, piece, height, width)
                if not temp1: # if we cant make future jumps
                    result.append(new_board)
                else:
                    result.extend(temp2)

    return jumped, result


                

        

#########################################   ALPHA-BETA PRUNING   #########################################
# Pseudo-code idea taken from https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
def minimax(state: State, depth: int, alpha: int, beta: int, perspective: str, max_depth: int):
    """
    Minimax search to find the best solution for the current state
    """
    if depth == max_depth or state.winner() is not None:
        return (None, state.util(perspective, depth))

    if state.turn == perspective: # if we are on the maximizing player's turn
        max_value = -LARGE_NUM
        max_state = None
        children = state.generate_successors()
        children.sort(key=lambda x: x.util(perspective, depth), reverse=True) # Sort in descending order for max's children
        for child in children:
            # Find opponent's best move after we made some move
            _, value = minimax(child, depth + 1, alpha, beta, perspective, max_depth)
            if value > max_value: # If we found a better state with a better value
                max_value = value
                max_state = child
            alpha = max(alpha, max_value)
            if beta <= alpha:
                break
        return (max_state, max_value)

    else: # if we are on the minimizing player's turn
        min_value = LARGE_NUM
        min_state = None
        children = state.generate_successors()
        children.sort(key=lambda x: x.util(perspective, depth), reverse=False) # Sort in ascending for min's children
        for child in children:
            _, value = minimax(child, depth + 1, alpha, beta, perspective, max_depth)
            if value < min_value:
                min_value = value
                min_state = child
            beta = min(beta, min_value)
            if beta <= alpha:
                break
        return (min_state, min_value)


def display_minimax_search(state: State, max_depth: int) -> None:
    curr_state = state
    counter = 0
    start_time = time.time()

    # While we haven't found a winner
    while curr_state.winner() is None:

        # print(f"curr player:{curr_state.turn}")
        curr_state.display()
        counter += 1
        # Perform minimax search with the curr_player being the maximizing player
        curr_state = minimax(curr_state, 0, -LARGE_NUM, LARGE_NUM, state.turn, max_depth)[0]

    # print the final board after victory
    curr_state.display()
    end_time = time.time()
    print(f"{counter} moves")
    print(f"{end_time - start_time} seconds")

    return None



def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']

def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'

def read_from_file(filename):

    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()

    return board

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    # initial_board = read_from_file("checkers3.txt")

    state = State(initial_board, 'r')
    sys.stdout = open(args.outputfile, 'w')
    print(f"solving puzzle  {args.inputfile}")
    display_minimax_search(state, 9)
    sys.stdout = sys.__stdout__
