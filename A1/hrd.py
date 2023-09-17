from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys

#====================================================================================

char_goal = '1'
char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()


    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()


        

class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(str(board.grid))  # The id for breaking ties.


    def __lt__(self, other):
        """
        Compare function for priority queue.
        """
        return self.f < other.f

    


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^': # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<': # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)
    
    return board


#========================================== Helper Functions ==========================================
import copy

def is_goal(s: State) -> bool:
    """
    Return true iff the state is a goal state
    """
    b = s.board
    piece_lst = b.pieces

    # Iterate over the pieces and check if the goal piece is in the correct position
    for p in piece_lst:
        if p.is_goal:
            if p.coord_x == 1 and p.coord_y == 3:
                return True
    return False


def get_successors(s: State) -> list[State]:
    """
    Return a list of possible successor states from the given state s
    """

    result = []
    grid = s.board.grid
    d = s.depth + 1

    for p in s.board.pieces:
        x = p.coord_x
        y = p.coord_y
        
        if p.orientation == 'h':
            if x - 1 >= 0 and grid[y][x - 1] == '.':
                # Move piece to left
                p.coord_x -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x += 1
                result.append(new_state)

            if x + 2 < len(grid[0]) and grid[y][x + 2] == '.':
                # Move piece to right
                p.coord_x += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x -= 1
                result.append(new_state)
                
            if y - 1 >= 0 and grid[y - 1][x] == '.' and grid[y - 1][x + 1] == '.':
                # Move piece up
                p.coord_y -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y += 1
                result.append(new_state)

            if y + 1 < len(grid) and grid[y + 1][x] == '.' and grid[y + 1][x + 1] == '.':
                # Move piece down
                p.coord_y += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y -= 1
                result.append(new_state)


        elif p.orientation == 'v':
            if x - 1 >= 0 and grid[y][x - 1] == '.' and grid[y + 1][x - 1] == '.':
                # Move piece to left
                p.coord_x -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x += 1
                result.append(new_state)

            if x + 1 < len(grid[0]) and grid[y][x + 1] == '.' and grid[y + 1][x + 1] == '.':
                # Move piece to right
                p.coord_x += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x -= 1
                result.append(new_state)

            if y - 1 >= 0 and grid[y - 1][x] == '.':
                # Move piece up
                p.coord_y -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y += 1
                result.append(new_state)

            if y + 2 < len(grid) and grid[y + 2][x] == '.':
                # Move piece down
                p.coord_y += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y -= 1
                result.append(new_state)


        elif p.is_single:
            if x - 1 >= 0 and grid[y][x - 1] == '.':
                # Move piece to left
                p.coord_x -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x += 1
                result.append(new_state)

            if x + 1 < len(grid[0]) and grid[y][x + 1] == '.':
                # Move piece to right
                p.coord_x += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x -= 1
                result.append(new_state)

            if y - 1 >= 0 and grid[y - 1][x] == '.':
                # Move piece up
                p.coord_y -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y += 1
                result.append(new_state)

            if y + 1 < len(grid) and grid[y + 1][x] == '.':
                # Move piece down
                p.coord_y += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y -= 1
                result.append(new_state)

        elif p.is_goal:
            if x - 1 >= 0 and grid[y][x - 1] == '.' and grid[y + 1][x - 1] == '.':
                # Move piece to left
                p.coord_x -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x += 1
                result.append(new_state)

            if x + 2 < len(grid[0]) and grid[y][x + 2] == '.' and grid[y + 1][x + 2] == '.':
                # Move piece to right
                p.coord_x += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_x -= 1
                result.append(new_state)

            if y - 1 >= 0 and grid[y - 1][x] == '.' and grid[y - 1][x + 1] == '.':
                # Move piece up
                p.coord_y -= 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y += 1
                result.append(new_state)

            if y + 2 < len(grid) and grid[y + 2][x] == '.' and grid[y + 2][x + 1] == '.':
                # Move piece down
                p.coord_y += 1
                new_board = Board(copy.deepcopy(s.board.pieces))
                f = d + compute_heuristic(new_board)
                new_state = State(new_board, f, d, s)
                p.coord_y -= 1
                result.append(new_state)

    return result
            


def find_solution_path(s: State) -> list[State]:
    """
    Given a goal state s, back-track to find the path from the initial state to s.
    """
    path = []
    while s.parent is not None:
        path.append(s)
        s = s.parent

    path.append(s) # Add the initial state

    # Note the path is in reverse-order, so reverse it to have the first state as initial state, and last state as the goal
    path.reverse()
    return path



def compute_heuristic(b: Board) -> int:
    """
    For a given board, compute the manhattan distance from the goal piece to the goal.
    """
    # Find the goal piece
    piece_list = b.pieces
    goal_piece = None
    for p in piece_list:
        if p.is_goal:
            goal_piece = p
            break

    if goal_piece == None:
        raise Exception("No goal piece found.")

    # Compute manhattan distance
    goal_x = (b.width / 2) - 1
    goal_y = b.height - 2

    actual_x = goal_piece.coord_x
    actual_y = goal_piece.coord_y

    return abs(1 - actual_x) + abs(3 - actual_y)



def seen_board(b1: Board, b_set: set[Board]) -> bool:
    """
    Returns true iff there is some board in b_set with the same grid as b1.
    """

    for b2 in b_set:
        if compare_boards(b1, b2):
            return True

    return False



def compare_boards(b1: Board, b2: Board) -> bool:
    """
    Returns true iff b1 has the same grid as b2.
    """
    for i in range(b1.height):
        for j in range(b1.width):
            if b1.grid[i][j] != b2.grid[i][j]:
                return False
    return True


#====================================== Stack Class ======================================
class Stack:
    def __init__(self):
        self.stack = []
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        if self.is_empty():
            return None
        else:
            return self.stack.pop()
    
    def is_empty(self):
        return len(self.stack) == 0
    


import heapq
class PriorityQueue:
    """ 
    A priority queue implemented using heap queue, where the priority is the f value of the state.
    Lower f <-> higher priority.
    """

    def __init__(self):
        self.queue = []
    
    def put(self, item):
        heapq.heappush(self.queue, item)
    
    def pop(self):
        return heapq.heappop(self.queue)
    
    def is_empty(self):
        return len(self.queue) == 0
    


#====================================== A* and DFS ======================================

def DFS(initial_state: State):

    counter = 0
    frontier = Stack()
    frontier.push(initial_state)
    explored = dict()

    while not frontier.is_empty():
        curr = frontier.pop()
    
        # curr.board.display()
        # print('\n')

        if curr.id in explored:
            # If we have seen this state before, skip it
            continue

        explored[curr.id] = curr.depth

        if is_goal(curr):
            # We found a solution
            return curr
    
        # Get successors
        successors = get_successors(curr)
        for s in successors:
            frontier.push(s)

    # Otherwise, no solution is found
    return None






def A_star(initial_state: State):
    """
    A* search algorithm.
    """
    frontier = PriorityQueue()
    explored = {}

    # frontier.put((initial_state.f, initial_state))
    frontier.put(initial_state)

    while not frontier.is_empty():
        curr = frontier.pop()

        # curr.board.display()
        # print('\n')
        # print(f"depth:{curr.depth}, heuristic:{curr.f}")
        # print('\n')

        if curr.id in explored and curr.f >= explored[curr.id]:
            # If we have seen this state (with better f value), we skip the current state
            continue

        explored[curr.id] = curr.f

        if is_goal(curr):
            # We found a solution
            return curr
        
        # Get successors
        successors = get_successors(curr)
        for s in successors:
            # frontier.put((s.f, s))
            frontier.put(s)

    # Otherwise, no solution is found
    return None










if __name__ == "__main__":
    # board = read_from_file('testhrd_hard1.txt')
    # s = State(board, compute_heuristic(board), 0, None)
    # result_state = A_star(s)
    # # result_state = DFS(s)

    # if result_state != None:
    #     path = find_solution_path(result_state)
    #     for s in path:
    #         s.board.display()
    #         print("")

    # assert False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()



    # read the board from the file
    board = read_from_file(args.inputfile)
    s = State(board, compute_heuristic(board), 0, None)

    if args.algo == 'astar':
        result_state = A_star(s)

    elif args.algo == 'dfs':
        result_state = DFS(s)

    else:
        raise Exception("Invalid algorithm.")
    
    original_stdout = sys.stdout

    with open(args.outputfile, 'w') as f:
        sys.stdout = f
        if result_state != None:
            path = find_solution_path(result_state)
            for s in path:
                s.board.display()
                print("")
        else:
            print("No solution found.")

        sys.stdout = original_stdout
    




