import copy
import argparse
import time
import sys

# Queue implementation borrowed from https://www.programiz.com/dsa/queue
class Queue: 
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0
    

def init_example_CSP():
    """
    Initialize an example CSP, useful for debugging
    """
    row_c = (2,1,1,2,2,2)
    col_c = (1,4,0,2,1,2)
    num_c = (3,2,1,0)
    new_csp = CSP(row_c, col_c, num_c)
    return new_csp

def read_from_file(filename):
    """
    Read from a given file to output a CSP plus an assignment
    """
    f = open(filename)
    lines = f.readlines()  # list of strings

    row_constraints = tuple([int(item) for item in lines[0].strip()])
    col_constraints = tuple([int(item) for item in lines[1].strip()])
    num_constraints = tuple([int(item) for item in lines[2].strip()])
    csp = CSP(row_constraints, col_constraints, num_constraints)
    N = csp.size

    # Next, populate the assignment
    assignment = {}
    lines = lines[3:]
    for i in range(N):
        ith_row = lines[i]
        for j in range(N):
            char = ith_row[j]

            if char == 'S':  # if we see a sub
                assignment[(i,j)] = '1'
                if 0 < i:  # if we aren't on first row
                    assignment[(i-1,j)] = '0'
                if i < N-1:  # if we aren't on the last row
                    assignment[(i+1,j)] = '0'
                if 0 < j:  # if not on first column
                    assignment[(i,j-1)] = '0'
                if j < N-1:  # if not on last column
                    assignment[(i,j+1)] = '0'

            elif char == '<':
                assignment[(i,j)] = '1'
                if 0 < i:  
                    assignment[(i-1,j)] = '0'
                if i < N-1: 
                    assignment[(i+1,j)] = '0'
                if 0 < j:  
                    assignment[(i,j-1)] = '0'
                if j < N-1:
                    assignment[(i,j+1)] = '1'

            elif char == '>':
                assignment[(i,j)] = '1'
                if 0 < i:  
                    assignment[(i-1,j)] = '0'
                if i < N-1: 
                    assignment[(i+1,j)] = '0'
                if 0 < j:  
                    assignment[(i,j-1)] = '1'
                if j < N-1:  
                    assignment[(i,j+1)] = '0'

            elif char == '^':
                assignment[(i,j)] = '1'
                if 0 < i:  
                    assignment[(i-1,j)] = '0'
                if i < N-1: 
                    assignment[(i+1,j)] = '1'
                if j < N-1:  
                    assignment[(i,j+1)] = '0'
                if 0 < j:  
                    assignment[(i,j-1)] = '0'

            elif char == 'v':
                assignment[(i,j)] = '1'
                if 0 < i:  
                    assignment[(i-1,j)] = '1'
                if i < N-1: 
                    assignment[(i+1,j)] = '0'
                if j < N-1:  
                    assignment[(i,j+1)] = '0'
                if 0 < j:  
                    assignment[(i,j-1)] = '0'

            elif char == 'M':
                assignment[(i,j)] = '2'  # A special ship that must be differentiated from subs

            elif char == '.':
                assignment[(i,j)] = '0'

    return csp, assignment





# Note: when initializing domain, on the outer most edges we cannot have outer brackets
# e.g. '>' on the leftmost edge of the board

class CSP:
    def __init__(self, row_constraint, col_constraint, num_constraint):
        self.row_constraint = row_constraint
        self.col_constraint = col_constraint
        self.num_constraint = num_constraint  # tuple of (1,2,3,4 length-ed ships)
        self.size = self._get_width()
        self.domains = {} # domain of each variable

        # Populate domains
        for x in range(self.size):
            for y in range(self.size):
                self.domains[(x,y)] = {'1', '0'}
            

    def _get_width(self):
        """
        Get the length, or width (the same) for this board
        """
        N = len(self.row_constraint)
        return N
        

    def is_complete(self, assignment):
        """
        Is this assignment complete? 
        """
        N = self.size
        if len(assignment) != N**2:  # if we haven't assigned everything
            return False
        # else:
        # for i in range(N):
        #     parts_on_row_i = self._count_shipparts_on_row(i, assignment)
        #     if parts_on_row_i != self.row_constraint[i]:
        #         print("row unsatisfied")
        #         return False
                
        #     parts_on_col_i = self._count_shipparts_on_col(i, assignment)
        #     if parts_on_col_i != self.col_constraint[i]:
        #         print("col unsatisfied")
        #         return False
                
        # num_ships = self._count_total_ships(assignment)
        # if tuple(num_ships) != self.num_constraint:
        #     print("num unsatisfied")
        #     return False
        
        return True
    

    def is_correct(self, assignment):
        """
        Is this assignment correct?
        """
        N = self.size
        for i in range(N):
            parts_on_row_i = self._count_shipparts_on_row(i, assignment)
            if parts_on_row_i != self.row_constraint[i]:
                print(f"row unsatisfied on row {i}")
                return False
                
            parts_on_col_i = self._count_shipparts_on_col(i, assignment)
            if parts_on_col_i != self.col_constraint[i]:
                print(f"col unsatisfied on column {i}")
                return False
                
        num_ships = self._count_total_ships(assignment)
        if tuple(num_ships) != self.num_constraint:
            print("num unsatisfied")
            return False
        
        return True


    def print_board(self, assignment):
        """
        Print the current assignment
        """
        N = self.size
        for i in range(N):
            for j in range(N):
                if (i,j) in assignment:
                    print(assignment[(i,j)], end='')
                else:
                    print('X', end='')
            print()
        print()


    def portrait_completed_board(self, assignment):
        """
        Portrait the COMPLETED board from '0', '1' assignment to 
        the standardized one.
        """
        N = self.size
        for i in range(N):
            for j in range(N):
                if assignment[(i,j)] in ['1','2']:  # we know (i,j) is in assignment since assignment is COMPLETE
                    # possibilities = ['<', '>', 'v', '^', 'M', 'S']
                    if (j==0 or assignment[(i,j-1)] == '0') and (j<N-1 and assignment[(i,j+1)] in ['1','2']):
                        # if we have nothing on the left and something on the right
                        print('<', end='')
                    
                    elif (j==N-1 or assignment[(i,j+1)] == '0') and (j>0 and assignment[(i,j-1)] in ['1','2']):
                        # if nothing on the right and something on the left
                        print('>', end='')

                    elif (i==0 or assignment[(i-1,j)] == '0') and (i<N-1 and assignment[(i+1,j)] in ['1','2']):
                        # if nothing on top and something on bottom
                        print('^', end='')

                    elif (i==N-1 or assignment[(i+1,j)] == '0') and (i>0 and assignment[(i-1,j)] in ['1','2']):
                        # if nothing on bottom and something on top
                        print('v', end='')
                    
                    elif ((i-1,j) in assignment and (i+1,j) in assignment and assignment[(i-1,j)] in ['1','2'] and assignment[(i+1,j)] in ['1','2']) or\
                        ((i,j-1) in assignment and (i,j+1) in assignment and assignment[(i,j-1)] in ['1','2'] and assignment[(i,j+1)] in ['1','2']):
                        print('M', end='')
                    else:
                        print('S', end='')
                else:
                    print('.', end='')
            print() #newline
                        



    def copy_self(self):
        return copy.deepcopy(self)


    def backtrack(self, assignment):
        """
        Backtracking search with FC algo
        """
        self.print_board(assignment)  # for debugging

        if self.is_complete(assignment):
            return assignment
        
        var = self._select_unassigned_var(assignment) # Use the MRV heuristic to choose next variable
        for value in self._select_domain_value(var, assignment):  # Use the LCV heuristic to choose next value
            assignment[var] = value
            new_csp = self.copy_self()
            inferences = new_csp.inference(var, assignment)
            # Returns False upon failure, and True otherwise, delete all contrdicting arcs along the way
            if inferences is not False:
                result = new_csp.backtrack(assignment)
                if result is not None:
                    return result     
            del assignment[var] # If such an assignment does not lead to solution, cancel it
        return None

    

    def _get_unassigned_vars(self, assignment):
        """
        Return all unassigned variables in the csp
        """
        unassigned_vars = []
        for v in self.domains:
            if v not in assignment:
                unassigned_vars.append(v)
        return unassigned_vars
        

    def _select_unassigned_var(self, assignment):
        """
        Select the next unassigned variable using MRV heuristic (find var with smallest domain)
        """
        unassigned_vars = self._get_unassigned_vars(assignment)
        if len(unassigned_vars) != 0:
            return min(unassigned_vars, key=lambda x: len(self.domains[x]))
            # return unassigned_vars[0]
        else:
            return None
    

    def _select_domain_value(self, variable, assignment):
        """
        Order the domain values for 'variable' using LCV heuristic 

        ACTUALLY... Nah let's keep it simple
        """
        if variable is None:  # if the variable passed in is no good (None indicates no unassigned vars left)
            return []
        # else:
        return self.domains[variable]


        

    def _count_domain_sizes(self, assignment):
        """
        Sum up the number of options in the current domain
        """
        count = 0
        unassigned = self._get_unassigned_vars(assignment)
        for key in unassigned:
            count += len(self.domains[key])
        return count

    

    def inference(self, variable, assignment):
        """
        The Forward Checking Algo, check after assigning 'variable' with 'value'.
        Return False if inconsistency found, True otherwise.
        Shrink the domains within the CSP along the way.
        """
        # After assigning variable with value, check:
        # its row and column for row col constraints
        # num ship constraints  (limits how long we can construct the ship)
        # water constraint (all blocks on diagonal should be water)

        # if point is a ship part, then its surrounding undetermined points are influenced by total ship num.
        # if point not ship part, then it is water or undertermined, surrounding are not influenced

        for v in self._get_unassigned_vars(assignment): # assignment has been populated, so we won't encounter 'variable'
            result1 = self.num_consistent(assignment, v)
            # shrink v's domain to make v have consistent number of ships
            if result1 is False: # if failed
                return False
            
            result2 = self.row_consistent(variable, assignment, v)  # make v consistent if v is on same row to 'variable'
            if result2 is False:
                return False
            
            result3 = self.col_consistent(variable, assignment, v)  # make v consistent if v is on same column to 'variable'
            if result3 is False:
                return False
            
            result4 = self.water_consistent(assignment, v)
            if result4 is False:
                return False
            
            result5 = self.mid_section_consistent(assignment, v)
            if result5 is False:
                return False
            
        return True
    


    

    def mid_section_consistent(self, assignment, unassigned):
        """
        We don't want mid-sections to be surrounded by water,
        so when we assign 'unassigned' with '0', check if it
        is near a designated mid-section, and if it completes
        water surrounded by that section, it is false!
        """
        neighbors = self._get_nbhood(assignment, unassigned)
        for key in neighbors:
            if neighbors[key] == '2':  # if this var is near a mid-section
                assignment_copy = copy.deepcopy(assignment)
                assignment_copy[unassigned] = '0'  # try assigning water to unassigned
                non_water = 0  # a mid-section must have EXACTLY 2 non-waters on its sides
                # keep track of the potential non-waters near a mid-section
                
                ship_neighbors = self._get_nbhood(assignment_copy, key)
                for n in ship_neighbors:
                    if ship_neighbors[n] in ['-1', '1', '2']:  # if the mid-section is near unassigned, a shippart, or a designated mid-section:
                        non_water += 1

                if non_water < 2:  # if we can't achieve EXACTLY 2 non-waters on its sides
                    self.domains[unassigned].discard('0')

        if len(self.domains[unassigned]) == 0:
            return False
        return  True

    
    def _get_nbhood(self, assignment, var):
        """
        Get the neighborhood of 'var' as a dict,
        key = possible neighbor, 
        val = assigned value or -1 if unassigned
        """
        N = self.size
        nbhood = {}
        x, y = var
        if 0 < x:  # if possible to check up
            nbhood[(x-1, y)] = '-1'
            if (x-1, y) in assignment:  # check up
                nbhood[(x-1, y)] = assignment[(x-1, y)]

        if x < N-1:
            nbhood[(x+1, y)] = '-1'
            if (x+1, y) in assignment:  # check down
                nbhood[(x+1, y)] = assignment[(x+1, y)]

        if 0 < y:
            nbhood[(x, y-1)] = '-1'
            if (x, y-1) in assignment:  # check left
                nbhood[(x, y-1)] = assignment[(x, y-1)]

        if y < N-1:
            nbhood[(x, y+1)] = '-1'
            if (x, y+1) in assignment:  # check right
                nbhood[(x, y+1)] = assignment[(x, y+1)]
        return nbhood
        
    

    def num_consistent(self, assignment, unassigned):
        """
        Check if 'unassigned' has consistent domain, if not, 
        prune the inconsistent elements, return False if we
        pruned everything. True otherwise.
        """
        # What would happen if we assign '1' to unassigned?
        # Well, if assigning this '1' completes a ship with a
        # length that is unavailable, it is inconsistent
        x1, x2, x3, x4 = self.num_constraint  # returns the total # of constraining ships
        assignment_copy = copy.deepcopy(assignment)

        assignment_copy[unassigned] = '1'  # Assign this with '1'
        y1, y2, y3, y4 = self._count_total_ships(assignment_copy) # count # of ships after new assignment
        if (x1-y1 < 0) or (x2-y2 < 0) or (x3-y3 < 0) or (x4-y4 < 0) or (y1 < 0):
            # if any ship count is negative, that is, we exceeded the allocated number of ships
            # or if we see a ship of length 5 or more (which is impossible!)
            self.domains[unassigned].discard('1')
        
        assignment_copy[unassigned] = '0'  # Now change the assigned value to '0' and try again
        y1, y2, y3, y4 = self._count_total_ships(assignment_copy)
        if (x1-y1 < 0) or (x2-y2 < 0) or (x3-y3 < 0) or (x4-y4 < 0) or (y1 < 0):
            self.domains[unassigned].discard('0')
        
        if len(self.domains[unassigned]) == 0:
            return False
        return  True


    def _count_total_ships(self, assignment):
        """
        Count the # of completed ships in assignment

        # ALSO: Return [-1,-1,-1,-1] if we see an incomplete ship but with 5 or more ship parts
        """
        # break into N rows and N columns of assignments
        N = self.size
        row_assignments = [{} for i in range(N)]
        col_assignments = [{} for i in range(N)]
        num_ships = [0, 0, 0 ,0]

        for var in assignment:
            x, y = var  # x is the row number from 0 to N-1
            row_assignments[x][var] = assignment[var]  # allocate this key-value pair to the specific row
            col_assignments[y][var] = assignment[var]

        # Check row-wise for valid long ships  (Not subs)
        for i in range(N):  
            ith_row = row_assignments[i]
            j = 0  # initial column
            while j < N:
                if (i,j) in ith_row and ith_row[(i,j)] == '1':  # if we are on a ship part
                    if ((i,j-1) in ith_row and ith_row[(i,j-1)] == '0') or j==0:  # if this ship part is the start of a ship
                        len = 0
                        while (i,j) in ith_row and ith_row[(i,j)] in ['1', '2'] and len<5:
                            len += 1
                            j += 1
                        # when this is done, either (i,j) not in ith_row, or ith_row[(i,j)] is water.
                        # or length of possible ship >= 5

                        # if ship length is bad, return [-1,-1,-1,-1] this instance, as we know this is impossible already
                        if len >= 5:
                            return [-1,-1,-1,-1]
                        
                        # if (i,j) not assigned, do nothing, we can't conclude ship length yet
                        # However:
                        if len > 1: # we are not considering subs now
                            if j == N or (i,j) in ith_row:  # if the next step is the void, or water
                                num_ships[len - 1] += 1
                j += 1  # Move onto the next cell

        # Check column-wise for valid long ships  (Also not subs)
        for i in range(N):
            ith_column = col_assignments[i]
            j = 0 # initial row
            while j < N:
                if (j,i) in ith_column and ith_column[(j,i)] == '1': 
                    if ((j-1,i) in ith_column and ith_column[(j-1,i)] == '0') or j==0:
                        len = 0
                        while (j,i) in ith_column and ith_column[(j,i)] in ['1', '2'] and len<5:  # while we keep seeing the ship
                            len += 1
                            j += 1
                        if len >= 5:
                            return [-1,-1,-1,-1]
                        if len > 1:
                            if j == N or (j,i) in ith_column: # Got to the bottom of the column or we saw water
                                num_ships[len - 1] += 1
                j += 1

        # Check subs
        for poten_sub in assignment:
            if assignment[poten_sub] == '1':
                x,y = poten_sub
                if x == 0 or ((x-1,y) in assignment and assignment[(x-1,y)] == '0'):  # Void or water above
                    if x == N-1 or ((x+1,y) in assignment and assignment[(x+1,y)] == '0'):  # Below
                        if y == 0 or ((x,y-1) in assignment and assignment[(x,y-1)] == '0'):  # Left
                            if y == N-1 or ((x,y+1) in assignment and assignment[(x,y+1)] == '0'):  # Right
                                num_ships[0] += 1

        return num_ships



    def water_consistent(self, assignment, unassigned):
        """
        Check if assigning unassigned with '1' 
        contradicts water constraint.
        Note: The water constraint here is weakened
        and we only consider: "diagonals of ship parts
        has to be waters"
        """
        # Check if unassigned has any '1' in its diagonals
        # Equivalent to checking if any '1' has diagonals with unassigned

        x, y = unassigned
        top_left = (x-1, y-1)
        top_right = (x-1, y+1)
        down_left = (x+1, y-1)
        down_right = (x+1, y+1)

        if top_left in assignment and assignment[top_left] in ['1', '2']:  # if we have a ship part on the top left
            self.domains[unassigned].discard('1') # Then we can't assign 'unassigned' with a ship part

        elif top_right in assignment and assignment[top_right] in ['1', '2']:
            self.domains[unassigned].discard('1')

        elif down_left in assignment and assignment[down_left] in ['1', '2']:
            self.domains[unassigned].discard('1')
        
        elif down_right in assignment and assignment[down_right] in ['1', '2']:
            self.domains[unassigned].discard('1')
        
        if len(self.domains[unassigned]) == 0:
            return False
        return True



    

    def row_consistent(self, variable, assignment, unassigned):
        """
        Prune 'unassigned' if it is on the same row as 'variable'
        Return True if pruning domain did not result in DWO (domain wipeout).
        Return False otherwise.

        Only considers unassigned variables on the same row as variable
        """
        x,_ = variable
        y,_ = unassigned
        if x == y:  # if on the same row (starting from 0)
            row_index = x
            constraint_on_row_x = self.row_constraint[row_index]
            current_ships = self._count_shipparts_on_row(row_index, assignment)

            # if we assign 'unassigned' with '1': 
            # problem arises when we get over the capacity
            if current_ships >= constraint_on_row_x: # if we are full on row x
                # Throw away every ship segment
                self.domains[unassigned].discard('1')
                if len(self.domains[unassigned]) == 0: # if failure
                    return False
            
            # if we assign 'unassigned' with '0':
            # we face the chance of never fulfilling the requirement in the given slots,
            # so the plan is doomed to fail
            parts_to_fill = constraint_on_row_x - current_ships
            empty_slots = self._count_empty_slots_on_row(row_index, assignment) - 1
            # print(f"parts to fill {parts_to_fill}")
            # print(f"slots to fill {empty_slots}")
            # minus 1 because we are assuming we assigned 'unassigned' with '0'
            if parts_to_fill > empty_slots:  # if we have more parts than spaces avilable
                self.domains[unassigned].discard('0')
                if len(self.domains[unassigned]) == 0:
                    return False
        return True
    

    def _count_empty_slots_on_row(self, index, assignment):
        """
        Count the number of unassigned spaces on index'th row
        """
        count = 0
        unassigned = self._get_unassigned_vars(assignment)
        for key in unassigned:
            x, _ = key
            if x == index:
                count += 1
        return count


    
    def _count_shipparts_on_row(self, index, assignment):
        """
        Count the number of ship segments on the index'th row
        """
        count = 0
        for key in assignment:
            x, _ = key
            if x == index and assignment[key] in ['1', '2']:
                # if the cell on the designated row has a ship part
                count += 1
        return count


    def col_consistent(self, variable, assignment, unassigned):
        """
        Prune 'unassigned' if it is on the same column as 'variable'
        Return True if pruning domain did not result in DWO (domain wipeout).
        Return False otherwise.
        """
        _, x = variable
        _, y = unassigned
        if x == y:  # if on the same column (starting from 0)
            column_index = x
            constraint_on_col_x = self.col_constraint[column_index]
            current_ships = self._count_shipparts_on_col(column_index, assignment)

            # if assign 'unassigned' with '1'
            if current_ships >= constraint_on_col_x: # if we are full on column x
                # Throw away every ship segment
                self.domains[unassigned].discard('1')
                if len(self.domains[unassigned]) == 0: # if failure
                    return False
                
            # if assign 'unassigned' with '0'
            parts_to_fill = constraint_on_col_x - current_ships
            empty_slots = self._count_empty_slots_on_col(column_index, assignment) - 1
            # minus 1 because we are assuming we assigned 'unassigned' with '0'
            if parts_to_fill > empty_slots:  # if we have more parts than spaces avilable
                self.domains[unassigned].discard('0')
                if len(self.domains[unassigned]) == 0:
                    return False
        return True
    

    def _count_empty_slots_on_col(self, index, assignment):
        """
        Count # of unassigned blocks on the index'th column
        """
        count = 0
        unassigned = self._get_unassigned_vars(assignment)
        for key in unassigned:
            _, x = key
            if x == index:
                count += 1
        return count
    
    
    def _count_shipparts_on_col(self, index, assignment):
        """
        Count the number of ship segments on the index'th row
        """
        count = 0
        for key in assignment:
            _, x = key
            if x == index and assignment[key] in ['1', '2']:
                # if the cell on the designated row has a ship part
                count += 1
        return count
    


if __name__ == '__main__':
    start_time = time.time()
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

    csp, assignment = read_from_file(args.inputfile)
    # csp, assignment = read_from_file('input_easy1.txt')
    new_assignment = csp.backtrack(assignment)

    sys.stdout = open(args.outputfile, 'w')
    # x = csp.is_correct(new_assignment)
    # print(f"Is this correct? {x}")
    # csp.print_board(new_assignment)
    csp.portrait_completed_board(new_assignment)
    sys.stdout = sys.__stdout__
    end_time = time.time()
    print(f"time taken: {end_time - start_time}")
