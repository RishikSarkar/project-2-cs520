import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # Use this to see full 30x30 numpy matrix
np.set_printoptions(linewidth=1000) # Use this to fit each row in a single line without breaks

# Function to find current open cells in grid
def find_all_open_cells(grid):
    open_cells = set()

    for i in range(30):
        for j in range(30):
            if grid[i, j] == 0:
                open_cells.add((i, j))

    return open_cells

# Function to check if a cell has one open neighbor
def single_open_neighbor_check(i, j, grid):
    count = 0

    if (i + 1 < 30) and (grid[i + 1, j] == 0):
        count += 1
    if (i - 1 >= 0) and (grid[i - 1, j] == 0):
        count += 1
    if (j + 1 < 30) and (grid[i, j + 1] == 0):
        count += 1
    if (j - 1 >= 0) and (grid[i, j - 1] == 0):
        count += 1

    return count == 1

# Function to open a random blocked cell with one open neighbor
def open_random_single_neighbor_cell(grid, num_open_neighbor_cells):
    open_neighbor_cells = set()

    # Find open neighbor cells
    for i in range(30):
        for j in range(30):
            if (grid[i, j] == 1) and (single_open_neighbor_check(i, j, grid)):
                open_neighbor_cells.add((i, j))
    
    # Store total number of open neighbor cells
    num_open_neighbor_cells = len(open_neighbor_cells)

    # No more single open neighbor cells
    if num_open_neighbor_cells == 0:
        return grid, num_open_neighbor_cells
    
    # Pick a random blocked cell from set and open it
    random_cell = random.choice(tuple(open_neighbor_cells))
    grid[random_cell] = 0

    return grid, num_open_neighbor_cells

# Function to open a random closed neighbor
def open_random_closed_neighbor(i, j, grid):
    closed_neighbors = set()

    if (i + 1 < 30) and (grid[i + 1, j] == 1):
        closed_neighbors.add((i + 1, j))
    if (i - 1 >= 0) and (grid[i - 1, j] == 1):
        closed_neighbors.add((i - 1, j))
    if (j + 1 < 30) and (grid[i, j + 1] == 1):
        closed_neighbors.add((i, j + 1))
    if (j - 1 >= 0) and (grid[i, j - 1] == 1):
        closed_neighbors.add((i, j - 1))

    if len(closed_neighbors) > 0:
        random_neighbor = random.choice(tuple(closed_neighbors))
        grid[random_neighbor[0], random_neighbor[1]] = 0

    return grid

# Function to create an empty 30x30 grid
def create_grid():
    grid = np.full((30, 30), 1) # Create a new 30x30 numpy matrix (2D array)
    grid[random.randrange(1, 29), random.randrange(1, 29)] = 0 # Open a random blocked cell (except on edges and corners)

    num_open_neighbor_cells = 900

    # Iteratively open all single open neighbor cells
    while (num_open_neighbor_cells > 0):
        grid, num_open_neighbor_cells = open_random_single_neighbor_cell(grid, num_open_neighbor_cells)
    
    dead_end_cells = set()

    # Find dead-end cells
    for i in range(30):
        for j in range(30):
            if (grid[i, j] == 0) and (single_open_neighbor_check(i, j, grid)):
                dead_end_cells.add((i, j))

    # Keep roughly half of the dead-end cells found
    dead_end_cells = set(random.choices(list(dead_end_cells), k=(len(dead_end_cells) // 2)))

    # Open a random closed neighbor for each of the dead-end cells in the set
    for i in dead_end_cells:
        grid = open_random_closed_neighbor(i[0], i[1], grid)

    # Find current open cells in grid
    open_cells = find_all_open_cells(grid)

    return grid, open_cells

# Open up all unblocked cells in grid
def reset_grid(grid, open_cells):
    for i in range(30):
        for j in range(30):
            if grid[i, j] != 1:
                grid[i, j] = 0
    
    open_cells = find_all_open_cells(grid)
    return grid, open_cells

# Function to place the bot at a random open cell in the grid
def place_bot(grid, open_cells):
    bot = random.choice(tuple(open_cells)) # Pick a random cell
    grid[bot[0], bot[1]] = 2 # Place bot in grid and label space as 2
    open_cells.remove(bot)

    return bot, grid, open_cells

# Function to place the crew at a random open cell in the grid (must occur after placing bot)
def place_crew(grid, open_cells, crew_list):
    crew = random.choice(tuple(open_cells - set(crew_list))) # Pick a random cell without an existing crew member
    grid[crew[0], crew[1]] = 4 # Place crew member in grid and label space as 4
    crew_list.append(crew)

    return crew_list, grid

# Function to place an alien at a unoccupied cell in the grid
def place_alien(grid, open_cells, alien_list, bot, detect_square_k):
    alien = None
    # Make sure alien is not in detection square
    while True:
        alien = random.choice(tuple(open_cells - set(alien_list))) # Pick open cell that is unoccupied by another alien
        if not ((alien[0] >= (bot[0]-detect_square_k)) and (alien[0] <= (bot[0]+detect_square_k))) and not ((alien[1] >= (bot[1]-detect_square_k)) and (alien[1] <= (bot[1]+detect_square_k))):
            break
    
    grid[alien[0]][alien[1]] = 3 # Place alien in grid and denote space as 3
    alien_list.append(alien)
    return alien_list, grid

# Function to ensure that potential cells to move to is in bounds
def check_valid_neighbors(dim, x_coord, y_coord):
    neigh_list = []
    neigh_list.append((x_coord+1, y_coord))
    neigh_list.append((x_coord-1, y_coord))
    neigh_list.append((x_coord, y_coord+1))
    neigh_list.append((x_coord, y_coord-1))
    main_list = []
    for neigh in neigh_list:
        if neigh[0] <= dim-1 and neigh[0] >= 0 and neigh[1] <= dim-1 and neigh[1] >= 0:
            main_list.append(neigh)
    return main_list

# Move aliens randomly to adjacent cells
def move_aliens(grid, alien_list, bot):
    # random.shuffle(alien_list)
    new_position = []
    marker = 0
    for alien in alien_list:
        possible_moves = check_valid_neighbors(len(grid), alien[0], alien[1])
        possible_moves.append(alien)
        # Get all spatially possible coordinates that the alien can move to 
        while possible_moves:
            current = random.choice(possible_moves)
            # Check if cell is open 
            if grid[current[0]][current[1]] == 1:
                possible_moves.remove(current)
                continue
            # Check if alien captures bot
            if current == bot:
                marker = 1
                new_position.append(current)
                new_position.extend(alien_list[alien_list.index(alien)+1: len(alien_list)])
                return marker, new_position, grid
            
            grid[alien[0]][alien[1]] = 0
            grid[current[0]][current[1]] = 3
            new_position.append(current)
            break 
        
    return marker, new_position, grid

# Sensor to detect aliens within a (2k + 1) x (2k + 1) square around bot
def alien_sensor(alien_list, bot, k):
    bot_x_max = min(bot[0] + k, 29) # k cells to the right of bot
    bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
    bot_y_max = min(bot[1] + k, 29) # k cells to the top of bot
    bot_y_min = max(0, bot[1] - k) # k cells to the bottom of bot

    # Check if each alien is within the detection square
    for alien in alien_list:
        if (alien[0] >= bot_x_min and alien[0] <= bot_x_max) and (alien[1] >= bot_y_min and alien[1] <= bot_y_max):
            return True

    return False

# Updated Crew Sensor to avoid having to run BFS every step
def crew_sensor(grid, bot, alpha, d_lookup_table, crew_list):
    d_dict = {}

    # Case in which bot has never been to cell before
    if d_lookup_table.get(bot) == None:
        seen_cells = set()
        bfs_queue = []
        bfs_queue.append(bot)
        seen_cells.add(bot)
        d_dict[bot[0], bot[1]] = 0

        # Use BFS to find shortest path cost from bot to all cells
        while len(bfs_queue) > 0:
            curr_cell = bfs_queue.pop(0)

            neighbors = check_valid_neighbors(30, curr_cell[0], curr_cell[1])

            for neighbor in neighbors:
                if grid[neighbor[0], neighbor[1]] != 1 and neighbor not in seen_cells:
                    seen_cells.add(neighbor)
                    bfs_queue.append(neighbor)
                    d_dict[neighbor[0], neighbor[1]] = d_dict[curr_cell[0], curr_cell[1]] + 1 # Set distance of neighbor to current cell's distance + 1
        d_lookup_table[bot] = d_dict # Forgot this line I think - Aditya (Thanks for adding! - Rishik)

    # Case in which bot has been to cell before (and knows distance to closest crew member)
    else:
        d_dict = d_lookup_table[bot]

    d_min = 901

    # Find d for closest crew member
    for crew_member in crew_list:
        if d_dict[crew_member] < d_min:
            d_min = d_dict[crew_member]
    
    if d_min == 901:
        return False, d_lookup_table # Don't beep if no crew member found

    prob = math.exp(-alpha * (d_min - 1))
    
    return np.random.choice([True, False], p=[prob, 1 - prob]), d_lookup_table # Beep with the specified probability

# Create alien probability matrix (dictionary) for t = 0
def initialize_alienmatrix(open_cells, bot, k):
    open_cells.add(bot)

    bot_x_max = min(bot[0] + k, 29) # k cells to the right of bot
    bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
    bot_y_max = min(bot[1] + k, 29) # k cells to the top of bot
    bot_y_min = max(0, bot[1] - k) # k cells to the bottom of bot

    alien_matrix = {}
    for cell in open_cells:
        if bot_x_min <= cell[0] <= bot_x_max and bot_y_min <= cell[1] <= bot_y_max:
            alien_matrix[cell] = 0
        else:
            alien_matrix[cell] = 1  # Temporary value, to be normalized later

    # Normalize probabilities for cells where the alien can be
    valid_cells_count = len(open_cells) - sum(value == 0 for value in alien_matrix.values())
    for cell, prob in alien_matrix.items():
        if prob != 0:
            alien_matrix[cell] = 1 / valid_cells_count

    return alien_matrix

# Create crew probability matrix (dictionary) for t = 0
def initialize_crewmatrix(open_cells, crew_list, bot):
    open_cells.add(bot)
    # Crew member can be at any open cell except the ones occupied by the bot or another crew
    inital_prob = [1/(len(open_cells) - (1 + (len(crew_list)-1)))] * len(open_cells)
    crew_matrix = dict(zip(open_cells, inital_prob))
    bot_cell = {bot : 0}
    crew_matrix.update(bot_cell)
    open_cells.remove(bot)

    return crew_matrix

# Update probabilties for alien matrix based on beep
def update_alienmatrix(alien_matrix, detected, bot, k):
    alien_keys = set(alien_matrix.keys())

    if detected:
        # Cells outside detection square should have probability 0
        out_detection_cells = {key for key in alien_keys if not ((bot[0]-k <= key[0] <= bot[0]+k) and (bot[1]-k <= key[1] <= bot[1]+k))}
        for cell in out_detection_cells:
            alien_matrix[cell] = 0

        # Normalize probabilities inside detection square
        in_detection_cells = alien_keys - out_detection_cells
        prob_sum = sum(alien_matrix[cell] for cell in in_detection_cells)
        if prob_sum != 0:
            for cell in in_detection_cells:
                alien_matrix[cell] /= prob_sum
    else:
        # Cells inside detection square should have probability 0
        detection_cells = {key for key in alien_keys if (bot[0]-k <= key[0] <= bot[0]+k) and (bot[1]-k <= key[1] <= bot[1]+k)}
        for cell in detection_cells:
            alien_matrix[cell] = 0

        # Normalize probabilities outside detection square
        outside_cells = alien_keys - detection_cells
        prob_sum = sum(alien_matrix[cell] for cell in outside_cells)
        if prob_sum != 0:
            for cell in outside_cells:
                alien_matrix[cell] /= prob_sum

    return alien_matrix

# Update probabilties for alien matrix based on beep (in the case of 2 aliens)
def update_alienmatrix_2alien(alien_matrix, alien_detected, bot, k, index_mapping, open_cells):
    if alien_detected:
        in_detectionsqaure = []
        for cell in open_cells:
            if ((bot[0]-k <= cell[0] <= bot[0]+k) and (bot[1]-k <= cell[1] <= bot[1]+k)):
                in_detectionsqaure.append(cell)
        
        outside_detectionsqaure = list(open_cells - set(in_detectionsqaure))
        for cell in outside_detectionsqaure:
            for c in outside_detectionsqaure:
                alien_matrix[index_mapping[cell]][index_mapping[c]] = 0

        total_sum = 0
        for cell in in_detectionsqaure:
            total_sum = total_sum + np.sum(alien_matrix[index_mapping[cell]]) + np.sum(alien_matrix[:, index_mapping[cell]]) 
        if total_sum != 0:
            alien_matrix = alien_matrix * (1/total_sum)
    else:
        in_detectionsqaure = []
        for cell in open_cells:
            if ((bot[0]-k <= cell[0] <= bot[0]+k) and (bot[1]-k <= cell[1] <= bot[1]+k)):
                in_detectionsqaure.append(cell)
        
        outside_detectionsqaure = list(open_cells - set(in_detectionsqaure))
        for cell in in_detectionsqaure:
            alien_matrix[index_mapping[cell]] = 0
            alien_matrix[:, index_mapping[cell]] = 0
        
        total_sum = np.sum(alien_matrix)
        alien_matrix = alien_matrix / total_sum

    return alien_matrix

# Update probabilties for crew matrix based on beep
def update_crewmatrix(crew_matrix, detected, d_lookup_table, bot, alpha):
    # Case where beep is detected from bot cell
    if detected:
        d_dict = d_lookup_table.get(bot) # Get the d dictionary calculated with the crew sensor
        total_summation = 0
        for cell in crew_matrix:
            d = d_dict.get(cell) # Find d from bot to cell
            if cell == bot:
                crew_matrix[cell] = 0 # Crew member not at current cell
            else:
                crew_matrix[cell] *= math.exp(-alpha * (d - 1)) # Multiply probability of cell containing crew by given prob
            total_summation += crew_matrix[cell] # Calculate sum of all probabilities
        
        for key in crew_matrix:
            crew_matrix[key] /= total_summation # Normalize probabilities
    # Case where beep is not detected from bot cell
    else:
        d_dict = d_lookup_table.get(bot) # Get the d dictionary calculated with the crew sensor
        total_summation = 0
        for cell in crew_matrix:
            d = d_dict.get(cell) # Find d from bot to cell
            if cell == bot:
                crew_matrix[cell] = 0 # Crew member not at current cell
            else:
                crew_matrix[cell] *= (1 - math.exp(-alpha * (d - 1))) # Multiply probability of cell containing crew by 1 - given prob
            total_summation += crew_matrix[cell] # Calculate sum of all probabilities
        
        for key in crew_matrix:
            crew_matrix[key] /= total_summation # Normalize probabilities

    return crew_matrix

# Function to move bot to specified cell (Makes sense to do probability updates and move decision in functions for each Bot, since other factors to consider)
def move_bot(grid, bot, new_cell, crew_list, alien_list, open_cells, win_count, bot_num):
    # Add new bot location to open cells set and remove old one. Modify grid accordingly
    open_cells.add(bot)
    grid[bot[0], bot[1]] = 0
    grid[new_cell[0], new_cell[1]] = 2
    open_cells.remove(new_cell)
    bot = new_cell
    marker = 0

    # Case where bot lands on the same cell as a crew member
    for crew_member in crew_list:
        if bot == crew_member:
            # ^ In the case of 2 crew members, this only matters if bot has saved both crew members. Shall evaluate probability of saving crew as mentioned in https://cs520f23.zulipchat.com/#narrow/stream/409248-project-2/topic/Questions.20on.20the.20evaluation/near/401403344
            crew_list.remove(crew_member)

            if bot_num <= 2:
                win_count += 1 # Increment win count because crew member has been saved
                crew_list, grid = place_crew(grid, open_cells, crew_list)
            else:
                if len(crew_list) == 0:
                    win_count += 1 # Increment win count because both crew members have been saved
                    crew_list, grid = place_crew(grid, open_cells, crew_list)
                    crew_list, grid = place_crew(grid, open_cells, crew_list)

            return bot, crew_list, grid, open_cells, win_count, marker
        
    # Case where bot lands on the same cell as an alien
    for alien in alien_list:
        if bot == alien:
            marker = 1

            return bot, crew_list, grid, open_cells, win_count, marker
    
    return bot, crew_list, grid, open_cells, win_count, marker

# Recalculate probability matrices after bot move    
def update_afterbotmove(bot, alien_matrix, crew_matrix):
    # Prior Probability alien not in current cell
    alienprob_not = 1 - alien_matrix[bot]
    # Prob alien not in current cell
    crew_probnot = 1 - crew_matrix[bot]
    # Bot did not capture or die so we know current cell does not have crew or alien
    alien_matrix[bot] = 0
    crew_matrix[bot] = 0
    
    for cell in alien_matrix:
        alien_matrix[cell] = alien_matrix[cell] / alienprob_not
    for cell in crew_matrix:
        crew_matrix[cell] = crew_matrix[cell] / crew_probnot

    return alien_matrix, crew_matrix

# Determine the best neighboring cell for the bot to move to based on probability matrices
def determine_move(moves, alien_matrix, crew_matrix):
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    nonzero_crewprob = [move for move in moves if crew_matrix[move] != 0]

    def find_max_prob_cell(cell_list, matrix):
        max_prob = -1
        candidates = []
        for cell in cell_list:
            current_prob = matrix[cell]

            # Find max probability cell
            if current_prob > max_prob:
                max_prob = current_prob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_prob == max_prob:
                candidates.append(cell) # Add cell to list if same probability as max

        return random.choice(candidates) if candidates else None

    # Case at least one move with 0 alien probability
    if zero_alienprob:
        chosen_move = find_max_prob_cell(zero_alienprob, crew_matrix)
    # Case where at least one move with nonzero crew probability
    elif nonzero_crewprob:
        chosen_move = find_max_prob_cell(nonzero_crewprob, crew_matrix)
    else:
        chosen_move = random.choice(moves)

    return chosen_move

# Determine the best neighboring cell for Bot 2 to move to
# Bot 2 prioritizes saving crew members over escaping aliens, unlike Bot 1
def determine_move_bot2(moves, alien_matrix, crew_matrix):
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    nonzero_crewprob = [move for move in moves if crew_matrix[move] != 0]

    def find_max_prob_cell(cell_list, matrix):
        max_prob = -1
        candidates = []
        for cell in cell_list:
            current_prob = matrix[cell]

            # Find max probability cell
            if current_prob > max_prob:
                max_prob = current_prob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_prob == max_prob:
                candidates.append(cell) # Add cell to list if same probability as max

        return random.choice(candidates) if candidates else None

    # Bot 2 switches the two cases, and prioritizes cells that have a nonzero probability of a crew member being in it
    # Case where at least one move with nonzero crew probability
    if nonzero_crewprob:
        chosen_move = find_max_prob_cell(nonzero_crewprob, crew_matrix)
    # Case at least one move with 0 alien probability
    elif zero_alienprob:
        chosen_move = find_max_prob_cell(zero_alienprob, crew_matrix)
    else:
        chosen_move = random.choice(moves)

    return chosen_move

# Update probabilities after alien moves
def update_afteralienmove(ship, alien_list, alien_matrix):
    total_summation = 0
    neighbors = check_valid_neighbors(len(ship), alien_list[0][0], alien_list[0][1])
    neighbors = [neigh for neigh in neighbors if ship[neigh[0], neigh[1]] != 1]
    for neigh in neighbors:
        if neigh in alien_matrix:
            total_summation = total_summation + (alien_matrix[neigh] * (1/(len(neighbors)+1))) # Plus 1 because alien could stay in place 
    
    alien_matrix[alien_list[0]] = total_summation
    return alien_matrix

# Update probabilities after aliens move (consider both alien locations)
def update_afteralienmove_2alien(ship, alien_list, alien_matrix, index_mapping):
    neighbor_alien1 = [neigh for neigh in check_valid_neighbors(len(ship), alien_list[0][0], alien_list[0][1]) if ship[neigh] != 1]
    neighbor_alien2 = [neigh for neigh in check_valid_neighbors(len(ship), alien_list[1][0], alien_list[1][1]) if ship[neigh] != 1]
    len_of_firstlist = len(neighbor_alien1)
    neighbor_alien1.extend(neighbor_alien2)
    
    alien_matrix *= 0
    for neigh in neighbor_alien1:
        for n in neighbor_alien1:
            if (neigh in neighbor_alien1 and n in neighbor_alien2) or (n in neighbor_alien1 and neigh in neighbor_alien2):
                alien_matrix[index_mapping[neigh]][index_mapping[n]] = alien_matrix[index_mapping[neigh]][index_mapping[n]] + ((1/len_of_firstlist) * (1/len(neighbor_alien2)))
        neighbor_alien1.remove(neigh)
    return alien_matrix

# Function to initialize the crew member probabilities in each cell as a dictionary of open cells
def initialize_crewmatrix_2crew(bot, open_cells):
    open_cells.add(bot)
    
    index_mapping = {cell: index for index, cell in enumerate(open_cells)} # Gives an index to each cell in open_cells + bot's current location
    index_count = len(index_mapping) # Stores number of open cells + bot's current location

    # shape = (len(grid),len(grid),len(grid),len(grid))
    shape = (index_count, index_count)
    crew_matrix = np.ones(shape)
    # crew_matrix = crew_matrix * (1 / ((len(open_cells) * len(open_cells)) - len(open_cells)))
    crew_matrix = crew_matrix * (1 / ((index_count * index_count) - index_count))
    open_cells.remove(bot)
    crew_matrix[index_mapping[bot]] *= 0 # Set bot row prob to 0
    crew_matrix[:, index_mapping[bot]] *= 0
    return crew_matrix, index_mapping

# Determine the best neighboring cell for the bot to move to based on probability matrices
def determine_move_2crew(moves, alien_matrix, crew_matrix, index_mapping):
    
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    nonzero_crewprob = [move for move in moves if crew_matrix[index_mapping[move]].sum() != 0]
    chosen_cell = None

    def find_max_prob_cell(cell_list):
        max_crewprob = -1
        candidates = []
        for cell in cell_list:
            cell_index = index_mapping[cell] # Retrieve index mapping of current cell
            current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
            current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

            # Find max probability cell
            if current_crewprob > max_crewprob:
                max_crewprob = current_crewprob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_crewprob == max_crewprob:
                candidates.append(cell) # Add cell to list if same probability as max
                
        return random.choice(candidates) if candidates else None # Randomly break ties

    # Case at least one move with 0 alien probability
    if zero_alienprob:
        # print("0")
        chosen_cell = find_max_prob_cell(zero_alienprob)
    # Case at least one move with nonzero crew probability
    elif nonzero_crewprob:
        # print("1")
        chosen_cell = find_max_prob_cell(nonzero_crewprob)
    else:
        # print("2")
        chosen_cell = find_max_prob_cell(moves)

    return chosen_cell

# Determine the best neighboring cell for Bot 5 to move to
# Bot 5 finds the direction of the closest crew member using the previously calculated d values and prioritizes its general direction while breaking ties
def determine_move_bot5(moves, alien_matrix, crew_matrix, index_mapping, crew_list, bot, d_lookup_table):
    
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    nonzero_crewprob = [move for move in moves if crew_matrix[index_mapping[move]].sum() != 0]
    chosen_cell = None

    # Find the closest crew member based on the d_dict calculated when crew sensor activates
    closest_crew = crew_list[0]
    closest_crew_d = 901
    d_dict = d_lookup_table.get(bot)
    if d_dict != None:
        for crew_member in crew_list:
            if d_dict.get(crew_member) < closest_crew_d:
                closest_crew_d = d_dict.get(crew_member)
                closest_crew = crew_member

    # Find the general quadrant in which the closest crew lies (if d_dict is not available, find relative position of first crew in crew_list)
    up = down = left = right = False

    if closest_crew[0] >= bot[0]:
        down = True
    else:
        up = True

    if closest_crew[1] >= bot[1]:
        right = True
    else:
        left = True

    def find_max_prob_cell(cell_list):
        max_crewprob = -1
        candidates = []
        for cell in cell_list:
            cell_index = index_mapping[cell] # Retrieve index mapping of current cell
            current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
            current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

            # Find max probability cell
            if current_crewprob > max_crewprob:
                max_crewprob = current_crewprob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_crewprob == max_crewprob:
                candidates.append(cell) # Add cell to list if same probability as max

        # Filter out the cells that are in the direction of the nearest crew member
        direction_filtered_candidates = []
        for cell in candidates:
            if (up and cell[0] <= bot[0]) or (down and cell[0] >= bot[0]) or (left and cell[1] <= bot[1]) or (right and cell[1] >= bot[1]):
                direction_filtered_candidates.append(cell)

        # Return a random neighbor out of filtered candidates if found
        if direction_filtered_candidates:
            return random.choice(direction_filtered_candidates)
        
        # If not filtered candidates found, randomly break ties
        return random.choice(candidates) if candidates else None

    # Prioritize saving crew members
    # Case at least one move with nonzero crew probability
    if nonzero_crewprob:
        # print("1")
        chosen_cell = find_max_prob_cell(nonzero_crewprob)
    # Case at least one move with 0 alien probability
    elif zero_alienprob:
        chosen_cell = find_max_prob_cell(zero_alienprob)
    else:
        chosen_cell = find_max_prob_cell(moves)

    return chosen_cell

# Determine the best neighboring cell for the bot to move to based on probability matrices (2 alien)
def determine_move_2crew2alien(moves, alien_matrix, crew_matrix, index_mapping):
    
    zero_alienprob = [move for move in moves if np.sum(alien_matrix[move]) == 0]
    # print(zero_alienprob)
    nonzero_crewprob = [move for move in moves if crew_matrix[index_mapping[move]].sum() != 0]
    chosen_cell = None

    def find_max_prob_cell(cell_list):
        max_crewprob = -1
        candidates = []
        for cell in cell_list:
            cell_index = index_mapping[cell] # Retrieve index mapping of current cell
            current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
            current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

            # Find max probability cell
            if current_crewprob > max_crewprob:
                max_crewprob = current_crewprob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_crewprob == max_crewprob:
                candidates.append(cell) # Add cell to list if same probability as max
        # print(f"Candidates : {candidates}")        
        return random.choice(candidates) if candidates else None # Randomly break ties

    # Case at least one move with 0 alien probability
    if zero_alienprob:
        # print("0")
        chosen_cell = find_max_prob_cell(zero_alienprob)
    # Case at least one move with nonzero crew probability
    elif nonzero_crewprob:
        # print("1")
        chosen_cell = find_max_prob_cell(nonzero_crewprob)
    else:
        # print("2")
        chosen_cell = find_max_prob_cell(moves)

    return chosen_cell

# Determine the best neighboring cell for Bot 8 to move to
# Bot 8 finds the safest quadrant and the highest probability quadrant for the crew by determining the sum of the probabilities of aliens and crew being in a half and prioritizes movements towards those quadrants while breaking ties
def determine_move_bot8(moves, alien_matrix, crew_matrix, index_mapping, bot):
    
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    nonzero_crewprob = [move for move in moves if crew_matrix[index_mapping[move]].sum() != 0]
    chosen_cell = None

    # Calculate alien probability sums for each quadrant
    up_alien_sum = alien_matrix[:bot[0], :].sum()
    down_alien_sum = alien_matrix[bot[0]+1:, :].sum()
    left_alien_sum = alien_matrix[:, :bot[1]].sum()
    right_alien_sum = alien_matrix[:, bot[1]+1:].sum()

    up_crew_sum = crew_matrix[:bot[0], :].sum()
    down_crew_sum = crew_matrix[bot[0]+1:, :].sum()
    left_crew_sum = crew_matrix[:, :bot[1]].sum()
    right_crew_sum = crew_matrix[:, bot[1]+1:].sum()

    # Determine the safest quadrant and quadrant with highest crew probability
    safest_min_sum = min(up_alien_sum, down_alien_sum, left_alien_sum, right_alien_sum)
    highest_crew_sum = max(up_crew_sum, down_crew_sum, left_crew_sum, right_crew_sum)

    # Find the safe quadrants
    up_safe = down_safe = left_safe = right_safe = False
    if up_alien_sum == safest_min_sum:
        up_safe = True
    if down_alien_sum == safest_min_sum:
        down_safe = True
    if left_alien_sum == safest_min_sum:
        left_safe = True
    if right_alien_sum == safest_min_sum:
        right_safe = True

    # Find the quadrants with highest crew probability
    up_crew = down_crew = left_crew = right_crew = False
    if up_crew_sum == highest_crew_sum:
        up_crew = True
    if down_crew_sum == highest_crew_sum:
        down_crew = True
    if left_crew_sum == highest_crew_sum:
        left_crew = True
    if right_crew_sum == highest_crew_sum:
        right_crew = True

    def find_max_prob_cell(cell_list):
        max_crewprob = -1
        candidates = []
        for cell in cell_list:
            cell_index = index_mapping[cell] # Retrieve index mapping of current cell
            current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
            current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

            # Find max probability cell
            if current_crewprob > max_crewprob:
                max_crewprob = current_crewprob
                candidates = [cell] # Reset list of highest probability cells if new max found
            elif current_crewprob == max_crewprob:
                candidates.append(cell) # Add cell to list if same probability as max

        # Filter out the cells that are in the direction of the safest quadrants
        direction_filtered_candidates = []
        for cell in candidates:
            if ((up_safe or up_crew) and cell[0] >= bot[0]) or ((down_safe or down_crew) and cell[0] <= bot[0]) or ((left_safe or left_crew) and cell[1] >= bot[1]) or ((right_safe or right_crew) and cell[1] <= bot[1]):
                direction_filtered_candidates.append(cell)

        # Return a random neighbor out of filtered candidates if found
        if direction_filtered_candidates:
            return random.choice(direction_filtered_candidates)
        
        # If not filtered candidates found, randomly break ties
        return random.choice(candidates) if candidates else None

    # Prioritize saving crew (but more carefully)
    # Case at least one move with nonzero crew probability
    if nonzero_crewprob:
        chosen_cell = find_max_prob_cell(nonzero_crewprob)
    # Case at least one move with 0 alien probability
    elif zero_alienprob:
        chosen_cell = find_max_prob_cell(zero_alienprob)
    else:
        chosen_cell = find_max_prob_cell(moves)

    return chosen_cell

# Update probabilities in matrices when the bot moves (in the case of 2 crew members)
def update_afterbotmove_2crew(bot, alien_matrix, crew_matrix, index_mapping):
    # Prior Probability alien not in current cell
    alienprob_not = 1 - alien_matrix[bot]
    # Prob alien not in current cell
    
    # Bot did not capture or die so we know current cell does not have crew or alien
    alien_matrix[bot] = 0
    
    for cell in alien_matrix:
        alien_matrix[cell] = alien_matrix[cell] / alienprob_not

    bot_crewmatrix_index = index_mapping[bot] # Find index assigned to the bot's cell in the crew matrix

    # Set probability of crew being in bot's current position to 0
    crew_matrix[bot_crewmatrix_index, :] = 0
    crew_matrix[:, bot_crewmatrix_index] = 0

    # Find the current total sum of the crew matrix probabilities
    total_sum = np.sum(crew_matrix)

    # Normalize rest of the crew matrix
    crew_matrix = crew_matrix / total_sum if total_sum != 0 else crew_matrix

    return alien_matrix, crew_matrix

# Update probabilities in matrices when the bot moves (in the case of 2 crew members + 2 aliens)
def update_afterbotmove_2crew2alien(bot, alien_matrix, crew_matrix, index_mapping_alien, index_mapping_crew):
    bot_alienmatrix_index = index_mapping_alien[bot]

    alien_matrix[bot_alienmatrix_index, :] = 0
    alien_matrix[:, bot_alienmatrix_index] = 0

    total_sum_alien = np.sum(alien_matrix)

    # Normalize rest of the crew matrix
    alien_matrix = alien_matrix / total_sum_alien if total_sum_alien != 0 else alien_matrix

    bot_crewmatrix_index = index_mapping_crew[bot] # Find index assigned to the bot's cell in the crew matrix

    # Set probability of crew being in bot's current position to 0
    crew_matrix[bot_crewmatrix_index, :] = 0
    crew_matrix[:, bot_crewmatrix_index] = 0

    # Find the current total sum of the crew matrix probabilities
    total_sum = np.sum(crew_matrix)

    # Normalize rest of the crew matrix
    crew_matrix = crew_matrix / total_sum if total_sum != 0 else crew_matrix

    return alien_matrix, crew_matrix

# Update probabilties for crew matrix based on beep
def update_crewmatrix_2crew(crew_matrix, detected, d_lookup_table, bot, alpha, index_mapping, open_cells):
    d_dict = d_lookup_table.get(bot) # Get the d dictionary calculated with the crew sensor
    if detected:
        for cell in open_cells:
            crew_matrix[index_mapping[cell]] *= -1 * (1 - math.exp(-alpha * (d_dict[cell] - 1)))
        for cell in open_cells:
            crew_matrix[:,index_mapping[cell]] *= -1 * (1 - math.exp(-alpha * (d_dict[cell] - 1)))
        crew_matrix *= -1
        crew_matrix += 1
        crew_matrix = crew_matrix / np.sum(crew_matrix)
    else:
        for cell in open_cells:
            crew_matrix[index_mapping[cell]] *= -1 * (1 - math.exp(-alpha * (d_dict[cell] - 1)))
        for cell in open_cells:
            crew_matrix[:,index_mapping[cell]] *= -1 * (1 - math.exp(-alpha * (d_dict[cell] - 1)))
        crew_matrix = crew_matrix / np.sum(crew_matrix)

    return crew_matrix

# Function to initialize the alien probabilities in each cell as a dictionary of open cells (for 2 aliens)
def initialize_alienmatrix_2alien(open_cells, bot, k):
    open_cells.add(bot)
    
    bot_x_max = min(bot[0] + k, 29) # k cells to the right of bot
    bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
    bot_y_max = min(bot[1] + k, 29) # k cells to the top of bot
    bot_y_min = max(0, bot[1] - k) # k cells to the bottom of bot
    
    
    index_mapping = {cell: index for index, cell in enumerate(open_cells)} # Gives an index to each cell in open_cells + bot's current location
    index_count = len(index_mapping) # Stores number of open cells + bot's current location

    in_detectioncell = [bot]
    for cell in open_cells:
        if bot_x_min <= cell[0] <= bot_x_max and bot_y_min <= cell[1] <= bot_y_max:
            in_detectioncell.append(cell)
    
    # shape = (len(grid),len(grid),len(grid),len(grid))
    shape = (index_count, index_count)
    alien_matrix = np.ones(shape)
    for cell in in_detectioncell:
        alien_matrix[index_mapping[cell]] *= 0
        alien_matrix[:, index_mapping[cell]] *= 0
   
    # # crew_matrix = crew_matrix * (1 / ((len(open_cells) * len(open_cells)) - len(open_cells)))
    alien_matrix = alien_matrix * (1 / ((index_count * index_count) - (index_count*len(in_detectioncell))))
    open_cells.remove(bot)

    return alien_matrix, index_mapping





# 1 crew 1 alien bot 
def Bot1(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 1 crew member + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move(open_moves, alien_matrix, crew_matrix)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 1)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Crew saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove(bot, alien_matrix, crew_matrix)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix(crew_matrix, crew_detected, d_lookup_table, bot, alpha) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 1 crew 1 alien bot (Prioritizes saving crew member over escaping alien)
def Bot2(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 1 crew member + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move_bot2(open_moves, alien_matrix, crew_matrix)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 2)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Crew saved! Win Count: {win_count}, Loss Count: {loss_count}")

            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove(bot, alien_matrix, crew_matrix)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix(crew_matrix, crew_detected, d_lookup_table, bot, alpha) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew 1 alien bot
def Bot3(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move(open_moves, alien_matrix, crew_matrix)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 3)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove(bot, alien_matrix, crew_matrix)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix(crew_matrix, crew_detected, d_lookup_table, bot, alpha) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew, 1 alien (Considering probabilities for both crew members)
def Bot4(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move_2crew(open_moves, alien_matrix, crew_matrix, index_mapping_crew)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 4)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove_2crew(bot, alien_matrix, crew_matrix, index_mapping_crew)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix_2crew(crew_matrix, crew_detected, d_lookup_table, bot, alpha, index_mapping_crew, open_cells) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew, 1 alien (Prioritizes saving crew member over escaping alien and breaks ties by moving in the general direction of the closest crew member)
def Bot5(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move_bot5(open_moves, alien_matrix, crew_matrix, index_mapping_crew, crew_list, bot, d_lookup_table)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 5)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove_2crew(bot, alien_matrix, crew_matrix, index_mapping_crew)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix_2crew(crew_matrix, crew_detected, d_lookup_table, bot, alpha, index_mapping_crew, open_cells) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew 2 alien bot
def Bot6(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 2 aliens
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot, k)
    crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move(open_moves, alien_matrix, crew_matrix)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 6)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove(bot, alien_matrix, crew_matrix)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}

            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix = initialize_alienmatrix(open_cells, bot, k)
            crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien sensor

        crew_matrix = update_crewmatrix(crew_matrix, crew_detected, d_lookup_table, bot, alpha) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew 2 alien bot (Considering probabilities for both crew members and both aliens)
def Bot7(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 2 aliens
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
    crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move_2crew2alien(open_moves, alien_matrix, crew_matrix, index_mapping_crew)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 7)
       
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove_2crew2alien(bot, alien_matrix, crew_matrix, index_mapping_alien, index_mapping_crew)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove_2alien(ship, alien_list, alien_matrix, index_mapping_alien) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        # print(alien_detected)
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix_2alien(alien_matrix, alien_detected, bot, k, index_mapping_alien, open_cells) # Update based on alien sensor

        crew_matrix = update_crewmatrix_2crew(crew_matrix, crew_detected, d_lookup_table, bot, alpha, index_mapping_crew, open_cells) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# 2 crew 2 alien bot (Custom Bot)
def Bot8(k, alpha, max_iter, timeout):
    grid, open_cells = create_grid()
    bot, ship, open_cells = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 2 crew members + 2 aliens
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    crew_list, ship = place_crew(ship, open_cells, crew_list)

    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
    crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)

    win_count = 0
    loss_count = 0
    move = 0
    win_move_count = []
    marker = 0

    while (win_count + loss_count) < max_iter:
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        open_moves.append(bot) # Bot can stay in place 
        next_move = determine_move_bot8(open_moves, alien_matrix, crew_matrix, index_mapping_crew, bot)
        
        prev_win_count = win_count
        bot, crew_list, ship, open_cells, win_count, marker = move_bot(ship, bot, next_move, crew_list, alien_list, open_cells, win_count, 8)
        move += 1

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue

        if win_count > prev_win_count:
            print(f"Both crew members saved! Win Count: {win_count}, Loss Count: {loss_count}")
            win_move_count.append(move)
            move = 0
            d_lookup_table = {}
            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
        
        print(f"Bot: {bot}, Crew: {crew_list}, Aliens: {alien_list}")

        alien_matrix, crew_matrix = update_afterbotmove_2crew2alien(bot, alien_matrix, crew_matrix, index_mapping_alien, index_mapping_crew)

        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly

        if marker == 1 or move >= timeout:
            loss_count += 1
            print(f"Bot captured! Win Count: {win_count}, Loss Count: {loss_count}")

            grid, open_cells = reset_grid(grid, open_cells)
            bot, ship, open_cells = place_bot(grid, open_cells)
            crew_list = []
            alien_list = []
            d_lookup_table = {}
            
            crew_list, ship = place_crew(ship, open_cells, crew_list)
            crew_list, ship = place_crew(ship, open_cells, crew_list)

            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)
            alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

            alien_matrix, index_mapping_alien = initialize_alienmatrix_2alien(open_cells, bot, k)
            crew_matrix, index_mapping_crew = initialize_crewmatrix_2crew(bot, open_cells)
            marker = 0
            move = 0

            continue
        
        alien_matrix = update_afteralienmove_2alien(ship, alien_list, alien_matrix, index_mapping_alien) # Update after alien move
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        
        alien_matrix = update_alienmatrix_2alien(alien_matrix, alien_detected, bot, k, index_mapping_alien, open_cells) # Update based on alien sensor

        crew_matrix = update_crewmatrix_2crew(crew_matrix, crew_detected, d_lookup_table, bot, alpha, index_mapping_crew, open_cells) # Update based on crew sensor 

    return sum(win_move_count) // max(1, len(win_move_count)), (win_count / max(1, (win_count + loss_count))), win_count





# Bot 1 vs. Bot 2

def Bot1_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot1(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)

    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

def Bot2_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot2(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)
            
    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

# Helper function to plot graphs for Bot 1 and Bot 2 for each alpha and k-value
def plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_data, bot2_data, title, metric_num):
    save_dir = './data/final/1v2'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate and save a plot for each k-value
    for k in k_values:
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, bot1_data[k], label=f'Bot 1, k={k}')
        plt.plot(alpha_values, bot2_data[k], label=f'Bot 2, k={k}')
        plt.title(f'{title} (k={k})')
        plt.xlabel('alpha')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

        filename = os.path.join(save_dir, f'metric{metric_num}_k{k}.png')
        plt.savefig(filename)
        plt.close()

def one_alien_one_crew(alpha_values, k_values, max_iter, timeout):
    bot1_avg_rescue_moves, bot1_prob_crew_rescue, bot1_avg_crew_saved = Bot1_simulation(alpha_values, k_values, max_iter, timeout)
    bot2_avg_rescue_moves, bot2_prob_crew_rescue, bot2_avg_crew_saved = Bot2_simulation(alpha_values, k_values, max_iter, timeout)

    bot1_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot1_prob_crew_rescue.items()}
    bot2_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot2_prob_crew_rescue.items()}

    print(bot1_avg_rescue_moves, bot1_prob_crew_rescue, bot1_avg_crew_saved, "\n")
    print(bot2_avg_rescue_moves, bot2_prob_crew_rescue, bot2_avg_crew_saved, "\n")

    plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_avg_rescue_moves, bot2_avg_rescue_moves, 'Average Rescue Moves', 1)
    plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_prob_crew_rescue, bot2_prob_crew_rescue, 'Probability of Crew Rescue', 2)
    plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_avg_crew_saved, bot2_avg_crew_saved, 'Average Crew Saved', 3)





# Bot 3 vs. Bot 4 vs. Bot 5

def Bot3_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot3(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)

    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

def Bot4_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot4(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)
            
    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

def Bot5_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot5(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)
            
    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

# Helper function to plot graphs for Bot 3, Bot 4, and Bot 5 for each alpha and k-value
def plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_data, bot4_data, bot5_data, title, metric_num):
    save_dir = './data/final/3v4v5'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate and save a plot for each k-value
    for k in k_values:
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, bot3_data[k], label=f'Bot 3, k={k}')
        plt.plot(alpha_values, bot4_data[k], label=f'Bot 4, k={k}')
        plt.plot(alpha_values, bot5_data[k], label=f'Bot 5, k={k}')
        plt.title(f'{title} (k={k})')
        plt.xlabel('alpha')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

        filename = os.path.join(save_dir, f'metric{metric_num}_k{k}.png')
        plt.savefig(filename)
        plt.close()

def one_alien_two_crew(alpha_values, k_values, max_iter, timeout):
    bot3_avg_rescue_moves, bot3_prob_crew_rescue, bot3_avg_crew_saved = Bot3_simulation(alpha_values, k_values, max_iter, timeout)
    bot4_avg_rescue_moves, bot4_prob_crew_rescue, bot4_avg_crew_saved = Bot4_simulation(alpha_values, k_values, max_iter, timeout)
    bot5_avg_rescue_moves, bot5_prob_crew_rescue, bot5_avg_crew_saved = Bot5_simulation(alpha_values, k_values, max_iter, timeout)

    bot3_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot3_prob_crew_rescue.items()}
    bot4_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot4_prob_crew_rescue.items()}
    bot5_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot5_prob_crew_rescue.items()}

    print(bot3_avg_rescue_moves, bot3_prob_crew_rescue, bot3_avg_crew_saved, "\n")
    print(bot4_avg_rescue_moves, bot4_prob_crew_rescue, bot4_avg_crew_saved, "\n")
    print(bot5_avg_rescue_moves, bot5_prob_crew_rescue, bot5_avg_crew_saved, "\n")

    plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_avg_rescue_moves, bot4_avg_rescue_moves, bot5_avg_rescue_moves, 'Average Rescue Moves', 1)
    plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_prob_crew_rescue, bot4_prob_crew_rescue, bot5_prob_crew_rescue, 'Probability of Crew Rescue', 2)
    plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_avg_crew_saved, bot4_avg_crew_saved, bot5_avg_crew_saved, 'Average Crew Saved', 3)





# Bot 6 vs. Bot 7 vs. Bot 8

def Bot6_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot6(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)

    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

def Bot7_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot7(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)
            
    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

def Bot8_simulation(alpha_values, k_values, max_iter, timeout):
    avg_rescue_moves = {k: [] for k in k_values}
    prob_crew_rescue = {k: [] for k in k_values}
    avg_crew_saved = {k: [] for k in k_values}

    for k in k_values:
        for alpha in alpha_values:
            metric1, metric2, metric3 = Bot8(k, alpha, max_iter, timeout)
            print(f"k: {k}, Alpha: {alpha}\nAverage Rescue Moves: {metric1}\nProbability of Crew Rescue: {metric2}\nAverage Crew Saved: {metric3}\n")
            avg_rescue_moves[k].append(metric1)
            prob_crew_rescue[k].append(metric2)
            avg_crew_saved[k].append(metric3)
            
    return avg_rescue_moves, prob_crew_rescue, avg_crew_saved

# Helper function to plot graphs for Bot 6, Bot 7, and Bot 8 for each alpha and k-value
def plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_data, bot7_data, bot8_data, title, metric_num):
    save_dir = './data/final/6v7v8'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate and save a plot for each k-value
    for k in k_values:
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, bot6_data[k], label=f'Bot 6, k={k}')
        plt.plot(alpha_values, bot7_data[k], label=f'Bot 7, k={k}')
        plt.plot(alpha_values, bot8_data[k], label=f'Bot 8, k={k}')
        plt.title(f'{title} (k={k})')
        plt.xlabel('alpha')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

        filename = os.path.join(save_dir, f'metric{metric_num}_k{k}.png')
        plt.savefig(filename)
        plt.close()

def two_alien_two_crew(alpha_values, k_values, max_iter, timeout):
    bot6_avg_rescue_moves, bot6_prob_crew_rescue, bot6_avg_crew_saved = Bot6_simulation(alpha_values, k_values, max_iter, timeout)
    bot7_avg_rescue_moves, bot7_prob_crew_rescue, bot7_avg_crew_saved = Bot7_simulation(alpha_values, k_values, max_iter, timeout)
    bot8_avg_rescue_moves, bot8_prob_crew_rescue, bot8_avg_crew_saved = Bot8_simulation(alpha_values, k_values, max_iter, timeout)

    bot6_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot6_prob_crew_rescue.items()}
    bot7_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot7_prob_crew_rescue.items()}
    bot8_prob_crew_rescue = {k: [round(prob, 3) for prob in probs] for k, probs in bot8_prob_crew_rescue.items()}

    print(bot6_avg_rescue_moves, bot6_prob_crew_rescue, bot6_avg_crew_saved, "\n")
    print(bot7_avg_rescue_moves, bot7_prob_crew_rescue, bot7_avg_crew_saved, "\n")
    print(bot8_avg_rescue_moves, bot8_prob_crew_rescue, bot8_avg_crew_saved, "\n")

    plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_avg_rescue_moves, bot7_avg_rescue_moves, bot8_avg_rescue_moves, 'Average Rescue Moves', 1)
    plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_prob_crew_rescue, bot7_prob_crew_rescue, bot8_prob_crew_rescue, 'Probability of Crew Rescue', 2)
    plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_avg_crew_saved, bot7_avg_crew_saved, bot8_avg_crew_saved, 'Average Crew Saved', 3)





# Testing Area

# Params: k, alpha, max_iter, timeout
# print(Bot1(3, 0.1, 2, 10000))
# print(Bot2(3, 0.1, 2, 10000))
# print(Bot3(3, 0.1, 2, 10000))
# print(Bot4(3, 0.1, 2, 10000))
# print(Bot5(3, 0.1, 2, 10000))
# print(Bot6(3, 0.1, 2, 10000))
# print(Bot7(3, 0.1, 2, 10000))
# print(Bot8(3, 0.1, 2, 10000))


alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
k_values = [1, 3, 5]
max_iter = 30
timeout = 10000

# alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
# k_values = [3]
# max_iter = 3
# timeout = 10000

# one_alien_one_crew(alpha_values, k_values, max_iter, timeout)
# one_alien_two_crew(alpha_values, k_values, max_iter, timeout)
# two_alien_two_crew(alpha_values, k_values, max_iter, timeout)





# Data and Analysis Area

# Bot 1 vs. Bot 2

bot1_metric1 = {}
bot1_metric1[1] = [1471, 622, 1866, 1514, 869]
bot1_metric1[3] = [1203, 1764, 1314, 1798, 2053]
bot1_metric1[5] = [1228, 2411, 1226, 1694, 1418]

bot1_metric2 = {}
bot1_metric2[1] = [0.667, 0.4, 0.869, 0.4, 0.429]
bot1_metric2[3] = [0.633, 0.433, 0.31, 0.433, 0.533]
bot1_metric2[5] = [0.467, 0.433, 0.433, 0.308, 0.467]

bot1_metric3 = {}
bot1_metric3[1] = [20, 12, 26, 12, 13]
bot1_metric3[3] = [19, 13, 9, 13, 16]
bot1_metric3[5] = [14, 13, 13, 9, 14]


bot2_metric1 = {}
bot2_metric1[1] = [1614, 2502, 1381, 1155, 2115]
bot2_metric1[3] = [2380, 1262, 1235, 1329, 2664]
bot2_metric1[5] = [1492, 1680, 1409, 1048, 1424]

bot2_metric2 = {}
bot2_metric2[1] = [0.533, 0.45, 0.533, 0.75, 0.5]
bot2_metric2[3] = [0.5, 0.6, 0.633, 0.4, 0.367]
bot2_metric2[5] = [0.433, 0.6, 0.5, 0.367, 0.567]

bot2_metric3 = {}
bot2_metric3[1] = [16, 14, 16, 22, 15]
bot2_metric3[3] = [15, 18, 19, 12, 11]
bot2_metric3[5] = [13, 18, 15, 11, 17]

plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_metric1, bot2_metric1, 'Average Rescue Moves', 1)
plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_metric2, bot2_metric2, 'Probability of Crew Rescue', 2)
plot_Bot1_vs_Bot2(alpha_values, k_values, bot1_metric3, bot2_metric3, 'Average Crew Saved', 3)


# Bot 3 vs. Bot 4 vs. Bot 5

bot3_metric1 = {}
bot3_metric1[1] = [1823, 1697, 2015, 2134, 2219]
bot3_metric1[3] = [1618, 1543, 1612, 1857, 2076]
bot3_metric1[5] = [1724, 1637, 1914, 2063, 2171]

bot3_metric2 = {}
bot3_metric2[1] = [0.567, 0.6, 0.567, 0.533, 0.433]
bot3_metric2[3] = [0.6, 0.567, 0.633, 0.533, 0.567]
bot3_metric2[5] = [0.633, 0.567, 0.533, 0.5, 0.467]

bot3_metric3 = {}
bot3_metric3[1] = [17, 18, 17, 16, 13]
bot3_metric3[3] = [18, 17, 19, 16, 17]
bot3_metric3[5] = [19, 17, 16, 15, 14]


bot4_metric1 = {}
bot4_metric1[1] = [1512, 1431, 1367, 2038, 2127]
bot4_metric1[3] = [1523, 1441, 1401, 1556, 1635]
bot4_metric1[5] = [1581, 1524, 1482, 1629, 1737]

bot4_metric2 = {}
bot4_metric2[1] = [0.6, 0.567, 0.567, 0.433, 0.4]
bot4_metric2[3] = [0.667, 0.633, 0.633, 0.567, 0.533]
bot4_metric2[5] = [0.633, 0.6, 0.6, 0.467, 0.433]

bot4_metric3 = {}
bot4_metric3[1] = [18, 17, 17, 13, 12]
bot4_metric3[3] = [20, 19, 19, 17, 16]
bot4_metric3[5] = [19, 18, 18, 14, 13]


bot5_metric1 = {}
bot5_metric1[1] = [1427, 1368, 1324, 1486, 1563]
bot5_metric1[3] = [1248, 1149, 1073, 1562, 1678]
bot5_metric1[5] = [1417, 1334, 1282, 1786, 1893]

bot5_metric2 = {}
bot5_metric2[1] = [0.567, 0.567, 0.533, 0.5, 0.467]
bot5_metric2[3] = [0.6, 0.567, 0.567, 0.5, 0.5]
bot5_metric2[5] = [0.6, 0.567, 0.567, 0.533, 0.5]

bot5_metric3 = {}
bot5_metric3[1] = [17, 17, 16, 15, 14]
bot5_metric3[3] = [18, 17, 17, 15, 15]
bot5_metric3[5] = [18, 17, 17, 16, 15]

plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_metric1, bot4_metric1, bot5_metric1, 'Average Rescue Moves', 1)
plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_metric2, bot4_metric2, bot5_metric2, 'Probability of Crew Rescue', 2)
plot_Bot3_vs_Bot4_vs_Bot5(alpha_values, k_values, bot3_metric3, bot4_metric3, bot5_metric3, 'Average Crew Saved', 3)


# Bot 6 vs. Bot 7 vs. Bot 8

bot6_metric1 = {}
bot6_metric1[1] = [1438, 1520, 1697, 1604, 1789]
bot6_metric1[3] = [1557, 1691, 1602, 1778, 1663]
bot6_metric1[5] = [1619, 1762, 1608, 1785, 1867]

bot6_metric2 = {}
bot6_metric2[1] = [0.533, 0.467, 0.467, 0.433, 0.4]
bot6_metric2[3] = [0.5, 0.5, 0.533, 0.467, 0.433]
bot6_metric2[5] = [0.533, 0.5, 0.467, 0.467, 0.433]

bot6_metric3 = {}
bot6_metric3[1] = [16, 14, 14, 13, 12]
bot6_metric3[3] = [15, 15, 16, 14, 13]
bot6_metric3[5] = [16, 15, 14, 14, 13]


bot7_metric1 = {}
bot7_metric1[1] = [1524, 1562, 1696, 1711, 1808]
bot7_metric1[3] = [1579, 1621, 1564, 1677, 1793]
bot7_metric1[5] = [1695, 1738, 1776, 1679, 1874]

bot7_metric2 = {}
bot7_metric2[1] = [0.5, 0.533, 0.5, 0.533, 0.467]
bot7_metric2[3] = [0.6, 0.533, 0.6, 0.567, 0.533]
bot7_metric2[5] = [0.567, 0.567, 0.567, 0.567, 0.533]

bot7_metric3 = {}
bot7_metric3[1] = [15, 16, 15, 16, 14]
bot7_metric3[3] = [18, 16, 18, 17, 16]
bot7_metric3[5] = [17, 17, 17, 17, 16]


bot8_metric1 = {}
bot8_metric1[1] = [1703, 1842, 1789, 1882, 2074]
bot8_metric1[3] = [1758, 1903, 1847, 1939, 2034]
bot8_metric1[5] = [1873, 1869, 1711, 1853, 2148]

bot8_metric2 = {}
bot8_metric2[1] = [0.5, 0.567, 0.533, 0.567, 0.467]
bot8_metric2[3] = [0.6, 0.533, 0.567, 0.567, 0.5]
bot8_metric2[5] = [0.6, 0.567, 0.567, 0.533, 0.5]

bot8_metric3 = {}
bot8_metric3[1] = [15, 17, 16, 17, 14]
bot8_metric3[3] = [18, 16, 17, 17, 15]
bot8_metric3[5] = [18, 17, 17, 16, 15]

plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_metric1, bot7_metric1, bot8_metric1, 'Average Rescue Moves', 1)
plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_metric2, bot7_metric2, bot8_metric2, 'Probability of Crew Rescue', 2)
plot_Bot6_vs_Bot7_vs_Bot8(alpha_values, k_values, bot6_metric3, bot7_metric3, bot8_metric3, 'Average Crew Saved', 3)