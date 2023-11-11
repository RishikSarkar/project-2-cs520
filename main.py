import numpy as np
import random
import math
np.set_printoptions(threshold=np.inf) # Use this to see full 50x50 numpy matrix
np.set_printoptions(linewidth=1000) # Use this to fit each row in a single line without breaks

# Function to find current open cells in grid
def find_all_open_cells(grid):
    open_cells = set()

    for i in range(50):
        for j in range(50):
            if grid[i, j] == 0:
                open_cells.add((i, j))

    return open_cells

# Function to check if a cell has one open neighbor
def single_open_neighbor_check(i, j, grid):
    count = 0

    if (i + 1 < 50) and (grid[i + 1, j] == 0):
        count += 1
    if (i - 1 >= 0) and (grid[i - 1, j] == 0):
        count += 1
    if (j + 1 < 50) and (grid[i, j + 1] == 0):
        count += 1
    if (j - 1 >= 0) and (grid[i, j - 1] == 0):
        count += 1

    return count == 1

# Function to open a random blocked cell with one open neighbor
def open_random_single_neighbor_cell(grid, num_open_neighbor_cells):
    open_neighbor_cells = set()

    # Find open neighbor cells
    for i in range(50):
        for j in range(50):
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

    if (i + 1 < 50) and (grid[i + 1, j] == 1):
        closed_neighbors.add((i + 1, j))
    if (i - 1 >= 0) and (grid[i - 1, j] == 1):
        closed_neighbors.add((i - 1, j))
    if (j + 1 < 50) and (grid[i, j + 1] == 1):
        closed_neighbors.add((i, j + 1))
    if (j - 1 >= 0) and (grid[i, j - 1] == 1):
        closed_neighbors.add((i, j - 1))

    if len(closed_neighbors) > 0:
        random_neighbor = random.choice(tuple(closed_neighbors))
        grid[random_neighbor[0], random_neighbor[1]] = 0

    return grid

# Function to create an empty 50x50 grid
def create_grid():
    grid = np.full((50, 50), 1) # Create a new 50x50 numpy matrix (2D array)
    grid[random.randrange(1, 49), random.randrange(1, 49)] = 0 # Open a random blocked cell (except on edges and corners)

    num_open_neighbor_cells = 2500

    # Iteratively open all single open neighbor cells
    while (num_open_neighbor_cells > 0):
        grid, num_open_neighbor_cells = open_random_single_neighbor_cell(grid, num_open_neighbor_cells)
    
    dead_end_cells = set()

    # Find dead-end cells
    for i in range(50):
        for j in range(50):
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

# Function to place the bot at a random open cell in the grid
def place_bot(grid, open_cells):
    bot = random.choice(tuple(open_cells)) # Pick a random cell
    grid[bot[0], bot[1]] = 2 # Place bot in grid and label space as 2
    open_cells.remove(bot)

    return bot, grid

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
    random.shuffle(alien_list)
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
                return marker, new_position
            
            grid[alien[0]][alien[1]] = 0
            grid[current[0]][current[1]] = 3
            new_position.append(current)
            break 
        
    return marker, new_position, grid

# Sensor to detect aliens within a (2k + 1) x (2k + 1) square around bot
def alien_sensor(alien_list, bot, k):
    bot_x_max = min(bot[0] + k, 49) # k cells to the right of bot
    bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
    bot_y_max = min(bot[1] + k, 49) # k cells to the top of bot
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

            neighbors = check_valid_neighbors(50, curr_cell[0], curr_cell[1])

            for neighbor in neighbors:
                if grid[neighbor[0], neighbor[1]] != 1 and neighbor not in seen_cells:
                    seen_cells.add(neighbor)
                    bfs_queue.append(neighbor)
                    d_dict[neighbor[0], neighbor[1]] = d_dict[curr_cell[0], curr_cell[1]] + 1 # Set distance of neighbor to current cell's distance + 1
        d_lookup_table[bot] = d_dict # Forgot this line I think - Aditya

    # Case in which bot has been to cell before (and knows distance to closest crew member)
    else:
        d_dict = d_lookup_table[bot]

    d_min = 2501

    # Find d for closest crew member
    for crew_member in crew_list:
        if d_dict[crew_member] < d_min:
            d_min = d_dict[crew_member]
    
    if d_min == 2501:
        return False, d_lookup_table # Don't beep if no crew member found

    prob = math.exp(-alpha * (d_min - 1))
    
    return np.random.choice([True, False], p=[prob, 1 - prob]), d_lookup_table # Beep with the specified probability

# Create alien probability matrix (dictionary) for t = 0
def initialize_alienmatrix(open_cells, bot):
    open_cells.add(bot)
    # Alien can be at any open cell except the one occupied by the bot
    inital_prob = [1/(len(open_cells) - 1)] * len(open_cells)
    alien_matrix = dict(zip(open_cells, inital_prob))
    bot_cell = {bot : 0}
    alien_matrix.update(bot_cell)
    open_cells.remove(bot)

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

#Update probabilties for alien matrix based on detection 
def update_alienmatrix(alien_matrix, detected, bot, k):
    if detected:
        # Cells outside detection square should have probability  0
        out_detection_cells = [key for key in alien_matrix if not (((key[0] >= (bot[0]-k)) and (key[0] <= (bot[0]+k))) and ((key[1] >= (bot[1]-k)) and (key[1] <= (bot[1]+k))))] 
        for cell in out_detection_cells:
            new_prob = { cell: 0 }
            alien_matrix.update(new_prob)
        in_square_cells = alien_matrix.keys() - out_detection_cells
        sum = 0
        for cell in in_square_cells:
            sum = sum + alien_matrix[cell]
        for cell in in_square_cells:
            alien_matrix.update({cell : alien_matrix[cell] * (1/sum)})
    else:
        # Cells inside detection square show have probability 0
        detection_cells = [key for key in alien_matrix if (((key[0] >= (bot[0]-k)) and (key[0] <= (bot[0]+k))) and ((key[1] >= (bot[1]-k)) and (key[1] <= (bot[1]+k))))]
        for cell in detection_cells:
            new_prob = { cell: 0}
            alien_matrix.update(new_prob)
        outside_cells = alien_matrix.keys() - detection_cells
        sum = 0
        for cell in outside_cells:
            sum = sum + alien_matrix[cell]
        for cell in outside_cells:
            alien_matrix.update({cell : alien_matrix[cell] * (1/sum)})

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


# def move_bot(grid, bot, alien_matrix, crew_matrix):
#     neighbors = check_valid_neighbors(len(grid), bot[0], bot[1])
#     open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)] # Changed (grid[neigh] == 0) since bot can move to any unblocked cell
#     zero_alienprob = [move for move in open_moves if alien_matrix[move] == 0]
#     determined_move = None
#     if zero_alienprob:
#         max_crewprob = -1
#         for cell in zero_alienprob:
#             if crew_matrix[cell] > max_crewprob:
#                 max_crewprob = crew_matrix[cell]
#         determined_move  = random.choice(tuple([c for c in zero_alienprob if crew_matrix[c] == max_crewprob]))
#     else:
#         max_crewprob = -1
#         for cell in open_moves:
#             if crew_matrix[cell] > max_crewprob:
#                 max_crewprob = crew_matrix[cell]
#         determined_move = random.choice(tuple([c for c in open_moves if crew_matrix[c] == max_crewprob]))

#     print(determined_move)
#     return(determined_move)


# Function to move bot to specified cell (Makes sense to do probability updates and move decision in functions for each Bot, since other factors to consider)
def move_bot(grid, bot, new_cell, crew_list, open_cells, win_count):
    # Add new bot location to open cells set and remove old one. Modify grid accordingly
    open_cells.add(bot)
    grid[bot[0], bot[1]] = 0
    grid[new_cell[0], new_cell[1]] = 2
    open_cells.remove(new_cell)
    bot = new_cell

    # Case where bot lands on the same cell as a crew member
    for crew_member in crew_list:
        if bot == crew_member:
            win_count += 1 # Increment win count because crew member has been saved
            crew_list.remove(crew_member)
            crew_list, grid = place_crew(grid, open_cells, crew_list) #TODO: Might need to modify depending on the order of saving crew (i.e., if all current crew members need to be saved before new ones are added, etc.)
            return bot, crew_list, grid, open_cells, win_count
    
    return bot, crew_list, grid, open_cells, win_count

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

def determine_move(moves, alien_matrix, crew_matrix):
    
    zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
    max_crewprob = -1
    for cell in moves:
        if crew_matrix[cell] > max_crewprob:
            max_crewprob = crew_matrix[cell]
   
    chosen_move = None
    if not zero_alienprob:
        chosen_move = random.choice(moves)
    else:
        chosen_move = random.choice(zero_alienprob)
    
    return chosen_move
    

def update_afteralienmove(ship, alien_list, alien_matrix):
    total_summation = 0
    neighbors = check_valid_neighbors(len(ship), alien_list[0][0], alien_list[0][1])
    neighbors = [neigh for neigh in neighbors if ship[neigh] != 1]
    for neigh in neighbors:
        if neigh in alien_matrix:
            total_summation = total_summation + (alien_matrix[neigh] * (1/(len(neighbors)+1))) # Plus 1 because alien could stay in place 

    
    alien_matrix[alien_list[0]] = total_summation
    return alien_matrix
    

# 1 crew 1 alien bot 
# Notes: (Setup done inside this method, since we are only testing by varying k and alpha values. 
# Might reuse grid if it takes too long to regenerate for each test)
def Bot1(k, alpha):
    grid, open_cells = create_grid()
    bot, ship = place_bot(grid, open_cells)

    crew_list = []
    alien_list = []
    d_lookup_table = {}

    # Place 1 crew member + 1 alien
    crew_list, ship = place_crew(ship, open_cells, crew_list)
    alien_list, ship = place_alien(ship, open_cells, alien_list, bot, k)

    alien_matrix = initialize_alienmatrix(open_cells, bot)
    crew_matrix = initialize_crewmatrix(open_cells, crew_list, bot)
    win_count = 0
    marker = 0
    while win_count < 10: # Arbitrary temporary win condition for testing
        neighbors = check_valid_neighbors(len(ship), bot[0], bot[1])
        open_moves = [neigh for neigh in neighbors if (grid[neigh] != 1)]
        #TODO: Logic to calculate best neighbor to move to (based on min alien probability and max crew probability) 
        open_moves.append(bot)
        next_move = determine_move(open_moves, alien_matrix, crew_matrix)

        
        # next_move = bot # Temporarily set to bot. Must calculate using previous alien_matrix and crew_matrix values
        
        bot, crew_list, ship, open_cells, win_count = move_bot(ship, bot, next_move, crew_list, open_cells, win_count)
       
        alien_matrix, crew_matrix = update_afterbotmove(bot, alien_matrix, crew_matrix)
        # Move bot to optimal neighbor
        marker, alien_list, ship = move_aliens(ship, alien_list, bot) # Move alien randomly
        alien_matrix = update_afteralienmove(ship, alien_list, alien_matrix) # Update after alien move
        # If bot is captured, end simulation
        if marker == 1:
            return False
        
        alien_detected = alien_sensor(alien_list, bot, k) # Run Alien Sensor
        crew_detected, d_lookup_table = crew_sensor(ship, bot, alpha, d_lookup_table, crew_list) # Run Crew Sensor
        print(alien_detected)
        alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update based on alien censor 

        crew_matrix = update_crewmatrix(crew_matrix, crew_detected, d_lookup_table, bot, alpha) # Update based on crew censor 
        #print(alien_matrix)
        # sum = 0
        # for x in alien_matrix:
        #     sum = sum + alien_matrix[x]
        # print("Alien: " + str(sum))
        # sum1 = 0
        # for x in crew_matrix:
        #     sum1 = sum1 + crew_matrix[x]
        # print("Crew: " + str(sum1))
        # print("Win count" + str(win_count))
        #print(alien_matrix)
        # print(alien_list)
        # print(bot)
        
        #print("-----------")
    return True


# def Bot1(k, alpha):
    # while True:
    #     move = move_bot(grid, bot, alien_matrix, crew_matrix)
    #     if move in crew_list:
    #         print("Crew rescused!")
    #         break
    #     else:
    #         bot = move
    #     update_afterbotmove(bot, alien_matrix, crew_matrix)
    #     break
        
    #     # alien_detected = alien_sensor(alien_list, bot, alpha) #Alien detector ran
    #     # alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update beliefs 
    #     # marker, alien_list = move_aliens(grid, alien_list, bot) # Move aliens
    #     # if marker:
    #     #     print("Bot captured by alien!")
    #     #     break
    #     # alien_detected = alien_sensor(alien_list, bot, alpha) 
    #     # alien_matrix = update_alienmatrix(alien_matrix, alien_detected, bot, k) # Update beliefs 
    #     # crew_detected = crew_sensor(grid, bot, crew_list, alien_list, 2) #
    # return True


# Testing Area

# ship, open_cells = create_grid()
# bot, ship = place_bot(ship, open_cells)

# crew_list = []
# alien_list = []

# d_lookup_table = {}

# crew_list, ship = place_crew(ship, open_cells, crew_list)
# crew_list, ship = place_crew(ship, open_cells, crew_list)

# alien_list, ship = place_alien(ship, open_cells, alien_list, bot, 1)

# print(f"Ship: {ship}\nBot: {bot}\nCrew: {crew_list}\nAliens: {alien_list}\n")
# print(f"Alien Sensor: {alien_sensor(alien_list, bot, 5)}\nCrew Sensor: {crew_sensor(ship, bot, 0.1, d_lookup_table, crew_list)}\n")

# marker, alien_list, ship = move_aliens(ship, alien_list, bot)
# print(f"Ship: {ship}\nBot: {bot}\nCrew: {crew_list}\nAliens: {alien_list}\nMarker: {marker}\n")

Bot1(3, 0.3)

