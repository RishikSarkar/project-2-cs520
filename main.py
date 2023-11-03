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

    return bot

# Function to place the crew at a random open cell in the grid (must occur after placing bot)
def place_crew(grid, open_cells, crew_list):
    crew = random.choice(tuple(open_cells - set(crew_list))) # Pick a random cell without an existing crew member
    grid[crew[0], crew[1]] = 4 # Place crew member in grid and label space as 4
    crew_list.append(crew)

    return crew_list

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
    return alien_list

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
                
        
    return marker, new_position

# Sensor to detect aliens within a (2k + 1) x (2k + 1) square around bot
def alien_sensor(alien_list, bot, k):
    bot_x_max = min(bot[0] + k, 49) # k cells to the right of bot
    bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
    bot_y_max = min(bot[1] + k, 49) # k cells to the top of bot
    bot_y_min = max(0, bot[1] - k) # k cells to the bottom of bot

    # Check if each alien is within the detection square
    for alien in alien_list:
        if (alien[0] > bot_x_min and alien[0] < bot_x_max) and (alien[1] > bot_y_min and alien[1] < bot_y_max):
            return True
    
    return False

# Generate cost map, i.e., distance of each cell on grid from bot
# This might help update the bot's knowledge of alien and crew positions after every time step
def find_cost_map(grid, bot):
    cost_map = np.full((50, 50), 100)
    seen_cells = set()
    bfs_queue = []
    bfs_queue.append(bot)
    seen_cells.add(bot)
    cost_map[bot[0], bot[1]] = 0

    # Use BFS to find shortest path cost from bot to every unblocked cell (including aliens + crew)
    while len(bfs_queue) > 0:
        curr_cell = bfs_queue.pop(0)
        neighbors = check_valid_neighbors(50, curr_cell[0], curr_cell[1])

        for neighbor in neighbors:
            if grid[neighbor[0], neighbor[1]] != 1 and neighbor not in seen_cells:
                seen_cells.add(neighbor)
                bfs_queue.append(neighbor)
                cost_map[neighbor[0], neighbor[1]] = cost_map[curr_cell[0], curr_cell[1]] + 1 # Set distance of neighbor to current cell's distance + 1

    return cost_map

# Sensor to detect distance d to closest crew member and beep with probability exp(-alpha * (d - 1))
def crew_sensor(crew_list, cost_map, grid, bot, alpha):
    cost_map = find_cost_map(grid, bot) # This can be where cost map update occurs (might change later)
    min_d = 100 # Actual distance of closest crew member
    
    # Find distance to closest crew member
    for crew in crew_list:
        if cost_map[crew[0], crew[1]] <= min_d:
            min_d = cost_map[crew[0], crew[1]]
    
    prob = math.exp(-alpha * (min_d - 1)) 
    return np.random.choice([True, False], p=[prob, 1 - prob]) # Beep with the specified probability



# Testing Area

ship, open_cells = create_grid()
# print(ship, open_cells, "\n")

bot = place_bot(ship, open_cells)
# print(ship, bot, open_cells.__contains__(bot), "\n")

crew_list = []
alien_list = []

crew_list = place_crew(ship, open_cells, crew_list)
crew_list = place_crew(ship, open_cells, crew_list)
crew_list.append(bot)
# print(ship, crew_list, set(crew_list).issubset(open_cells), "\n")

alien_list = place_alien(ship, open_cells, alien_list, bot, 1)
# print(alien_list)
# marker, alien_list = move_aliens(ship, alien_list, bot)
# print(ship)
# print(alien_list)

# print(alien_sensor(alien_list, bot, 5))
# print(f"Aliens: {alien_list} \n Bot: {bot} \n Ship: {ship}")

cost_map = find_cost_map(ship, bot)

print(crew_sensor(crew_list, cost_map, ship, bot, 0.5, 10))
print(f"Crew Members: {crew_list} \n Bot: {bot} \n Ship: {ship} \n Cost Map: {cost_map}")