# Obsolete Code (For dev, testing, etc.):

# print(ship, open_cells, "\n")
# print(ship, bot, open_cells.__contains__(bot), "\n")
# print(ship, crew_list, set(crew_list).issubset(open_cells), "\n")

# print(alien_list)
# marker, alien_list = move_aliens(ship, alien_list, bot)
# print(ship)
# print(alien_list)

# print(alien_sensor(alien_list, bot, 5))
# print(f"Aliens: {alien_list} \n Bot: {bot} \n Ship: {ship}")

# cost_map = find_cost_map(ship, bot)

# print(crew_sensor(crew_list, cost_map, ship, bot, 0.5, 10))
# print(f"Crew Members: {crew_list} \n Bot: {bot} \n Ship: {ship} \n Cost Map: {cost_map}")

# # Generate cost map, i.e., distance of each cell on grid from bot
# # This might help update the bot's knowledge of alien and crew positions after every time step
# def find_cost_map(grid, bot):
#     cost_map = np.full((50, 50), 100)
#     seen_cells = set()
#     bfs_queue = []
#     bfs_queue.append(bot)
#     seen_cells.add(bot)
#     cost_map[bot[0], bot[1]] = 0

#     # Use BFS to find shortest path cost from bot to every unblocked cell (including aliens + crew)
#     while len(bfs_queue) > 0:
#         curr_cell = bfs_queue.pop(0)
#         neighbors = check_valid_neighbors(50, curr_cell[0], curr_cell[1])

#         for neighbor in neighbors:
#             if grid[neighbor[0], neighbor[1]] != 1 and neighbor not in seen_cells:
#                 seen_cells.add(neighbor)
#                 bfs_queue.append(neighbor)
#                 cost_map[neighbor[0], neighbor[1]] = cost_map[curr_cell[0], curr_cell[1]] + 1 # Set distance of neighbor to current cell's distance + 1

#     return cost_map

# # Sensor to detect distance d to closest crew member and beep with probability exp(-alpha * (d - 1))
# def crew_sensor(crew_list, cost_map, grid, bot, alpha):
#     cost_map = find_cost_map(grid, bot) # This can be where cost map update occurs (might change later)
#     min_d = 100 # Actual distance of closest crew member
    
#     # Find distance to closest crew member
#     for crew in crew_list:
#         if cost_map[crew[0], crew[1]] <= min_d:
#             min_d = cost_map[crew[0], crew[1]]
    
#     prob = math.exp(-alpha * (min_d - 1)) 
#     return np.random.choice([True, False], p=[prob, 1 - prob]) # Beep with the specified probability

# win_count = 0
# bot, crew_list, ship, open_cells, win_count = move_bot(ship, bot, crew_list[0], crew_list, open_cells, win_count)
# print(f"Ship: {ship}\nBot: {bot}\nCrew: {crew_list}\nOpen Cells: {bot in open_cells}\nWin Count: {win_count}")

# _, d_lookup_table = crew_sensor(ship, (bot[0], bot[1]), 0.1, d_lookup_table, crew_list)
# print(d_lookup_table)

# _, d_lookup_table = crew_sensor(ship, (bot[0], bot[1] + 1), 0.1, d_lookup_table, crew_list)
# print(d_lookup_table)

# _, d_lookup_table = crew_sensor(ship, (bot[0], bot[1]), 0.1, d_lookup_table, crew_list)
# print(d_lookup_table)

# # Determine shortest path (Constructs 2D array that keep track of the coordinates of the cells that can be reach a certain cell, Stop when crew coordinates reached)    
# def BFS(bot, crew, board, aliens):
#     q = collections.deque([bot])
#     visited = set(bot)
#     construct = np.full((50,50), None)
#     while q:
#         current = q.popleft()
#         possible_neigh = check_valid_neighbors(50, current[0], current[1])
#         if not possible_neigh or board[current[0]][current[1]] == 1:
#             continue
#         if current == crew:
#             break
#         for neigh in possible_neigh:
#             if neigh not in visited:
#                 q.append(neigh)
#                 visited.add(neigh)
#                 construct[neigh[0]][neigh[1]] = current
    
    
#     return construct


# # Sensor to detect distance d to closest crew member and beep with probability exp(-alpha * (d - 1))
# def crew_sensor(grid, bot, alpha):
#     d_map = np.full((50, 50), 100)
#     seen_cells = set()
#     bfs_queue = []
#     bfs_queue.append(bot)
#     seen_cells.add(bot)
#     d_map[bot[0], bot[1]] = 0

#     # Use BFS to find shortest path cost from bot to closest crew member
#     while len(bfs_queue) > 0:
#         curr_cell = bfs_queue.pop(0)

#         neighbors = check_valid_neighbors(50, curr_cell[0], curr_cell[1])

#         for neighbor in neighbors:
#             if grid[neighbor[0], neighbor[1]] != 1 and neighbor not in seen_cells:
#                 if grid[neighbor[0], neighbor[1]] == 4: # Case where closest crew member is found
#                     d = d_map[curr_cell[0], curr_cell[1]] + 1 # Distance to crew member = distance to crew neighbor + 1
#                     print("D: ", d)
#                     prob = math.exp(-alpha * (d - 1))
#                     return np.random.choice([True, False], p=[prob, 1 - prob]) # Beep with the specified probability

#                 seen_cells.add(neighbor)
#                 bfs_queue.append(neighbor)
#                 d_map[neighbor[0], neighbor[1]] = d_map[curr_cell[0], curr_cell[1]] + 1 # Set distance of neighbor to current cell's distance + 1

#     return False # Crew member not found, so no beep


# # Outline the length of the path determined by BFS
# def getPath(construct, start, crew):
#     if construct[crew[0]][crew[1]] == None:
#         return None
#     path = []
#     current = crew
#     while current != start:
#         path.append(current)
#         current = construct[current[0]][current[1]]
#     path.reverse()
#     return len(path)

# # Sensor to detect crew members within d-steps and beep with probability exp(-alpha * (d - 1))
# def crew_sensor(grid, bot, crew_list, aliens, alpha):
#     # cost_map = find_cost_map(grid, bot) # This can be where cost map update occurs (might change later)
    
#     # For each crew member, check if cost (i.e., distance from bot) is <= d
#     for crew in crew_list:
#         print(bot)
#         construct = BFS(bot, crew, grid, aliens)
#         d = getPath(construct, bot, crew)
#         print(d)
#         prob = math.exp(-alpha * (d - 1))
#         if np.random.choice([True, False], p=[prob, 1 - prob]): # Beep with the specified probability
#             return True 
        
#     return False

# # Based on suggestion from Professor's latest announcement (Havent actually used in the Bot code)
# def dijsktra(grid, open_cells):
#     distances = {}
#     run_distance = {}
#     for cur_cell in open_cells:
#         for cur_cell2 in open_cells:
#             run_distance[cur_cell] = 5000
    
#     for cell in open_cells:
#         run_distance_cpy = copy.deepcopy(run_distance)
#         run_distance_cpy[cell] = 0
#         tovisit_cells = copy.copy(open_cells)
#         while tovisit_cells:
#             cur_cell = None
#             for c in tovisit_cells:
#                 if cur_cell == None:
#                     cur_cell = c
#                 elif run_distance_cpy[cur_cell] > run_distance_cpy[c]:
#                     cur_cell = c
#             neighbors = check_valid_neighbors(50, cur_cell[0], cur_cell[1])
#             for neigh in neighbors:
#                 if grid[neigh] == 0:
#                     length = run_distance_cpy[cur_cell] + 1
#                     if run_distance_cpy[neigh] > length:
#                         run_distance_cpy[neigh] = length
#             tovisit_cells.remove(cur_cell)
#         for dis in run_distance_cpy:
#             distances[(cell, dis)] = run_distance_cpy[dis]
#         print(distances)
#     return distances

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

# # Determine the best neighboring cell for the bot to move to based on probability matrices
# def determine_move(moves, alien_matrix, crew_matrix):
#     zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
#     max_crewprob = -1
#     for cell in moves:
#         if crew_matrix[cell] > max_crewprob:
#             max_crewprob = crew_matrix[cell]
   
#     chosen_move = None
#     if not zero_alienprob:
#         chosen_move = random.choice(moves)
#     else:
#         chosen_move = random.choice(zero_alienprob)
    
#     return chosen_move

# # Determine the best neighboring cell for the bot to move to based on probability matrices
# def determine_move_2crew(moves, alien_matrix, crew_matrix, index_mapping):

#     # Case at least one move with 0 alien probability
#     if zero_alienprob:
#         print("0")
#         max_crewprob = -1
#         for cell in zero_alienprob:
#             cell_index = index_mapping[cell] # Retrieve index mapping of current cell
#             current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
#             current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

#             # Find max probability cell
#             if current_crewprob > max_crewprob:
#                 max_crewprob = current_crewprob
#                 chosen_cell = cell
#     # Case where at least one move with nonzero crew probability
#     elif nonzero_crewprob:
#         print("1")
#         max_crewprob = -1
#         for cell in nonzero_crewprob:
#             cell_index = index_mapping[cell] # Retrieve index mapping of current cell
#             current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
#             current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

#             # Find max probability cell
#             if current_crewprob > max_crewprob:
#                 max_crewprob = current_crewprob
#                 chosen_cell = cell
            
#             # Add statement for case where there are multiple choices with the same prob, so one is picked at random
#     else:
#         print("2")
#         max_crewprob = -1
#         for cell in moves:
#             cell_index = index_mapping[cell] # Retrieve index mapping of current cell
#             current_crewprob = np.sum(crew_matrix[cell_index, :]) + np.sum(crew_matrix[:, cell_index]) # Sum probability of crew in current cell with all other cells
#             current_crewprob -= crew_matrix[cell_index, cell_index] # Avoid double-counting

#             # Find max probability cell
#             if current_crewprob > max_crewprob:
#                 max_crewprob = current_crewprob
#                 chosen_cell = cell

#     return chosen_cell

#     # zero_alienprob = [move for move in moves if alien_matrix[move] == 0]
#     # chosen_cell = None
#     # if not zero_alienprob:
#     #     max_crewprob = -1
        
#     #     for cell in moves:
#     #         current_crewprob = 0
#     #         for i in range(0, crew_matrix.shape[2]):
#     #             for j in range(0, crew_matrix.shape[3]):
#     #                 current_crewprob = current_crewprob + crew_matrix[cell[0]][cell[1]][i][j]
            
#     #         if current_crewprob > max_crewprob:
#     #             max_crewprob = current_crewprob
#     #             chosen_cell = cell 
            
#     # else:  
#     #     max_crewprob = -1
        
#     #     for cell in zero_alienprob:
#     #         current_crewprob = 0
#     #         for i in range(0, crew_matrix.shape[2]):
#     #             for j in range(0, crew_matrix.shape[3]):
#     #                 current_crewprob = current_crewprob + crew_matrix[cell[0]][cell[1]][i][j]
#     #         if current_crewprob > max_crewprob:
#     #             max_crewprob = current_crewprob
#     #             chosen_cell = cell    
   
#     # chosen_move = None
#     # if not zero_alienprob:
#     #     chosen_move = random.choice(moves)
#     # else:
#     #     chosen_move = random.choice(zero_alienprob)
    
#     # return chosen_cell

# def update_afterbotmove_2crew(bot, alien_matrix, crew_matrix, index_mapping):
    
#     total_sum = 0
#     for i in range(0, crew_matrix.shape[0]):
#         for j in range(0, crew_matrix.shape[1]):
#             for k in range(0, crew_matrix.shape[2]):
#                 for m in range(0, crew_matrix.shape[3]):
#                     if ((i, j) == bot) or ((k,m) == bot):
#                         crew_matrix[i][j][k][m] = 0
#                     total_sum = total_sum + crew_matrix[i][j][k][m]

#     crew_matrix = crew_matrix / total_sum

#     return alien_matrix, crew_matrix

# # Update probabilties for crew matrix based on beep
# def update_crewmatrix_2crew(crew_matrix, detected, d_lookup_table, bot, alpha, grid, index_mapping):

#     # Previous Code:
#     # Case where beep is detected from bot cell
#     if detected:
#         d_dict = d_lookup_table.get(bot) # Get the d dictionary calculated with the crew sensor
#         total_summation = 0
#         for i in range(0, crew_matrix.shape[0]):
#             for j in range(0, crew_matrix.shape[1]):
#                 for k in range(0, crew_matrix.shape[2]):
#                     for m in range(0, crew_matrix.shape[3]):
#                         if grid[i][j] == 0 and grid[k][m] == 0:
#                             d1 = d_dict.get((i,j)) # Find d from bot to cell
#                             d2 = d_dict.get((k,m)) # Find d from bot to cell
#                             if ((i,j) == bot) or ((k, m) == bot):
#                                 crew_matrix[i][j][k][m] = 0 # Crew member not at current cell
#                             else:
#                                 crew_matrix[i][j][k][m] *= (1 - ((1-(math.exp(-alpha * (d1 - 1)))) * (1-(math.exp(-alpha * (d2 - 1)))))) # Multiply probability of cell containing crew by given prob
#                             total_summation += crew_matrix[i][j][k][m] # Calculate sum of all probabilities
        
#         crew_matrix = crew_matrix / total_summation # Normalize probabilities
#     # Case where beep is not detected from bot cell
#     else:
#         d_dict = d_lookup_table.get(bot) # Get the d dictionary calculated with the crew sensor
#         total_summation = 0
#         for i in range(0, crew_matrix.shape[0]):
#             for j in range(0, crew_matrix.shape[1]):
#                 for k in range(0, crew_matrix.shape[2]):
#                     for m in range(0, crew_matrix.shape[3]):
#                         if grid[i][j] == 0 and grid[k][m] == 0:
#                             d1 = d_dict.get((i,j)) # Find d from bot to cell
#                             d2 = d_dict.get((k,m)) # Find d from bot to cell
#                             if ((i,j) == bot) or ((k, m) == bot):
#                                 crew_matrix[i][j][k][m] = 0 # Crew member not at current cell
#                             else:
#                                 crew_matrix[i][j][k][m] *= ((1-(math.exp(-alpha * (d1 - 1)))) * (1-(math.exp(-alpha * (d2 - 1))))) # Multiply probability of cell containing crew by given prob
#                             total_summation += crew_matrix[i][j][k][m] # Calculate sum of all probabilities
        
#         crew_matrix = crew_matrix / total_summation # Normalize probabilities

#     return crew_matrix

# # Update probabilties for alien matrix based on detection 
# def update_alienmatrix(alien_matrix, detected, bot, k):

#     if detected:
#         # Cells outside detection square should have probability 0
#         out_detection_cells = [key for key in alien_matrix if not (((key[0] >= (bot[0]-k)) and (key[0] <= (bot[0]+k))) and ((key[1] >= (bot[1]-k)) and (key[1] <= (bot[1]+k))))] 
#         for cell in out_detection_cells:
#             new_prob = { cell: 0 }
#             alien_matrix.update(new_prob)
#         in_square_cells = alien_matrix.keys() - out_detection_cells
#         sum = 0
#         for cell in in_square_cells:
#             sum = sum + alien_matrix[cell]
#         for cell in in_square_cells:
#             alien_matrix.update({cell : alien_matrix[cell] * (1/sum)})
#     else:
#         # Cells inside detection square show have probability 0
#         detection_cells = [key for key in alien_matrix if (((key[0] >= (bot[0]-k)) and (key[0] <= (bot[0]+k))) and ((key[1] >= (bot[1]-k)) and (key[1] <= (bot[1]+k))))]
#         for cell in detection_cells:
#             new_prob = { cell: 0}
#             alien_matrix.update(new_prob)
#         outside_cells = alien_matrix.keys() - detection_cells
#         sum = 0
#         for cell in outside_cells:
#             sum = sum + alien_matrix[cell]
#         for cell in outside_cells:
#             alien_matrix.update({cell : alien_matrix[cell] * (1/sum)})

#     return alien_matrix

# #  Create alien probability matrix (dictionary) for t = 0
# def initialize_alienmatrix(open_cells, bot, k):

#     open_cells.add(bot)

#     bot_x_max = min(bot[0] + k, 49) # k cells to the right of bot
#     bot_x_min = max(0, bot[0] - k) # k cells to the left of bot
#     bot_y_max = min(bot[1] + k, 49) # k cells to the top of bot
#     bot_y_min = max(0, bot[1] - k) # k cells to the bottom of bot

#     empty_count = 0
#     for cell in open_cells:
#         if (cell[0] >= bot_x_min and cell[0] <= bot_x_max) and (cell[1] >= bot_y_min and cell[1] <= bot_y_max):
#             empty_count += 1

#     # Alien can be at any open cell except the ones occupied by the bot 
#     initial_prob = [1/(len(open_cells) - empty_count)] * len(open_cells)
#     alien_matrix = dict(zip(open_cells, initial_prob))
#     # bot_cell = {bot : 0}

#     for cell in open_cells:
#         if (cell[0] >= bot_x_min and cell[0] <= bot_x_max) and (cell[1] >= bot_y_min and cell[1] <= bot_y_max):
#             alien_matrix[cell] = 0

#     # alien_matrix.update(bot_cell)

#     open_cells.remove(bot)

#     return alien_matrix