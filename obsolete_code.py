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