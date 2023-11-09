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