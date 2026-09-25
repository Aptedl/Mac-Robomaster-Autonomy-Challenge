"""
Agent template for the navigation challenge.
"""

import heapq
import math


class Agent:

  def __init__(self, cfg: dict):
    """cfg keys: width_m, height_m, resolution, robot_radius, v_max, a_max, dt, sense_cells, goal_tol, goal (x, y)."""
    self.cfg = cfg
    self.WIDTH, self.HEIGHT = 80, 120
    self.grid = ["?" for i in range(self.WIDTH * self.HEIGHT)]
    self.last_path = []
    self.prev_vx = 0.0
    self.prev_vy = 0.0
    self.last_start_cell = None

    # Precompute kernel sizes for hierarchical buffering
    self.kernels = {}
    for r in range(1, 6):
      k = []
      for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
          if math.hypot(dx, dy) <= r:
            k.append((dx, dy))
      self.kernels[r] = k

  def read_cell(self, x, y):
    if not (0 <= x < 120 and 0 <= y < 80):
      return "#"
    return self.grid[y * 120 + x]

  def update_map(self, scan):
    cx0, cy0, rows = scan
    cy = cy0
    for row in rows:
      cx = cx0
      for ch in row:
        if 0 <= cx < 120 and 0 <= cy < 80:
          self.grid[cy * 120 + cx] = ch
        cx += 1
      cy += 1

  def is_offlimits(self, x, y, buffer):
    if not (0 <= x < 120 and 0 <= y < 80):
      return True
    for dx, dy in self.kernels.get(buffer, self.kernels[4]):
      nx, ny = x + dx, y + dy
      if 0 <= nx < 120 and 0 <= ny < 80:
        if self.read_cell(nx, ny) == "#":
          return True
      else:
        return True
    return False

  def h(self, cell, goal):
    return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

  def reconstruct_path(self, came_from, current):
    path = [current]
    while current in came_from:
      current = came_from[current]
      path.append(current)
    path.reverse()
    return path

  def get_neighbors(self, cell, buffer):
    x, y = cell
    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    result = []
    for dx, dy in moves:
      nx, ny = x + dx, y + dy
      if not (0 <= nx < 120 and 0 <= ny < 80):
        continue
      if self.is_offlimits(nx, ny, buffer):
        continue
      if dx != 0 and dy != 0:
        if self.is_offlimits(x + dx, y, buffer) or self.is_offlimits(x, y + dy, buffer):
          continue
      result.append((nx, ny))
    return result

  def step_cost(self, a, b):
    cost = math.hypot(b[0] - a[0], b[1] - a[1])
    if self.read_cell(b[0], b[1]) == "?":
      cost *= 1.1
    return cost

  def a_star(self, start, goal, buffer):
    open_list = [(self.h(start, goal), start)]
    g_score = {start: 0}
    came_from = {}

    while open_list:
      _, current = heapq.heappop(open_list)
      if current == goal:
        return self.reconstruct_path(came_from, current)

      for neighbor in self.get_neighbors(current, buffer):
        tentative_g = g_score[current] + self.step_cost(current, neighbor)
        if neighbor not in g_score or tentative_g < g_score[neighbor]:
          g_score[neighbor] = tentative_g
          came_from[neighbor] = current
          f = tentative_g + self.h(neighbor, goal)
          heapq.heappush(open_list, (f, neighbor))
    return None

  def nearest_free_cell(self, cell, buffer):
    if not self.is_offlimits(cell[0], cell[1], buffer):
      return cell

    visited = {cell}
    frontier = [cell]
    while frontier:
      next_frontier = []
      for c in frontier:
        x, y = c
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
          nx, ny = x + dx, y + dy
          if not (0 <= nx < 120 and 0 <= ny < 80):
            continue
          if (nx, ny) in visited:
            continue
          visited.add((nx, ny))
          if not self.is_offlimits(nx, ny, buffer):
            return (nx, ny)
          next_frontier.append((nx, ny))
      frontier = next_frontier
    return cell

  def step(
      self, pose: tuple[float, float], scan: tuple[int, int, list[str]]
  ) -> tuple[float, float]:
    goal_x, goal_y = self.cfg["goal"]

    self.update_map(scan)

    start_cell = (
        math.floor(pose[0] / self.cfg["resolution"]),
        math.floor(pose[1] / self.cfg["resolution"]),
    )
    goal_cell = (
        math.floor(goal_x / self.cfg["resolution"]),
        math.floor(goal_y / self.cfg["resolution"]),
    )

    # Recompute path only when entering a new cell, trying buffer 4 first for wide berth
    if not self.last_path or start_cell != self.last_start_cell:
      path = None
      for b in [4, 3, 2, 1]:
        s_cell = self.nearest_free_cell(start_cell, b)
        g_cell = self.nearest_free_cell(goal_cell, b)
        p = self.a_star(s_cell, g_cell, b)
        if p:
          path = p
          break
      if path:
        self.last_path = path
      self.last_start_cell = start_cell

    target_vx, target_vy = 0.0, 0.0
    if self.last_path and len(self.last_path) > 1:
      # Look further ahead (index 3 or end of path) to force a wider swing around turns
      target_idx = min(3, len(self.last_path) - 1)
      target_cell_x, target_cell_y = self.last_path[target_idx]

      target_x = (
          target_cell_x * self.cfg["resolution"] + self.cfg["resolution"] / 2
      )
      target_y = (
          target_cell_y * self.cfg["resolution"] + self.cfg["resolution"] / 2
      )
      distance_x, distance_y = target_x - pose[0], target_y - pose[1]
      dist = math.hypot(distance_x, distance_y)

      speed = self.cfg["v_max"]
      if dist > 0:
        target_vx = distance_x / dist * speed
        target_vy = distance_y / dist * speed

    # Acceleration smoothing
    max_dv = self.cfg["a_max"] * self.cfg["dt"]
    dvx = target_vx - self.prev_vx
    dvy = target_vy - self.prev_vy
    dv_mag = math.hypot(dvx, dvy)

    if dv_mag > max_dv:
      vx = self.prev_vx + (dvx / dv_mag) * max_dv
      vy = self.prev_vy + (dvy / dv_mag) * max_dv
    else:
      vx, vy = target_vx, target_vy

    self.prev_vx = vx
    self.prev_vy = vy

    return (vx, vy)

  def debug(self) -> dict:
    res = self.cfg["resolution"]
    path_m = [(cx * res + res / 2, cy * res + res / 2) for cx, cy in self.last_path]
    blocked = []
    for y in range(80):
      for x in range(120):
        if self.is_offlimits(x, y, 4):
          blocked.append((x, y))
    return {"path": path_m, "blocked": blocked}


"""
This is the code I could write without AI. It's worse,
but the implementation is simpler and I actually understand what's going on.
"""

# """
# agent template for the nav challenge
# """

# #I know you said no imports but these are both in the Python standard library (no pip installs) so I hope it's ok??
# import math
# import heapq

# class Agent:
#   def __init__(self, cfg:dict):
#     """cfg keys: width_m, height_m, resolution, robot_radius, v_max, a_max, dt, sense_cells, goal_tol, goal (x, y)."""
#     self.cfg = cfg
#     self.WIDTH, self.HEIGHT = 80, 120
#     self.grid = ["?" for i in range(self.WIDTH * self.HEIGHT)]#? represents unknown cells
#     self.offlimits = self.grid.copy() #This is a second, more conservative grid that includes walls
#                                       #around where the real walls are so the robot avoids those

#   def read_cell(self, x, y):
#     if not (0 <= x < 120 and 0 <= y < 80):
#       return "#"                       # outside the room counts as wall
#     return self.grid[y * 120 + x]

#   def write_cell(self, x, y, value, buffer=5):
#     if 0 <= x < 120 and 0 <= y < 80:   # skip spots outside the room
#       wall_location = y * 120 + x
#       self.grid[wall_location] = value

#       if value == "#": # A hashtag is a wall. We don't want to replace anything needlessly with this buffer loop
#         #We want a 5-cell margin for the robot to clear, since it's 0.3m wide, so 3 cells is the bare minimum. Extra 2 for safety.
#         for dy in range(-buffer, buffer + 1):#-buffer to buffer + 1 since we want it to create a full square around the wall buffer times.
#           for dx in range(-buffer, buffer + 1):
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < 120 and 0 <= ny < 80:#Checks if it's outside the map or going to write to do a different row
#               self.offlimits[ny * 120 + nx] = value

#   def update_map(self, scan):
#     cx0, cy0, rows = scan
#     cy = cy0
#     for row in rows: # each line of the picture
#       cx = cx0 # cell x of the first character on this line
#       for ch in row: # each character on the line
#         if self.read_cell(cx, cy) != ch:
#           self.write_cell(cx, cy, ch)
#         cx += 1  # next character = one cell to the right
#       cy += 1 # next line = one cell up

#   #Method to make the "guess" for A*
#   def h(self, cell, goal):
#     return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

#   #Method to get the optimal path from A*, then reverse it (so path[0] is the starting cell...):
#   def reconstruct_path(self, came_from, current):
#     path = [current]
#     while current in came_from:
#       current = came_from[current]
#       path.append(current)
#     path.reverse()
#     return path

#   #Method to get the valid neighboring cells to give to A*
#   def get_neighbors(self, cell):
#     x, y = cell#X and y location of the current cell
#     moves = [(-1,0),(1,0),(0,-1),(0,1),     (-1,-1),(-1,1),(1,-1),(1,1)]#Represents left, right, down, up, left/down diagonal, left/up diagonal,
#                                                                     #right/down diagonal, right/up diagonal respectively
#     result = []#An empty list to collect the valid neighbour cells
#     for dx, dy in moves:
#       neighbor_x, neighbor_y = x + dx, y + dy#Get the location of one of the neighboring cells
#       if not (0 <= neighbor_x < 120 and 0 <= neighbor_y < 80): continue #If this proposed neighbor is invalid, skip it
#       if self.offlimits[neighbor_y * 120 + neighbor_x] == "#": continue #If it's a wall/buffer, skip it
#       if dx != 0 and dy != 0:#it's a diagonal move (since one of dx/dy is 0 for a cardinal move)
#         if self.offlimits[y * 120 + (x + dx)] == "#" or self.offlimits[(y + dy) * 120 + x] == "#":#This is that corner cutting rule from the readme.
#                                                                                                   #The 2 cells in this conditional are the ones
#                                                                                                   #neighboring it. If either of these
#                                                                                                   #are blocked, then the robot
#                                                                                                   #can't fit through there
#           continue
#       result.append((neighbor_x, neighbor_y))
#     return result

#   #Computes the cost of a step for A*. A cardinal move = 1, but a digonal move is larger, so it's sqrt(2)
#   def step_cost(self, a, b):
#     return math.hypot(b[0] - a[0], b[1] - a[1])

#   #This is the A* path-finding algorithm
#   # f = g + h, where g is the distance to get from the start to the current node
#   # and h is the heuristic (guess of the cost of going from the current node to the goal)
#   def a_star(self, start, goal):
#     open_list = [(self.h(start, goal), start)] #Starts with just the robot's cell
#     g_score = {start: 0}  #Steps to reach each cell, best known so far
#     came_from = {} #The dictionary that holds the best path we've found

#     #NOTE: The loop ends either when we found our path (the return) or when there are no candidate cells left (no valid path)
#     while open_list:
#       _, current = heapq.heappop(open_list) #Grab the most promising cell by removing it from open list
#       if current == goal:#If we actually found the path that reaches the goal, get that path
#         return self.reconstruct_path(came_from, current)

#       for neighbor in self.get_neighbors(current):
#         tentative_g = g_score[current] + self.step_cost(current, neighbor)
#         if neighbor not in g_score or tentative_g < g_score[neighbor]:#If we've never reached this cell before (so it's the best route to it),
#                                                                       #or this is the best route to this cell found so far
#           g_score[neighbor] = tentative_g
#           came_from[neighbor] = current
#           f = tentative_g + self.h(neighbor, goal)
#           heapq.heappush(open_list, (f, neighbor))#Push the neighbor and its f score to the list of cells to be explored
#     return None #If there was no valid path

#   #This is the function that ensures we don't get stuck on a corner by entering cells that are now blocked
#   def nearest_free_cell(self, cell):
#     if self.offlimits[cell[1] * 120 + cell[0]] != "#":
#       return cell #What normally happens. Has to be at the top to not waste time.

#     x0, y0 = cell
#     for r in range(1, 121): #Grow the search ring outward. 120 because that's the width of the arena
#       for dx in range(-r, r + 1):
#         for dy in range(-r, r + 1):
#           if max(abs(dx), abs(dy)) != r:
#             continue   #We're only looking at the outer edge of the ring (we already checker the inner ones before)
#           nx, ny = x0 + dx, y0 + dy
#           if 0 <= nx < 120 and 0 <= ny < 80 and self.offlimits[ny * 120 + nx] != "#":#If it's in bounds and not a wall
#             return (nx, ny)
#     return cell #Just a fallback in case something happens and none of the other returns triggered (no idea what that'd be tho)

#   def step(self, pose:tuple[float, float], scan:tuple[int, int, list[str]]) -> tuple[float, float]:
#     """
#     called once per tick.

#     pose: (x, y) metres from SLAM, ~2 cm gaussian noise. WHERE THE ROBOT THINKS IT IS
#     scan: (cx0, cy0, rows) -- a (2*sense_cells+1)^2 window of '#'/'.' around the robot. rows[j][i] is cell (cx0+i, cy0+j).
#           everything in the window is observed, nothing outside it is.
#           the window origin comes from the noisy pose, so walls can land one cell off between scans.
#     returns: (vx, vy) world-frame velocity command in m/s. sim clamps speed and acceleration.
#     """
#     goal_x, goal_y = self.cfg["goal"]            # Position of the goal in metres

#     #Note: The scan window is 51x51
#     #Updates the map with any new info
#     self.update_map(scan)

#     #Run the A* path-finding algo to find the best path
#     #(This gives a a list of tuples with coords, like [(1,1), (2,2,)...])
#     start_cell = (math.floor(pose[0] / self.cfg["resolution"]), math.floor(pose[1] / self.cfg["resolution"]))#Convert from metres to cell indices
#     start_cell = self.nearest_free_cell(start_cell)
#     goal_cell = (math.floor(goal_x / self.cfg["resolution"]), math.floor(goal_y / self.cfg["resolution"]))
#     path = self.a_star(start_cell, goal_cell)

#     #Move using that path
#     if path:
#       target_cell_x, target_cell_y = path[1]#The coords in metres of the cell we wanna go to
#       target_x = target_cell_x * self.cfg["resolution"] + self.cfg["resolution"] / 2 #Convert back to m (the + part is to get the
#                                                                                       centre of the cell)
#       target_y = target_cell_y * self.cfg["resolution"] + self.cfg["resolution"] / 2
#       distance_x, distance_y = target_x - pose[0], target_y - pose[1]  # How far we are from that goal in both dimensions
#       dist = math.hypot(distance_x, distance_y)            # Straight-line distance from goal
#       vx = distance_x / dist * 2.0  #(distance_x/dist) is the fraction of the trip in the x dir (between -1,
#                                     needing to go left, and 1, going right.)
#                                     #Then multiply by 2 (max speed). Essentially this is making a unit vector, then saying we wanna go top speed.
#       vy = distance_y / dist * 2.0
#     else:
#       vx, vy = 0, 0

#     return (vx, vy)

#   def debug(self) -> dict:
#     """
#     optional, for `harness.py --viz` only.
#     keys: blocked (cells), free (cells), path ([(x, y), ...]).
#     """
#     return {}

# """
# The map works like:
# 80 high by 120 wide cells. Each cell has a number associated with it.

# Self.grid starts at y=0, x=0, and each row is 1 y higher (so index 120 is x=0 y=120)
# The robot only knows 51x51 cells.

# The target position (in m) is self.cfg["goal"]
# """