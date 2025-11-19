import pickle
import random
import numpy as np
from typing import List, Tuple, Deque, Dict, Optional
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Coord data object: (nrow, ncol)
Coord = Tuple[int, int]


def findOpenings(maze):
    x = []
    nr, nc = maze.shape
    for ir in range(nr):
        if maze[ir, 0]:
            x.append([ir, 0])
        if maze[ir, nc-1]:
            x.append([ir, nc-1])
    for ic in range(nc):
        if maze[0, ic]:
            x.append([0, ic])
        if maze[nr-1, ic]:
            x.append([nr-1, ic])
    return random.sample(x, 2)

class MazeSolver:
    DIRECTIONS: Tuple[Coord, ...]= ((1, 0), (-1, 0), (0, 1), (0, -1)) # up, down, left, right (van moore)

    def __init__(self, maze: np.ndarray):
        # Save maze and its shape
        self.maze = maze
        self.num_rows, self.num_cols = self.maze.shape

        # We obtain goal by using findOpenings
        openings = findOpenings(self.maze)
        (sr, sc), (gr, gc) = openings
        self.start: Coord = (sr, sc)
        self.goal:  Coord = (gr, gc)

        # Stores the predecessor node for path reconstruction
        self.parent: Dict[Coord, Optional[Coord]] = {}

        # run the BFS algorithm to generate tree
        self.bfs()

    def is_free(self, row: int, col: int):
        return (
            0 <= row < self.num_rows and  # are we within row bounds?
            0 <= col <= self.num_cols and # are we within col bounds?
            self.maze[row, col]   # maze is a boolean array, so True says we're open
            )
    
    def bfs(self):
        queue: Deque[Coord] = deque([self.start]) # the queue should start with the entrance
        self.parent = {self.start: None} # start cannot have parents (top of the tree)

        goal_found = False

        while queue:
            row, col = queue.popleft()

            if (row, col) == self.goal: # exit condition of while loop is when we find the goal
                goal_found = True
                break

            for delta_row, delta_col in self.DIRECTIONS: # van neumann neighbourhood
                next_row, next_col = row + delta_row, col + delta_col # gives us option chain
                neighbor = (next_row, next_col)

                if self.is_free(next_row, next_col) and neighbor not in self.parent: #we shouldn't go back up and it should be free
                    self.parent[neighbor] = (row, col)
                    queue.append(neighbor)
            
        if not goal_found and self.goal in self.parent: # if a goal cannot be found, probably the goal is the parent
            del self.parent[self.goal]

    def path(self):
        if self.goal not in self.parent:
            return None # Path not found
            
        p: List[Coord] = []
        cur: Optional[Coord] = self.goal
        
        while cur is not None: # traverse parent
            p.append(cur)
            cur = self.parent[cur]
        return p[::-1] # Reverse to get path from start to goal

    def plot_path(self):
        path_coords = self.path()
        
        # create the base visualization array (0=Wall, 1=Free, 2=Path, 3=Start/Goal)
        plot_array = np.zeros(self.maze.shape, dtype=int)
        
        # free space (where maze is True) is marked as 1, 0 is wall
        plot_array[self.maze] = 1 
        
        # 2. Mark the Path
        if path_coords:
            # Mark the path points as 2
            r_path, c_path = zip(*path_coords)
            plot_array[r_path, c_path] = 2

        # Mark Start and Goal (Highest priority: 3)
        plot_array[self.start] = 3
        plot_array[self.goal] = 3

        # Custom colors for the four values (0, 1, 2, 3)
        colors = ['#111827', '#E5E7EB', '#3B82F6', '#FBBF24']
        cmap = ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(self.num_cols / 5, self.num_rows / 5), dpi = 150) # size scales with maze

        # Display the array as an image
        ax.imshow(plot_array, cmap=cmap)

        # grid lines for cell boundaries
        #ax.set_xticks(np.arange(self.num_cols + 1) - 0.5, minor=False)
        #ax.set_yticks(np.arange(self.num_rows + 1) - 0.5, minor=False)
        #ax.grid(which="major", color='k', linestyle='-', linewidth=0.5)

        # Remove tick labels and borders for a cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_frame_on(False)
        
        # text labels for Start and Goal
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center', color='black', fontsize=12)
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', color='black', fontsize=12)

        path_len = len(path_coords)

        plt.savefig(f"{self.num_rows}x{self.num_cols}sol.png", dpi=300, bbox_inches="tight")
        title = f"Maze Solution (Path Length: {path_len})"
        ax.set_title(title, fontsize=self.num_cols / 2)
        plt.tight_layout()
        plt.show()


with open ("maze4x6.pckl", "rb") as f:
  maze = pickle.load(f)
  solve = MazeSolver(maze)
  solve.plot_path()

with open ("maze5x5.pckl", "rb") as f:
  maze = pickle.load(f)
  solve = MazeSolver(maze)
  solve.plot_path()

with open ("maze9x9.pckl", "rb") as f:
  maze = pickle.load(f)
  solve = MazeSolver(maze)
  solve.plot_path()


with open ("maze10x10.pckl", "rb") as f:
  maze = pickle.load(f)
  solve = MazeSolver(maze)
  solve.plot_path()

with open ("maze100x100.pkl", "rb") as f:
  maze = pickle.load(f)
  solve = MazeSolver(maze)
  solve.plot_path()
