import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors


class Maze():
    """Cust Maze"""

    def __init__(self, dim, percent, q):
        percent = percent / 100
        # mazeArry = random.randint(2, size=(dim, dim))
        self.maze = np.random.choice(
            a=[0, 1],
            size=(dim, dim),
            p=[1 - percent, percent])

        # set the maze on fire - selecting only white space
        for x in range(0, dim):
            for y in range(0, dim):
                if self.maze[x][y] == 0:
                    if random.uniform(0, 1) <= q:
                        self.maze[x][y] = 2

        # idx = np.flatnonzero(mazeArry)
        # print("idx: " , idx)
        # N = np.count_nonzero(mazeArry != 0) - int(round(.25 * mazeArry.size))
        # print("N" , N)
        # np.put(mazeArry, np.random.choice(idx, size=N, replace=False), 0)

        # mazeArry[indices] = 0
        self.q = q
        self.maze[0][0] = 0
        self.maze[dim - 1][dim - 1] = 0
        print(self.maze)
        print("Wall & Fire Percentage: ", np.count_nonzero(self.maze != 0) / float(self.maze.size))
        self.dim = dim
        self.display_maze()
        print("before the maze")
        # self.fire_maze_runner()

    def display_maze(self):
        if np.any(self.maze == 2):
            colormap = colors.ListedColormap(["white", "black", "orange"])
        else:
            colormap = colors.ListedColormap(["white", "black"])
        # plt.figure(figsize=(5, 5))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.maze, cmap=colormap)
        ax.scatter(0, 0, marker="*", color="cyan", s=200)
        ax.scatter(self.dim - 1, self.dim - 1, marker="*", color="yellow", s=200)
        plt.show()

    def dfs(self, start, end):
        """DFS Implementation goes here"""

    def bfs(self, start, end):
        """BFS Implementation goes here"""

    def a_star(self, start, end):
        """A* Implementation goes here"""

    def advance_fire_one_step(self):
        tempmaze = self.maze
        neighbors_on_fire = 0
        k = 0
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                if self.maze[x][y] != 1 and self.maze[x][y] != 2:
                    # row above
                    if self.maze[x - 1][y - 1] == 2:
                        k += 1
                    if self.maze[x][y - 1] == 2:
                        k += 1
                    if self.maze[x + 1][y - 1] == 2:
                        k += 1
                    # row below
                    if self.maze[x - 1][y + 1] == 2:
                        k += 1
                    if self.maze[x][y + 1] == 2:
                        k += 1
                    if self.maze[x + 1][y + 1] == 2:
                        k += 1
                    # middle left
                    if self.maze[x - 1][y] == 2:
                        k += 1
                    # middle right
                    if self.maze[x + 1][y] == 2:
                        k += 1
                    prob = 1 - (1 - self.q) ** k

                    if random.uniform(0, 1) <= prob:
                        # self.maze[x][y] = 1
                        tempmaze[x][y] = 1
        return tempmaze

    def euclid_dist(self, loc):
        x = (self.dim - 1 - loc[0]) ** 2
        y = (self.dim - 1 - loc[1]) ** 2
        return np.sqrt(x + y)

    def fire_maze_runner(self):
        """Main game loop"""
        print("From the main game")
        while True:
            "Main loop"


if __name__ == '__main__':
    dimension = int(input("Enter Dimension of the maze:\n"))
    percent = float(input("Enter Percentage of Walls(60 -> 60%):\n"))
    flame_rate = float(input("Enter Flammability Rate (Max is 1)"))
    mazeRun = Maze(dimension, percent, flame_rate)
