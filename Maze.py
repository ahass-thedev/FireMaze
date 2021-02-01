from numpy import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


class Maze():
    """Cust Maze"""

    def __init__(self, dim, percent):
        percent = percent / 100
        #mazeArry = random.randint(2, size=(dim, dim))
        self.maze = np.random.choice(
            a=[0, 1],
            size=(dim, dim),
            p=[1-percent, percent])
        #idx = np.flatnonzero(mazeArry)
        #print("idx: " , idx)
        #N = np.count_nonzero(mazeArry != 0) - int(round(.25 * mazeArry.size))
        #print("N" , N)
        #np.put(mazeArry, np.random.choice(idx, size=N, replace=False), 0)

        #mazeArry[indices] = 0
        print(self.maze)
        self.maze[0][0] = 0
        self.maze[dim-1][dim-1] = 0
        print("Wall Percentage: " , np.count_nonzero(self.maze!=0)/float(self.maze.size))
        self.display_Maze(self.maze)

    def display_Maze(self, mazeArry):
        colormap = colors.ListedColormap(["white","black"])
        plt.figure(figsize=(5, 5))
        plt.imshow(mazeArry,cmap=colormap)
        plt.show()

    def DFS(self,start,end):
        """DFS Implementation goes here"""
    def BFS (self,start,end):
        """BFS Implementation goes here"""
    def A_Star(self,start,end):
        """A* Implementation goes here"""
    def advance_fire_one_step(self):
        """Look at project description for this method"""

if __name__ == '__main__':
    dimension = int(input("Enter Dimension of the maze:\n"))
    percent = float(input("Enter Percentage of Walls(60 -> 60%):\n"))
    mazeRun = Maze(dimension, percent)
