import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors
from itertools import product
import pandas as pd


class Node:
    # properties needed for each "cell"
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Maze:
    """Cust Maze"""

    def __init__(self, dim, wall_percent, q):
        self.dim = dim
        self.q = q
        # set the wall rate to a decimal
        wall_percent = wall_percent / 100
        # mazeArry = random.randint(2, size=(dim, dim))
        # randomly create 2D array with the correct percentages of walls and open space

        self.maze = np.random.choice(
            a=[0, 1],
            size=(dim, dim),
            p=[1 - wall_percent, wall_percent])

        # set the maze on fire - selecting only white space
        for x in range(0, dim):
            for y in range(0, dim):
                if self.maze[x][y] == 0:
                    if random.uniform(0, 1) <= q:  # q is the fire spread rate chosen by user
                        self.maze[x][y] = 2

        print("Before the fire spread")
        print(self.maze)
        # self.display_maze()
        # self.advance_fire_one_step()

        # ensures the first spot is open
        self.maze[0][0] = 0
        # elf.maze[1][1] = 1
        # ensures the goal spot is open
        self.maze[dim - 1][dim - 1] = 0
        self.maze[3][3] = 2
        # attempts to find a solved path
        path, nodes_visited = self.a_star()
        # print("Solved Path: ", path)
        # self.path = path

        # highlights the open path
        for x, y in path:
            self.maze[x][y] = 3
        # print("After the fire spread")
        print(self.maze)
        # print("Wall & Fire Percentage: ", np.count_nonzero(self.maze != 0) / float(self.maze.size))
        # print("Amount of nodes visited", nodes_visited)
        # print("Length of path", len(path))

        # calls to display the solved path
        # self.fire_maze_runner()
        self.display_maze()
        for x, y in path:
            self.maze[x][y] = 0

        count = 0
        for i in range (len(self.maze)):
            for j in range (len(self.maze[i])):
                if self.maze[i][j] == 2:
                    count += 1
        print("Current fire in maze is: ", count)
        self.advance_fire_one_step()
        path, nodes_visited = self.a_star()
        for x, y in path:
            self.maze[x][y] = 3
        count = 0
        for i in range (len(self.maze)):
            for j in range (len(self.maze[i])):
                if self.maze[i][j] == 2:
                    count += 1
        print("After fire in maze is: ", count)
        # self.advance_fire_one_step()
        self.display_maze()

        """
        density_vs_nodes = pd.DataFrame(columns=('Density', 'Average Nodes'))
        total_nodes = 0
        i = 0
        attempts = 0
        n_trials = 0
        for density in np.arange(0.6, 1, 0.1):
            total_nodes = 0
            successful_attempts = 0
            while successful_attempts < 10:
                print("Successful attempts so far", successful_attempts)
                self.maze = np.random.choice(
                    a=[0, 1],
                    size=(dim, dim),
                    p=[1 - density, density])
                self.maze[0][0] = 0
                self.maze[dim - 1][dim - 1] = 0
                try:
                    path, nodes_visited = self.a_star()
                    # for x, y in path:
                    # self.maze[x][y] = 3
                    # self.display_maze()
                    if attempts == -1:
                        print(
                            f"Test {n_trials} has failed multiple times at current density {density} and will not repeat")
                        n_trials += 1
                        attempts = 0
                        continue
                    if nodes_visited == -1:
                        print("No path found, rerunning test")
                        attempts += 1
                        print(f"Current trial:  {successful_attempts} ,Attempt:  {attempts}  at {density} density")
                        continue
                except:
                    print("An error has occurred, rerunning test")
                    attempts += 1
                    continue
                total_nodes += nodes_visited
                # attempts = 0
                successful_attempts += 1
                print("Density", density)
                print("Nodes visited", nodes_visited)
                n_trials += 1

            average_nodes = total_nodes / successful_attempts
            density_vs_nodes.loc[i] = [density, average_nodes]
            density_vs_nodes.to_csv("density_vs_nodes.csv", mode='a', index=False)
            i += 1

        print(density_vs_nodes.head())
        density_vs_nodes.plot(x="Density", y="Average Nodes")
        plt.show()
        """

    def display_maze(self):
        """check if there is fire in the maze"""
        if np.any(self.maze == 2):
            colormap = colors.ListedColormap(["white", "black", "orange", "green"])
        else:
            """if no fire choose only walls, path and path colors"""
            colormap = colors.ListedColormap(["white", "black", "green"])


        # plt.figure(figsize=(5, 5))
        fig, ax = plt.subplots(figsize=(7, 7))  # setting size of plot window
        # plt.imshow(self.maze)

        ax.imshow(self.maze,cmap=colormap)
        ax.scatter(0, 0, marker="*", color="cyan", s=200)  # show the beginning star
        ax.scatter(self.dim - 1, self.dim - 1, marker="*", color="yellow", s=200)  # show the goal
        plt.show()  # display the solved maze

    def dfs(self, start, end):
        """DFS Implementation goes here"""

    def bfs(self, start, end):
        """BFS Implementation goes here"""

    def a_star(self):

        """A* Implementation goes here"""
        start = (0, 0)
        end = (self.dim - 1, self.dim - 1)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        '''
        
        start = (0, 0)
        goal = (self.dim - 1, self.dim - 1)

        fringe = PriorityQueue()
        fringe.put(start, 0)
        closed_list = {start: None}
        total_cost = {start: 0}

        while not fringe.empty():
            current = fringe.get()
            if current == goal:
                break
            x = current[0]
            y = current[1]
            for next in list(self.neighbours_list(current)):
                new_cost = total_cost[current] + 1
                if next not in total_cost or new_cost < total_cost[next]:
                    total_cost[next] = new_cost
                    priority = new_cost + self.euclid_dist(x, y)
                    fringe.put(next, priority)
                    closed_list[next] = current
        return closed_list, total_cost
        '''

        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0
        nodes_visited = 0

        open_list = []
        closed_list = []
        empty_list = []
        path_exists = False

        open_list.append(start_node)

        while open_list:
            nodes_visited += 1
            # self.advance_fire_one_step()
            current_node = open_list[0]
            if nodes_visited % 500 == 0:
                print(nodes_visited)
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            open_list.pop(current_index)
            closed_list.append(current_node)
            if current_node == end_node:
                path_exists = True
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1], nodes_visited

            if nodes_visited > 10000:
                path_exists = False
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1], -1

            children = []
            for new_position in neighbors:
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                if node_position[0] > (len(self.maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(self.maze[len(self.maze) - 1]) - 1) or node_position[1] < 0:
                    continue

                if self.maze[node_position[0]][node_position[1]] != 0:
                    continue

                new_node = Node(current_node, node_position)

                children.append(new_node)

            for child in children:

                for closed_child in closed_list:
                    if child == closed_child:
                        continue
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                open_list.append(child)

    def advance_fire_one_step(self):
        """Look at project description for comments"""
        tempmaze = self.maze
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        k = 0
        print("Advancing fire one step")
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                if self.maze[x][y] == 0:
                    # print("Empty path at", x, " ", y)
                    for i, j in neighbors:
                        if x + i < self.dim and y + j < self.dim:
                            if self.maze[x + i][y + j] == 2:
                                # print("Neighbor is on fire at", x, " ", y)
                                k += 1
                    prob = 1 - (1 - self.q) ** k
                    k = 0

                    if random.uniform(0, 1) <= prob:
                        # self.maze[x][y] = 1
                        self.maze[x][y] = 2
                        # print("Fire advanced to", x, " ", y)

        # return self.maze

    """calculates the euclidean distance for bfs and A*"""

    def euclid_dist(self, x, y):
        return np.sqrt((self.dim - 1 - x) ** 2 + (self.dim - 1 - y) ** 2)

    """return a full list of coords of a node's neighbors"""

    def total_cost(self, node, goal):
        return node.cost + self.euclid_dist(node, goal)

    """returns a list of the current positions """

    def neighbors_list(self, size):
        for c in product(*(range(n - 1, n + 2) for n in self)):
            if c != self and all(0 <= n < size for n in c):
                yield c


if __name__ == '__main__':
    dimension = int(input("Enter Dimension of the maze:\n"))
    """Change?"""
    percent = float(input("Enter Percentage of Walls(60 -> 60%):\n"))
    flame_rate = float(input("Enter Flammability Rate (Max is 1)\n"))
    mazeRun = Maze(dimension, percent, flame_rate)
