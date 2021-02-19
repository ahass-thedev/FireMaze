import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors
import pandas as pd
from collections import deque


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


def get_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]


"""calculates the euclidean distance for A*"""


def euclid_dist(x, y):
    # return np.sqrt((self.dim - 1 - x) ** 2 + (self.dim - 1 - y) ** 2)
    return np.sqrt(((x.position[0] - y.position[0]) ** 2) + (
            (x.position[1] - y.position[1]) ** 2))


class Maze:
    """Cust Maze"""

    def __init__(self, dim, wall_percent, q):
        self.dim = dim
        self.q = q
        """
        # set the wall rate to a decimal
        wall_percent = wall_percent / 100
        # mazeArry = random.randint(2, size=(dim, dim))
        # randomly create 2D array with the correct percentages of walls and open space
        self.maze = np.random.choice(
            a=[0, 1],
            size=(dim, dim),
            p=[1 - wall_percent, wall_percent])
        """
        start = (0, 0)
        end = (self.dim - 1, self.dim - 1)

        """
        # set the maze on fire - selecting only white space
        for x in range(0, dim):
            for y in range(0, dim):
                if self.maze[x][y] == 0:
                    if random.uniform(0, 1) <= q:  # q is the fire spread rate chosen by user
                        self.maze[x][y] = 2
        """

        """
        #print("Before the fire spread")
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
        path, nodes_visited = self.a_star(start, end)

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
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 2:
                    count += 1
        print("Current fire in maze is: ", count)
        self.advance_fire_one_step()
        path, nodes_visited = self.a_star()
        for x, y in path:
            self.maze[x][y] = 3
        count = 0
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 2:
                    count += 1
        print("After fire in maze is: ", count)
        # self.advance_fire_one_step()
        self.display_maze()
        """
        density_vs_nodes = pd.DataFrame(columns=('Ignition Rate', 'Success Rate'))
        total_nodes = 0
        i = 0
        attempts = 0
        n_trials = 0
        for self.q in np.arange(0.0, 1, 0.05):
            total_nodes = 0
            successful_attempts = 0
            attempts = 0
            print("Setting new ignition rate:", self.q)
            while attempts < 10:
                print("Successful attempts so far", successful_attempts)
                self.maze = np.random.choice(
                    a=[0, 1],
                    size=(dim, dim),
                    p=[1 - .10, .10])
                self.maze[0][0] = 0
                self.maze[dim - 1][dim - 1] = 0

                for x in range(0, dim):
                    for y in range(0, dim):
                        if self.maze[x][y] == 0:
                            if random.uniform(0, 1) <= self.q:  # q is the fire spread rate chosen by user
                                self.maze[x][y] = 2
                try:
                    path, nodes_visited = self.bfs(start, end)
                    """for x, y in path:
                        self.maze[x][y] = 3
                    self.display_maze()"""
                    if attempts == -1:
                        print(
                            f"Test {n_trials} has failed multiple times at current ignition rate {self.q} and will "
                            f"not repeat")
                        n_trials += 1
                        attempts = 0
                        continue
                    if nodes_visited == -1:
                        print("No path found, rerunning test")
                        attempts += 1
                        print(
                            f"Current trial:  {successful_attempts} ,Attempt:  {attempts}  at {self.q} Ignition rate")
                        continue
                except ZeroDivisionError:
                    print("An error has occurred, rerunning test")
                    attempts += 1
                    continue
                total_nodes += nodes_visited
                # attempts = 0

                successful_attempts += 1
                attempts += 1
                print("Ignition rate", self.q)
                print("Nodes visited", nodes_visited)
                n_trials += 1
            #for x, y in path:
                #self.maze[x][y] = 3
            #self.display_maze()
            #print(self.maze)
            success_rate = successful_attempts / attempts
            density_vs_nodes.loc[i] = [self.q, success_rate]
            density_vs_nodes.to_csv("bfs_fire_success.csv", mode='a', index=False)
            i += 1
        print(density_vs_nodes.head())
        density_vs_nodes.plot(x="Ignition Rate", y="Success Rate")
        plt.show()


        """
        for density in np.arange(0.0, 1, 0.1):
            total_nodes = 0
            successful_attempts = 0
            attempts = 0
            while attempts < 10:
                print("Successful attempts so far", successful_attempts)
                self.maze = np.random.choice(
                    a=[0, 1],
                    size=(dim, dim),
                    p=[1 - density, density])
                self.maze[0][0] = 0
                self.maze[dim - 1][dim - 1] = 0
                try:
                    path, nodes_visited = self.bfs(start, end)
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
                attempts += 1
                print("Density", density)
                print("Nodes visited", nodes_visited)
                n_trials += 1

            success_rate = successful_attempts / attempts
            density_vs_nodes.loc[i] = [density, success_rate]
            density_vs_nodes.to_csv("bfs_success.csv", mode='a', index=False)
            i += 1

        print(density_vs_nodes.head())
        density_vs_nodes.plot(x="Density", y="Success Rate")
        plt.show()
        """
        """This loop is how we tested for successful attempts"""
        """density_vs_nodes_a_star = pd.DataFrame(columns=('Density', 'Average Nodes'))
        total_nodes = 0
        i = 0
        attempts = 0
        n_trials = 0

        for density in np.arange(0.3, 1, 0.1):
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
                    path, nodes_visited = self.bfs(start, end)

                    if attempts == -1:
                        print(f"Test {n_trials} has failed multiple times at current density {density} and will not repeat")
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
                This for loop and self.display_maze() print the maze with a solved path
                for x, y in path:
                    self.maze[x][y] = 3
                self.display_maze()
                successful_attempts += 1
                print("Density", density)
                print("Nodes visited", nodes_visited)
                n_trials += 1
            average_nodes = total_nodes / successful_attempts
            density_vs_nodes_a_star.loc[i] = [density, average_nodes]
            density_vs_nodes_a_star.to_csv("density_vs_nodes_dfs.csv", mode='a', index=False)
            i += 1

        print(density_vs_nodes_a_star.head())
        density_vs_nodes_a_star.plot(x="Density", y="Average Nodes")
        plt.show()"""

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

        ax.imshow(self.maze, cmap=colormap)
        ax.scatter(0, 0, marker="*", color="cyan", s=200)  # show the beginning star
        ax.scatter(self.dim - 1, self.dim - 1, marker="*", color="yellow", s=200)  # show the goal
        plt.show()  # display the solved maze

    def maze2graph(self):
        """Converts the maze into a graph for the dfs to solve easier"""
        height = len(self.maze)
        width = len(self.maze[0]) if height else 0
        """Creates the graph based on the open spots of the maze."""
        graph = {(x, y): [] for y in range(width) for x in range(height) if not self.maze[x][y]}
        for row, col in graph.keys():
            """Checks to see if the nodes in the graph are truly empty, and adds it to graph it it is"""
            if col < width - 1 and not self.maze[row][col + 1]:
                graph[(row, col)].append((row, col + 1))
                graph[(row, col + 1)].append((row, col))
            if row < height - 1 and not self.maze[row + 1][col]:
                graph[(row, col)].append((row + 1, col))
                graph[(row + 1, col)].append((row, col))
        return graph

    def dfs(self, start, end):
        """DFS Implementation goes here"""
        # use self.maze to get the maze...
        start = start
        goal = end
        start_node = Node(None, start)
        end_node = Node(None, goal)
        stack = deque([start])
        nodes_visited = 0
        visited = set()
        graph = self.maze2graph()
        empty_visit = set()
        has_path = False

        while stack:
            nodes_visited += 1
            current = stack.pop()
            # print(path)
            # print(current)
            if current == goal:
                """If the current node is the end node, we confirm that it has a path 
                and return the list of visited nodes to show the path"""
                has_path = True
                visited.add(current)
                print("Path Exists? ", has_path)
                return visited, nodes_visited
            if current in visited:
                """Helps with back-tracking"""
                continue
            visited.add(current)
            try:
                for neighbor in graph[current]:
                    """Checks if the neighbor is a true neighbor, if it is add it to the stack"""
                    stack.append(neighbor)
            except KeyError:
                continue
        # print(visited)
        print("Path Exists? ", has_path)
        return empty_visit, -1

    def get_bfs_path(self, path, ans):
        """Current node is set to the end node to begin path creation"""
        current_node = ans
        end = []
        """moves though the nodes that were visted to create the solved path rather than all nodes visited"""
        while (current_node != (0, 0)):
            end.append(current_node)
            current_node = path[current_node]
        """Adds the starter node to the path then reverses it to reset the queue"""
        end.append((0, 0))
        end.reverse()
        """returns the list of completed path"""
        return end

    def bfs(self, start, end):
        """Implementation of BFS"""
        """ BFS uses queue to search childern and add them to fringe"""
        """Maze length will allow to find end and the size of the maze"""
        maze_length = self.dim
        maze_sol = (maze_length-1, maze_length-1)
        """The visisted list is used to define the path that finds the solution as well as identify dead ends"""
        visited = [[False for x in range(maze_length)] for y in range(maze_length)]
        """path is the final path to the solution"""
        path = {}
        total = 0
        """The original neighbors that the children are made form"""
        neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        path_exists = False
        """The main queue that is manipulated through the algoritihim"""
        queue = []
        """Nodes_visited is incremented per node visited to calcualte the total nodes"""
        nodes_visited = 0
        queue.append((0,0))
        visited[0][0] = True

        nodes_visited = 0
        """While the queue is populated continue running the BFS algorithim"""
        while queue:
            self.advance_fire_one_step()
            nodes_visited += 1
            total = max(total, len(queue))
            """Current node is the node that is on the front of the queue"""
            current_node = queue.pop()
            """Checking if the the current node is the maze solution"""
            if current_node == maze_sol:
                path_exists = True
                break
            """For the child is a neighbor of the current node run the movement loop"""
            for child in neighbors:
                """This is for incrementing and moving the position of the node"""
                spot = tuple(sum(x) for x in zip(current_node, child))

                x = spot [0]
                y = spot[1]
                """These if statements are to ensure that the x and y spots are open nodes in the maze and good to move into"""
                if 0 <= x < self.dim:

                    if 0 <= y < self.dim:

                        if not self.maze[x][y]:

                            if not visited[x][y]:
                                """If the node is open then add it to the queue to be visited next"""
                                queue.append(spot)
                                path[spot] = current_node
                                visited[spot[0]][spot[1]] = True
        """If there is no possible paths return the path at the dead end and -1 so the Init is aware there is no path"""
        if not path_exists:
            path = []
            path.append(current_node)
            """Path exists will show the maze is blocked"""
            path_exists = False

            return path[::-1], -1
        """Return the get path method with the solved path and end node so the path is created and not all visited nodes"""
        return self.get_bfs_path(path, maze_sol), nodes_visited

    def a_star(self, start, end):

        """A* Implementation goes here"""

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
        end_node = Node(None, end)
        start_node.g = start_node.h = start_node.f = 0
        end_node.g = end_node.h = end_node.f = 0
        nodes_visited = 0

        fringe = []
        closed_list = []

        fringe.append(start_node)
        blocked_nodes = 0
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] != 0:
                    blocked_nodes += 1

        while fringe:
            nodes_visited += 1

            current_node = fringe[0]
            # if nodes_visited % 500 == 0:
                # print(nodes_visited)
                # print("Fire has advanced 500x")
            current_index = 0
            for index, item in enumerate(fringe):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            fringe.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                return get_path(current_node), nodes_visited

            if nodes_visited > (self.dim * self.dim - blocked_nodes):
                return get_path(current_node), -1

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
                child.h = euclid_dist(child, end_node)
                child.f = child.g + child.h
                for open_node in fringe:
                    if child == open_node and child.g > open_node.g:
                        continue
                fringe.append(child)

    def advance_fire_one_step(self):
        """Look at project description for comments"""
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        k = 0
        # print("Advancing fire one step")
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


if __name__ == '__main__':
    dimension = int(input("Enter Dimension of the maze:\n"))
    """Change?"""
    percent = float(input("Enter Percentage of Walls(60 -> 60%):\n"))
    flame_rate = float(input("Enter Flammability Rate (Max is 1)\n"))
    mazeRun = Maze(dimension, percent, flame_rate)
