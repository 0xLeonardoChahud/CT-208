import random
import numpy as np


class Node:
    def __init__(self, value, region, pos):
        self.value = value
        self.region = region
        self.i, self.j = pos


class Matrix:
    def __init__(self, size):

        # General information
        self.size = size
        self.node_cnt = self.size**2

        # Init matrix
        self.nodes = [Node(None, None, (i,j)) for i in range(size) for j in range(size)]
        self.matrix = np.array(self.nodes, dtype=object)
        self.matrix = self.matrix.reshape(size, size)

        # Regions matrix
        self.regions = np.zeros((size,size))

        # Draw regions
        pos = [(i,j) for i in range(self.size) for j in range(self.size)]
        
        label = 0
        p = (0,0)
        pn = p 
        nbs = [(1,0),(0,1),(-1,0),(0,-1)]
        
        while pos: 
            steps = random.randint(1, self.node_cnt//3)
            label += 1
            while steps != 0:
                i,j = p
                if self.regions[i][j] == 0:
                    self.regions[i][j] = label
                    pos.remove(p)
                
                random.shuffle(nbs)
                for k,l in nbs:
                    nbr = (i+k, j+l)
                    k,l = nbr
                    if self.valid_pos(k,l) and self.regions[k][l] == 0:
                        self.regions[k][l] = label
                        pn = nbr
                        steps -= 1
                        pos.remove(pn)
                        break
                if not pos:
                    break
                if pn == p:
                    p = random.choice(pos)
                    pn = p
                    break
                else:
                    p = pn
                    

        print(self.regions)

    def valid_pos(self, x, y):
        return (x >= 0 and x < self.size) and (y >= 0 and y < self.size)

    def get(self, x, y):
        return self.matrix[x][y]
    
    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            x,y = index
            if x < 0 or x > self.side:
                return None
            if y < 0 or y > self.side:
                return None
            return self.matrix[x][y]

Matrix(8)
