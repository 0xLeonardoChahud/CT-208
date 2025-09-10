import random
import tkinter as tk
import time
import math
import argparse
import numpy as np
import itertools
from collections import defaultdict

import suguru_gui

class Region:
    def __init__(self, id, cells):
        self.id = id
        self.length = len(cells)
        self.cells = {(c.row,c.col):c for c in cells}
    
    def try_fill(self):
        pass
 
class Cell:
    def __init__(self, row, col):
        self.numbers = set()
        self.value = 0
        self.region = 0
        self.row = row
        self.col = col
        self.side = 0

    def __repr__(self):
        return str(self.value)

    def get_index(self):
        return self.row, self.col

    def n8(self):
        aux = [(1,0),(1,1),(-1,0),(-1,-1),(0,1),(0,-1),(1,-1),(-1,1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y+self.col
            if 0 <= nx < self.side and 0 <= ny < self.side:
                ret.append((nx,ny))
        return ret

    def n4(self):
        aux = [(1,0),(-1,0),(0,1),(0,-1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y+self.col
            if 0 <= nx < self.side and 0 <= ny < self.side:
                ret.append((nx,ny))
            ret.append((nx,ny))
        return ret

class Suguru:
    def __init__(self, size):
        if np.sqrt(size) != int(np.sqrt(size)):
            raise ValueError("Size must be a perfect square.")    
        self.size = size
        self.side = np.sqrt(size).astype(int)
        
        
        self.grid = np.array([Cell(i,j) for i in range(self.side) for j in range(self.side)], dtype=object)
        self.grid = self.grid.reshape((self.side, self.side))
        self.regions = np.zeros((self.side, self.side), dtype=int)

        self._gen_regions() 
    
    def solve(self):
        return self._recursive_solver(0,0)

    def _solvable(self):
        while not self._solved():
            print(self.grid)
            if not self._try_fill():
                print(self.grid)
            time.sleep(2)
        print(self.grid)
        print(self.regions)

    def _try_fill(self):
        action = False
        
        # Fill obvious places
        for c1 in self.grid.flatten():
            if c1.value != 0:
                continue
            if len(c1.numbers) == 1:
                c1.value = list(c1.numbers)[0]
                c1.numbers = set()
                action = True

        # Forbidden Neighbour
        for c1 in self.grid.flatten():
            if c1.value != 0:
                continue
            for i,j in c1.n8():
                value = self.grid[i][j].value
                if value in c1.numbers:
                    c1.numbers.remove(value)
                    action = True

        # Exclusion Rule
        for c1 in self.grid.flatten():
            if c1.value != 0:
                continue
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region:
                    continue
                if c2.value != 0 and c2.value in c1.numbers:
                    c1.numbers.remove(c2.value)
                    action = True

        # Hidden Single
        for c1 in self.grid.flatten():
            if c1.value != 0:
                continue
            others = set()
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region:
                    continue
                others = others.union(c2.numbers)
            u = c1.numbers.difference(others)
            if len(u) == 1:
                c1.numbers.difference_update(others)
        
        # Naked Pairs
        for c1 in self.grid.flatten():
            if len(c1.numbers) != 2:
                continue
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region:
                    continue

                if c2.numbers != c1.numbers:
                    continue
                
                for c3 in self.grid.flatten():
                    if c3 == c1 or c3 == c2 or c3.region != c1.region:
                        continue
                    c3.numbers.difference_update(c1.numbers)
                    action = True 
        
        # Hidden Pairs
        for c1 in self.grid.flatten():
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region:
                    continue
                if len(c2.numbers.intersection(c1.numbers)) >= 2:
                    others = set()
                    for c3 in self.grid.flatten():
                        if c3 == c2 or c3 == c1 or c3.region != c1.region:
                            continue
                        others = others.union(c3.numbers)
                    n1 = c1.numbers - others
                    n2 = c2.numbers - others
                    if n1 == n2 and len(n1) == 2:
                        c1.numbers.difference_update(others)
                        c2.numbers.difference_update(others)
                        action = True
                                
        # Naked Triples
        for c1 in self.grid.flatten():
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region:
                    continue
                for c3 in self.grid.flatten():
                    if c3 == c2 or c3 == c1 or c3.region != c1.region:
                        continue
                    numbers = c1.numbers.union(c2.numbers.union(c3.numbers))
                    if len(numbers) == 3 and len(c1.numbers) and len(c2.numbers) and len(c3.numbers):
                        for c4 in self.grid.flatten():
                            if c4 == c1 or c4 == c2 or c4 == c3 or c4.region != c1.region:
                                continue
                            c4.numbers.difference_update(numbers)
                            action = True
                  
        # Hidden Triples
        for c1 in self.grid.flatten():
            if not len(c1.numbers):
                continue
            for c2 in self.grid.flatten():
                if c2 == c1 or c2.region != c1.region or not len(c2.numbers):
                    continue
                for c3 in self.grid.flatten():
                    if c3 == c2 or c3 == c1 or c3.region != c1.region or not len(c3.numbers):
                        continue
                    
                    numbers = c1.numbers.union(c2.numbers.union(c3.numbers))
                    others = set()
                    for c4 in self.grid.flatten():
                        if c4 == c1 or c4 == c2 or c4 == c3 or c4.region != c1.region:
                                continue
                        others = others.union(c4.numbers)
                    if len(numbers.difference(others)) == 3:
                        c1.numbers.difference_update(others)
                        c2.numbers.difference_update(others)
                        c3.numbers.difference_update(others)
                        action = True
             
        # Forbidden Pairs
        for c1 in self.grid.flatten():
            if not len(c1.numbers):
                continue
            for c2 in self.grid.flatten():
                if not len(c2.numbers) or c2 == c1 or c2.region != c1.region or len(c1.numbers.intersection(c2.numbers)) < 2:
                    continue
                others = set()
                for c3 in self.grid.flatten():
                    if not len(c3.numbers) or c3.region != c1.region or c3 == c1 or c3 == c2:
                        continue
                    others = others.union(c3.numbers)
                
                if len(others.difference(c1.numbers.union(c2.numbers))) == 2:
                    matches = others.difference(c1.numbers.union(c2.numbers))
                    for i,j in c1.n8():
                        c4 = self.grid[i][j]
                        c4.numbers.difference_update(matches)
                    for i,j in c2.n8():
                        c4 = self.grid[i][j]
                        c4.numbers.difference_update(matches)

                    action = True
          
        # Forbidden Triples
        for c1 in self.grid.flatten():
            if not len(c1.numbers):
                continue
            for c2 in self.grid.flatten():
                if not len(c2.numbers) or c2 == c1 or c2.region != c1.region:
                    continue
                for c3 in self.grid.flatten():
                    if not len(c3.numbers) or c3 == c1 or c3 == c2 or c3.region != c1.region:
                        continue
                    u2 = set()
                    for c4 in self.grid.flatten():
                        if not len(c4.numbers) or c4.region != c1.region or c4 == c3 or c4 == c2 or c4 == c1:
                            continue
                        u2 = u2.union(c4.numbers)
                    u = c1.numbers.union(c2.numbers.union(c3.numbers))
                    u3 = u2.difference(u)
                    if len(u3) == 3:
                        for i,j in c1.n8()+c2.n8()+c3.n8():
                            nc = self.grid[i][j]
                            nc.numbers.difference_update(u3)
                        action = True
        
        return action
    
    def _recursive_solver(self, i, j):

        if i >= self.side or j >= self.side:
            return True

        c = self.grid[i][j]
        found = True
        for option in c.numbers:
            found = True
            c.value = option

            # Check for neighbour conflicts
            for m,n in c.n8():
                c2 = self.grid[m][n]
                if c2.value == c.value:
                    found = False

            # Check for in-region conflicts
            for c3 in self.grid.flatten():
                if c3 != c and c3.region == c.region:
                    if c3.value == c.value:
                        found = False
            
            # If the current options doesn't work, go to the next option
            if not found:
                continue
            else:
                # Otherwise, recursive call to the next right and next down cells
                if j+1 >= self.side:
                    i += 1
                    j = 0
                    r1 = self._recursive_solver(i,j)
                else:
                    r1 = self._recursive_solver(i, j+1)
                
                if r1:
                    return True
                
        return False



    def _solved(self):
        for cell in self.grid.flatten():
            for i,j in cell.n8():
                if self.grid[i][j].value == cell.value:
                    return False
            for c2 in self.grid.flatten():
                if c2 != cell and c2.region == cell.region and c2.value == cell.value:
                    return False
        return True

    def _get_8_neighbors(self, r, c):
        n = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        ret = list()
        for dr, dc in n:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.side and 0 <= nc < self.side:
                ret.append((nr, nc))
        return ret

    def _gen_regions(self):
        pos = [(r, c) for r in range(self.side) for c in range(self.side)]
        random.shuffle(pos)

        while pos:
            r, c = pos.pop()
            if self.regions[r][c] != 0:
                continue
            region_id = np.max(self.regions) + 1
            region_size = random.randint(1, self.size)
            self.regions[r][c] = region_id
            region_cells = [(r, c)]

            while len(region_cells) < region_size:
                cell = random.choice(region_cells)
                neighbors = self._get_walkable_neighbors(cell[0], cell[1])
                random.shuffle(neighbors)
                before_size = len(region_cells)
                for nr, nc in neighbors:
                    if self.regions[nr][nc] == 0:
                        self.regions[nr][nc] = region_id
                        region_cells.append((nr, nc))
                        break
                after_size = len(region_cells)
                if before_size == after_size:
                    break
            
            # Set cells region
            for i,j in region_cells:
                self.grid[i][j].region = region_id
                self.grid[i][j].side = self.side
                self.grid[i][j].numbers = set(range(1, len(region_cells)+1))
            
            #self._Regions[region_id] = Region(region_id, [self.grid[i][j] for i,j in region_cells])
            
    def _get_walkable_neighbors(self, r, c):
        n = [(0,1), (1,0), (0,-1), (-1,0)]
        ret = list()
        for dr, dc in n:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.side and 0 <= nc < self.side:
                ret.append((nr, nc))
        return ret

    def get_regions(self):
        return self.regions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='The size of the side of the generated matrix')
    parser.add_argument('--cnt', help='Number of matrices to generate')
    parser.add_argument('--prefix', help='Prefix of the filenames generated for the matrices')     
    args = parser.parse_args()

    #size = int(args.size)
    #cnt = int(args.cnt)
    size = 25
    cnt = 1
    prefix = args.prefix
    
    start_time = time.perf_counter()
    for i in range(cnt):
        m = Suguru(size)
        if m.solve():
            suguru_gui.display_suguru(m)
        #conc = np.vstack([matrix, regions])
        #conc.astype('int16').tofile(f'{prefix}_{i}.data')
    end_time = time.perf_counter()
    print(f'Elapsed time: {end_time - start_time} seconds')   

if __name__ == '__main__':
    main()



