import time
import argparse
import numpy as np
import suguru_gui
import threading
import os
import typing
import tkinter as tk

class Cell:
    def __init__(self, row, col, x, y):
        self.numbers = set()
        self.value = 0
        self.region = 0
        self.row = row
        self.col = col
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self.value)

    def get_index(self):
        return self.row, self.col

    def n8(self):
        aux = [(1,0),(1,1),(-1,0),(-1,-1),(0,1),(0,-1),(1,-1),(-1,1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y + self.col
            if 0 <= nx < self.x and 0 <= ny < self.y:
                ret.append((nx,ny))
        return ret

    def n4(self):
        aux = [(1,0),(-1,0),(0,1),(0,-1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y + self.col
            if 0 <= nx < self.x and 0 <= ny < self.y:
                ret.append((nx,ny))
        
        return ret

class DeterministicEngine:
    def __init__(self, rows, cols, grid, regions):
        self.raw_grid = grid
        self.regions = regions
        self.rows = rows
        self.cols = cols
        self.grid = np.array([Cell(i,j,self.rows,self.cols) for i in range(self.rows) for j in range(self.cols)])
        self.grid = self.grid.reshape((self.rows, self.cols))
        self._setup_cells()

        self._Regions: dict[int, list] = dict()
        for c in self.grid.flatten():
            self._Regions.setdefault(c.region, []).append(c)
    
    def _apply_rules(self):
        if self.corrupted():
            return False
        if self._SizeOneRule():
            return True
        if self._GroupExclusionRule():
            return True
        if self._ForbiddenNeighbour():
            return True
        if self._HiddenSingle():
            return True
        if self._NakedPairs():
            return True
        if self._HiddenPairs():
            return True
        if self._NakedTriples():
            return True
        if self._HiddenTriples():
            return True
        if self._ForbiddenPairs():
            return True
        if self._ForbiddenTriples():
            return True
        return False

    def _setup_cells(self):
        region_sizes = dict()
        for i in range(self.rows):
            for j in range(self.cols):
                r = self.regions[i,j]
                self.grid[i,j].region = r
                self.grid[i,j].value = self.raw_grid[i,j]
                if r not in region_sizes:
                    region_sizes[r] = 0
                region_sizes[r] += 1
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.raw_grid[i,j] != 0:
                    continue
                r = self.regions[i,j]
                length = region_sizes[r]
                self.grid[i,j].numbers = set(range(1,length+1))
        
        
        
        



    def corrupted(self):
        for cells in self._Regions.values():
            for c1 in cells:
                if not c1.value:
                    continue
                n1 = [c2.value for c2 in cells if c2 is not c1]
                n2 = [self.grid[i][j].value for i,j in c1.n8()]
                neighbour_values = n1 + n2
                if c1.value in neighbour_values:
                    return True
        return False
                
    def solved(self):
        for cells in self._Regions.values():
            for c1 in cells:
                n1 = [c2.value for c2 in cells if c2 is not c1]
                n2 = [self.grid[i][j].value for i,j in c1.n8()]
                neighbour_values = n1 + n2
                if c1.value in neighbour_values or 0 in neighbour_values:
                    return False
        return True

    def _SizeOneRule(self):
        ret = False
        for cells in self._Regions.values():            
            for c1 in cells:
                c1 = typing.cast(Cell, c1)
                if len(c1.numbers) == 1:
                    value = list(c1.numbers)[0]
                    c1.value = value
                    c1.numbers = set()
                    ret = True
        return ret
    
    def _GroupExclusionRule(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                if c1.value != 0:
                    continue
                gcs = [c2 for c2 in cells if c2 is not c1]
                for c2 in gcs:
                    if c2.value in c1.numbers:
                        c1.numbers.remove(c2.value)
                        ret = True
        return ret
    
    def _ForbiddenNeighbour(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                neighbours = [self.grid[m][n] for m,n in c1.n8()]
                for c2 in neighbours:
                    if c2.value in c1.numbers:
                        c1.numbers.remove(c2.value)
                        ret = True
        return ret

    def _HiddenSingle(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                others = set().union(*(c2.numbers for c2 in cells if c2 is not c1))
                rest = c1.numbers - others
                if len(rest) == 1:
                    c1.numbers.difference_update(others)
                    ret = True
        return ret

    def _NakedPairs(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                if len(c1.numbers) != 2:
                    continue

                gcs = set(c2 for c2 in cells if c2 is not c1)
                matches = set(c2 for c2 in gcs if c2.numbers == c1.numbers)

                if len(matches) == 1:
                    gcs -= matches
                    others = list(matches)[0].numbers
                    for g in gcs:
                        up = g.numbers - others
                        if up != g.numbers:
                            g.numbers.difference_update(others)
                            ret = True
        return ret
    
    def _HiddenPairs(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                for c2 in cells:
                    if c2 is c1:
                        continue
                    others = set().union(*(c3.numbers for c3 in cells if c3 not in [c1, c2]))
                    n1 = c1.numbers - others
                    n2 = c2.numbers - others

                    if n1 == n2 and len(n1) == 2:
                        if n1 != c1.numbers or n2 != c2.numbers:
                            ret = True
                        c1.numbers -= others
                        c2.numbers -= others
        return ret

    def _NakedTriples(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                if not len(c1.numbers):
                    continue
                for c2 in cells:
                    if not len(c2.numbers) or c2 is c1:
                        continue
                    for c3 in cells:
                        if not len(c3.numbers) or c3 in [c1,c2]:
                            continue
                        numbers = set().union(*(c1.numbers, c2.numbers, c3.numbers))
                        if len(numbers) == 3:
                            for c4 in cells:
                                if c4 in [c1,c2,c3]:
                                    continue
                                up = c4.numbers - numbers
                                if up != c4.numbers:
                                    ret = True
                                c4.numbers.difference_update(numbers)
        return ret

    def _HiddenTriples(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                if not len(c1.numbers):
                    continue
                for c2 in cells:
                    if not len(c2.numbers) or c2 is c1:
                        continue
                    for c3 in cells:
                        if not len(c3.numbers) or c3 in [c1,c2]:
                            continue
                        numbers = set().union(*(c1.numbers, c2.numbers, c3.numbers))
                        others = set().union(*(c4.numbers for c4 in cells if c4 not in [c1,c2,c3]))
                        
                        if len(numbers - others) == 3:
                            u1 = c1.numbers - others
                            u2 = c2.numbers - others
                            u3 = c3.numbers - others

                            if u1 != c1.numbers or u2 != c2.numbers or u3 != c3.numbers:
                                ret = True
                            
                            c1.numbers = u1
                            c2.numbers = u2
                            c3.numbers = u3
        return ret
    
    def _ForbiddenPairs(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                for c2 in cells:
                    if c2 is c1 or len(c1.numbers.intersection(c2.numbers)) < 2:
                        continue
                    others = set().union(*(c3.numbers for c3 in cells if c3 not in [c1,c2]))
                    numbers = c1.numbers.union(c2.numbers)

                    if len(numbers - others) == 2:
                        neighbours1 = set(self.grid[m][n] for m,n in c1.n8())
                        neighbours2 = set(self.grid[m][n] for m,n in c2.n8())
                        neighbours = neighbours1.intersection(neighbours2)

                        for n in neighbours:
                            up = n.numbers - numbers
                            if up != n.numbers:
                                ret = True
                            n.numbers = up
        return ret
                        
    def _ForbiddenTriples(self):
        ret = False
        for cells in self._Regions.values():
            for c1 in cells:
                for c2 in cells:
                    if c1 is c2:
                        continue
                    for c3 in cells:
                        if c3 in [c1,c2]:
                            continue
                        numbers = set().union(*(c1.numbers, c2.numbers, c3.numbers))
                        if len(numbers) == 3:
                            n1 = set(self.grid[i][j] for i,j in c1.n8())
                            n2 = set(self.grid[i][j] for i,j in c2.n8())
                            n3 = set(self.grid[i][j] for i,j in c3.n8())
                            neighbours = n1.intersection(n2.intersection(n3))
                            for n in neighbours:
                                up = n.numbers - numbers
                                if up != n.numbers:
                                    ret = True
                                n.numbers = up
        return ret

class Suguru:
    def __init__(self, file_path, tips=0):
        if not os.path.isfile(file_path):
            raise Exception('Error: invalid file path')
        
        # Load from file
        self.file_path = file_path
        with open(self.file_path, 'rb') as fp:
            rows = int.from_bytes(fp.read(2))
            cols = int.from_bytes(fp.read(2))
            arr = np.fromfile(fp, dtype=np.int16).reshape(2, rows, cols)

        # Organize
        self.solved, self.regions = arr
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.size = self.rows*self.cols

        # Random number of tips
        pos = [(i,j) for i in range(self.rows) for j in range(self.cols)]
        if tips == 0:
            tips = np.random.randint(np.floor(np.sqrt(self.size)), np.ceil(2*np.sqrt(self.size)))
            np.random.shuffle(pos)
            for p in range(tips):
                i,j = pos[p]
                self.grid[i,j] = self.solved[i,j]
        else:
            tips = int(np.floor(tips*self.size))
            np.random.shuffle(pos)
            for p in range(tips):
                i,j = pos[p]
                self.grid[i,j] = self.solved[i,j]

        # Graphical control
        self.root_window = tk.Tk()
        self.gui = suguru_gui.SuguruGUI(self.root_window, self.rows, self.cols, self.grid, self.regions, cell_size=80)

        # Deterministic Engine
        self.de = DeterministicEngine(self.rows, self.cols, self.grid, self.regions)

    def show(self):
        thread = threading.Thread(target=self._update_grid_periodically, args=(self.gui, self), daemon=True)
        thread.start()
        self.root_window.mainloop()

    @staticmethod
    def _update_grid_periodically(gui, suguru):
        while True:

            # Solver
            if suguru.de._apply_rules():
                suguru.grid = suguru.de.grid
            else:
                print('nothing done')

            # Schedule the GUI update on the main thread
            gui.root.after(0, gui.set_grid, suguru.grid)
            time.sleep(0.2)  # update every second

def main():
    s = Suguru('./samples/10x10_1.data', 0.5)
    s.show()


if __name__ == '__main__':
    main()