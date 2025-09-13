import time
import argparse
import numpy as np
import os

class SuguruGenerator:
    def __init__(self, m, n):
        self.m = m
        self.n = n

        self.grid = np.zeros((self.m, self.n), dtype=int)
        self.regions = np.zeros((self.m, self.n), dtype=int)
        self.solved = np.zeros((self.m, self.n), dtype=int)

        self.region_map = dict()
        self.region_sizes = dict()

    def generate(self, unique=False):
        self._reset()

        region = 1
        pos = [(i,j) for i in range(self.m) for j in range(self.n)]
        while pos:
            init_pos = pos.pop()
            region_size = np.random.randint(2, 6)
            cnt = 0
            value = 1

            path = set()
            path.add(init_pos)
            
            while cnt < region_size and path:
                    
                p = path.pop()
                i,j = p

                n8v = [self.grid[x][y] for x,y in self._n8(i,j)]

                if value in n8v:
                    continue

                # Setup tile
                self.grid[i][j] = value
                self.regions[i][j] = region
                
                if p in pos:
                    pos.remove(p)

                if region not in self.region_map:
                    self.region_map[region] = list()
                self.region_map[region].append(p)

                neighbours = [n for n in self._n4(i,j) if self.grid[n[0]][n[1]] == 0]
                path = path.union(neighbours)
                path = list(path)
                np.random.shuffle(path)
                path = set(path)
                value += 1
                cnt += 1
                
            region += 1

        
        # Region expansion
        while np.any(self.regions == 0):
            pos = [(i,j) for i in range(self.m) for j in range(self.n)]
            zeros = np.count_nonzero(self.regions == 0)
            zeros_after = zeros
            while pos:
                p = pos.pop()
                k,l = p

                if self.regions[k][l] != 0:
                    continue
                
                n8tv = [self.grid[i][j] for i,j in self._n8(k,l)]
                n8tv = [n8t for n8t in n8tv if n8t != 0]

                n4r = [self.regions[i][j] for i,j in self._n4(k,l)]
                n4r = [r for r in n4r if r != 0]
                n4r = sorted(n4r, key=lambda r : len(self.region_map[r]))

                for r in n4r:
                    length = len(self.region_map[r])
                    value = length+1

                    # For string format
                    _n8 = self._n8(k,l)
                    _n8 = [t for t in _n8 if self.regions[t] == region]
                    if len(_n8) >= 2:
                        continue
                    if value not in n8tv:
                        self.regions[k][l] = r
                        self.grid[k][l] = value
                        self.region_map[r].append(p)
                        zeros_after -= 1
                        break     
            if zeros == zeros_after:
                break

               
        solved = self._solved()
        if not solved:
            return False
        
        # Save solved
        self.solved = self.grid.copy()

        # Uniqueness
        if unique:
            for x in range(self.m):
                for y in range(self.n):
                    v = self.grid[x][y]
                    self.grid[x][y] = 0
                    if self._count_solutions(0,0) != 1:
                        self.grid[x][y] = v 
        return solved

    def _count_solutions(self, i, j):
        # Base case: We've filled the entire grid. We found a valid solution.
        if i >= self.m:
            return 1

        # Get the next cell to fill.
        next_i, next_j = (i, j + 1) if j + 1 < self.n else (i + 1, 0)

        # If the current cell is pre-filled, skip it and continue the search.
        if self.grid[i, j] != 0:
            return self._count_solutions(next_i, next_j)

        # Calculate the set of available numbers for the current empty cell.
        r = self.regions[i][j]
        bruv_tiles = set(self.region_map[r])
        reg_values = {self.grid[x, y] for x, y in bruv_tiles if (x, y) != (i, j)}
        ngh_values = {self.grid[x, y] for x, y in self._n8(i, j)}
        used = ngh_values.union(reg_values)
        available = set(range(1, len(bruv_tiles) + 1)) - used

        total_solutions = 0
        # Try each available number.
        for value in available:
            self.grid[i, j] = value
            total_solutions += self._count_solutions(next_i, next_j)
            self.grid[i, j] = 0  # Backtrack!

            # Optimization: Stop as soon as you find more than one solution.
            if total_solutions > 1:
                return 2  # A value greater than 1 to signal multiple solutions.

        return total_solutions

    def _swap_tiles(self, t1, t2):
        i,j = t1
        k,l = t2
        self.grid[i,j], self.grid[k,l] = self.grid[k,l], self.grid[i,j]

    def _tile_dist(self, t1, t2):
        i,j = t1
        k,l = t2

        return np.sqrt(np.square(i-k)+np.square(j-l))

    def _tile_vh_dist(self, t1, t2):
        i,j = t1
        k,l = t2

        return np.abs(i-k), np.abs(j-l)

    def _reset(self):
        self.grid = np.zeros((self.m, self.n), dtype=int)
        self.regions = np.zeros((self.m, self.n), dtype=int)
        self.region_map = dict()

    def _n8(self, i, j):
        moves = [(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)]
        n8p = list()
        for move in moves:
            mx, my = i+move[0], j+move[1]
            if 0 <= mx < self.m and 0 <= my < self.n:
                n8p.append((mx,my))
        return n8p
             
    def _n4(self, i, j, k=1):
        moves = [(k,0),(0,k),(-k,0),(0,-k)]
        n4p = list()
        for move in moves:
            mx, my = i+move[0], j+move[1]
            if 0 <= mx < self.m and 0 <= my < self.n:
                n4p.append((mx,my))
        return n4p
    
    def _solved(self):
        if np.any(self.regions == 0) or np.any(self.grid == 0):
            return False
    
        # Group permutation check and valid numbers
        for _, pos in self.region_map.items():
            length = len(pos)
            values = set(range(1, length+1))
            tile_values = set([self.grid[i][j] for i,j in pos])
            if tile_values != values:
                return False

        # 8 Neighbours check
        for _, pos in self.region_map.items():
            for p in pos:
                k,l = p
                neighbour_values = [self.grid[i][j] for i,j in self._n8(k,l)]
                if self.grid[k][l] in neighbour_values:
                    return False

        return True
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', help='Number of rows')
    parser.add_argument('--cols', help='Number of columns')
    parser.add_argument('--count', help='Number of matrices to generate')
    parser.add_argument('--unique', help='If the puzzles need to be unique', action='store_true')
    parser.add_argument('--output', help='Output path')
    args = parser.parse_args()

    rows = int(args.rows)
    cols = int(args.cols)
    cnt = int(args.count)
    output_path = args.output
    unique = args.unique

    created = 0
    start_time = time.perf_counter()
    m = SuguruGenerator(rows, cols)
    while created < cnt:
        if m.generate(unique):
            created += 1
            conc = np.vstack([m.grid, m.solved, m.regions])
            path = os.path.join(output_path, f'{rows}x{cols}_{created}.data')
            with open(path, 'wb') as fp:
                fp.write(rows.to_bytes(2))
                fp.write(cols.to_bytes(2))
                conc.astype('int16').tofile(fp)
    end_time = time.perf_counter()
    print(f'Elapsed time: {end_time - start_time} seconds')  

if __name__ == '__main__':
    main()



