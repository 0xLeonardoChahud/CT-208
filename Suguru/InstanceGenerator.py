import time
import argparse
import numpy as np
import os
import itertools
import SuguruSolvers

class SuguruGenerator:
    def __init__(self, rows, cols, p_remove_tiles=0.75, max_region_length=9):
        self.rows = rows
        self.cols = cols

        self.grid = np.zeros((self.rows, cols), dtype=np.int8)
        self.regions = np.zeros((self.rows, cols), dtype=np.int16)
        self.solved = np.zeros((self.rows, cols), dtype=np.int8)

        self._tile_remove_prob = p_remove_tiles
        self._max_region_length = max_region_length


    def generate(self, unique=False):
        # Reset grid of values and regions.
        self._reset()

        # Randomly generate regions.
        self._random_region_creation()

        # Randomly expand regions always minimizing the length of them.
        self._perform_region_expansion2()

        # Randomize to break creation bias (we generate always assigning values sequentially)
        self._randomize_tiles()

        # Get a boolean value telling if the grid is solved.
        b_solved = SuguruSolvers.Checker.solved(self.grid, self.regions)

        # If its not solvable, return False.
        # This is easier than trying to solve the problems in the grid.
        if not b_solved:
            return False
        
        # Store solved grid
        self.solved = self.grid.copy()
        
        print('[+] Found instance')
        # Uniqueness
        if unique:
            print('[+] Making it unique...')
            self._make_it_unique()
        else:
            print('[+] Selecting tips...')
            print('[+] Removing tiles with {} of probability'.format(self._tile_remove_prob))
            for (i, j), _ in np.ndenumerate(self.grid):
                if np.random.rand() < self._tile_remove_prob:
                    self.grid[i, j] = 0
                    
        return b_solved

    def _random_region_creation(self):
        positions = list(map(tuple, np.argwhere(self.regions == 0)))

        while positions:
            x, y = positions.pop()
            moves = set()
            moves.add((x, y))

            region_size = np.random.randint(2, 6)
            region_id = np.max(self.regions) + 1
            count = 0
            value = 1

            while count < region_size and moves:
                i, j = moves.pop()

                n8v = self._n8_values(i, j)

                if value not in n8v:
                    # Setup tile
                    self.grid[i, j] = value
                    self.regions[i, j] = region_id

                    if (i, j) in positions:
                        positions.remove((i, j))
                    neighbours = [(x, y) for x, y in self._n4(i, j) if self.regions[x, y] == 0]
                    tmp = list(moves.union(neighbours))
                    np.random.shuffle(tmp)
                    moves = set(tmp)
                    
                    value += 1
                    count += 1


    def _perform_region_expansion2(self):

        while np.any(self.regions == 0):
            positions = list(map(tuple, np.argwhere(self.regions == 0)))
            change = False

            for p in positions:
                i, j = p
                n4r = [self.regions[x, y] for x, y in self._n4(i, j)]
                n4r = [r for r in n4r if r != 0]
                n4r = sorted(n4r, key=lambda r: np.count_nonzero(self.regions == r))

                n8_values = self._n8_values(i, j)

                for region in n4r:
                    length = np.count_nonzero(self.regions == region)
                    value = length + 1
                    if value > self._max_region_length:
                        continue
                    if value not in n8_values:
                        self.regions[i, j] = region
                        self.grid[i, j] = value
                        change = True
                        break
            if not change:
                break
            
    def _randomize_tiles(self):
        regions = np.unique(self.regions)
        for r in regions:
            polynomio = list(map(tuple, np.argwhere(self.regions == r)))
            for t1,t2 in itertools.combinations(polynomio, 2):
                if self._tile_dist(t1, t2) < 2:
                    continue
                
                i, j = t1
                m, n = t2
                v1 = self.grid[i, j]
                v2 = self.grid[m, n]

                t1vs = self._n8_values(i, j)
                t2vs = self._n8_values(m, n)

                if v1 in t2vs or v2 in t1vs:
                    continue
                else:
                    self._swap_tiles(t1, t2)

    def _make_it_unique(self):
        for (i, j), value in np.ndenumerate(self.grid):
            self.grid[i, j] = 0
            if self._count_solutions(0, 0) != 1:
                self.grid[i, j] = value

    def _count_solutions(self, i, j):
        # Base case: We've filled the entire grid. We found a valid solution.
        if i >= self.rows:
            return 1

        # Get the next cell to fill.
        next_i, next_j = (i, j + 1) if j + 1 < self.cols else (i + 1, 0)

        # If the current cell is pre-filled, skip it and continue the search.
        if self.grid[i, j] != 0:
            return self._count_solutions(next_i, next_j)

        # Calculate the set of available numbers for the current empty cell.
        r = self.regions[i, j]
        length = np.count_nonzero(self.regions == r)
        bruv_tiles = set(list(map(tuple, np.argwhere(self.regions == r))))
        reg_values = set([self.grid[x, y] for x, y in bruv_tiles if (x, y) != (i, j)])
        ngh_values = set([self.grid[x, y] for x, y in self._n8(i, j)])
        used = ngh_values.union(reg_values)
        available = set(range(1, length + 1)) - used

        total_solutions = 0
        # Try each available number.
        for value in available:
            self.grid[i, j] = value
            total_solutions += self._count_solutions(next_i, next_j)
            self.grid[i, j] = 0  # Backtrack

            # Optimization: Stop as soon as you find more than one solution.
            if total_solutions > 1:
                return 2

        return total_solutions

    def _swap_tiles(self, t1, t2):
        i, j = t1
        x, y = t2
        self.grid[i, j], self.grid[x, y] = self.grid[x, y], self.grid[i, j]

    def _tile_dist(self, t1, t2):
        i, j = t1
        x, y = t2

        return np.sqrt(np.square(i - x)+np.square(j - y))

    def _reset(self):
        self.grid[:] = 0
        self.regions[:] = 0
        self.region_map = dict()

    def _n8_values(self, i, j):
        return [self.grid[x, y] for x, y in self._n8(i, j) if self.grid[x, y] != 0]

    def _n8(self, i, j):
        moves = [
            (1, 1), (1, 0), (0, 1), (-1, -1),
            (-1, 0), (0, -1), (-1, 1), (1, -1)
        ]
        n8p = list()
        for move in moves:
            mx, my = i + move[0], j + move[1]
            if 0 <= mx < self.rows and 0 <= my < self.cols:
                n8p.append((mx, my))
        return n8p

    def _n4(self, i, j):
        moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        n4p = list()
        for move in moves:
            mx, my = i + move[0], j + move[1]
            if 0 <= mx < self.rows and 0 <= my < self.cols:
                n4p.append((mx, my))
        return n4p


        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', help='Number of rows')
    parser.add_argument('--cols', help='Number of columns')
    parser.add_argument('--count', help='Number of matrices to generate')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--tile-remove-prob', help='Probability of a tile being removed from the solved instance', default=0.85)
    parser.add_argument('--max-region-length', help='Maximum length of regions', default=9)
    parser.add_argument('--unique',
                        help='If the puzzles need to be unique',
                        action='store_true'
                        )

    args = parser.parse_args()

    rows = int(args.rows)
    cols = int(args.cols)
    cnt = int(args.count)
    output_path = args.output
    unique = args.unique
    tile_remove_prob = float(args.tile_remove_prob)
    max_region_length = int(args.max_region_length)

    created = 0
    start_time = time.perf_counter()
    m = SuguruGenerator(rows, cols, tile_remove_prob, max_region_length)
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
