import numpy as np
import time
import SuguruSolvers
import argparse
import os
import itertools

class SASolver(SuguruSolvers.BaseSolver):
    def __init__(self, grid, regions, delay=0):
        super().__init__(grid, regions, delay)

        self.changes = {
            'shuffle': None,
            'swap': None
        }

        self.tips = list(map(tuple, np.argwhere(self.grid != 0)))
    
        self.diff_cost = 0
        self.neighbours = dict({(i, j):self._n8(i, j) for (i, j), _ in np.ndenumerate(self.grid)})
        self.regions_map = dict({r:list(map(tuple, np.argwhere(self.regions == r))) for r in np.unique(self.regions)})
        self.regions_lengths = dict({r:len(self.regions_map[r]) for r in np.unique(self.regions)})

    def solve(self):
        # Preprocessing.
        #de = SuguruSolvers.DeterministicEngine(self.grid, self.regions)
        #while de._apply_rules():
        #    continue
        #de._update_main_grid()
        #self.grid = de.grid

        start = time.perf_counter()
        self._SA()
        elapsed = time.perf_counter() - start
        print('Elapsed: ', elapsed)

        return SuguruSolvers.Checker.solved(self.grid, self.regions)


    def _SA(self):

        # Fill with random numbers
        n = np.count_nonzero(self.grid == 0)
        null_positions = np.argwhere(self.grid == 0)
        for i, j in null_positions:
            region = self.regions[i, j]
            region_length = self.regions_lengths[region]
            region_values = self._region_values(region)

            numbers = set(list(range(1, region_length + 1)))
            numbers -= region_values
            available = list(numbers - set(self._n8_values(i, j)))
            numbers = list(numbers)

            if available:
                self.grid[i, j] = np.random.choice(available)
            else:
                if len(numbers):
                    self.grid[i, j] = np.random.choice(numbers)


        self.live_cost = self._get_current_cost()
        cost = self.live_cost
        temperature = 1
        min_temperature = 5e-1
        cooling_rate = 0.999

        #iterations_pert_t = 5*int(np.sqrt(n))
        iterations_pert_t = int(np.sqrt(n))

        count = 0
        while cost > 0:
            for i in range(iterations_pert_t):
                if temperature < min_temperature:
                    temperature *= 1.2
                self._do_change()

                new_cost = cost + self.diff_cost
                delta = new_cost - cost

                if delta <= 0 or np.random.rand() < np.exp(-delta/temperature):
                    cost = new_cost
                    count = 0
                else:
                    self._undo_change()
                    count += 1
                
                if cost <= 0:
                    break

                time.sleep(self.delay)

            print(cost, temperature)

            temperature *= cooling_rate


    def _get_random_pair_in_region(self):
        regions = list(np.unique(self.regions))
        np.random.shuffle(regions)

        for r in regions:
            tiles = self.regions_map[r]
            np.random.shuffle(tiles)
            if len(tiles) < 2:
                continue
            return tiles.pop(), tiles.pop()

    def _do_change(self):
        self.changes['shuffle'] = None
        self.changes['swap'] = None
        region = self._get_random_invalid_region()
        #region = np.random.choice(np.unique(self.regions))
        length = self.regions_lengths[region]
        positions = list()
        while not positions:
            region = self._get_random_invalid_region()
            #region = np.random.choice(np.unique(self.regions))
            length = self.regions_lengths[region]

            if length < 2:
                continue
        
            # Get positions within the random selected region
            positions = self.regions_map[region]

            # Do not change fixed tiles
            positions = [(i, j) for i, j in positions if (i, j) not in self.tips]

        # Extract rows and columns
        rows = [i for i, _ in positions]
        cols = [j for _, j in positions]


        # Bottom-right corner

        # Before cost
        top_left = (max(min(rows)-1, 0), max(min(cols)-1, 0))
        bottom_right = (min(max(rows)+1, self.rows-1), min(max(cols)+1, self.cols-1))

        before_cost = 0
        for i in range(top_left[0], bottom_right[0]+1):
            for j in range(top_left[1], bottom_right[1]+1):
                n8_values = self._n8_values(i, j)
                if self.grid[i,j] in n8_values:
                    before_cost += 1

        if np.random.rand() < 0.05:
            numbers = [self.grid[i, j] for i, j in positions]
            np.random.shuffle(numbers)
            original = {(i, j):self.grid[i, j] for i, j in positions}
            self.changes['shuffle'] = original
            for k, (i, j) in enumerate(original.keys()):
                self.grid[i, j] = numbers[k]
            self.changes['swap'] = None
        else:
            np.random.shuffle(positions)
            if len(positions) > 2:
                for t1, t2 in itertools.combinations(positions, 2):
                    self._swap_tiles(t1, t2)
                    self.changes['swap'] = [(t1[0], t1[1]), (t2[0], t2[1])]
                    self.changes['shuffle'] = None
                    break
        
        after_cost = 0
        for i in range(top_left[0], bottom_right[0]+1):
            for j in range(top_left[1], bottom_right[1]+1):
                n8_values = self._n8_values(i, j)
                if self.grid[i,j] in n8_values:
                    after_cost += 1
        
        self.diff_cost = after_cost - before_cost
                    
    def _undo_change(self):
        if self.changes['shuffle'] is not None:
            original = self.changes['shuffle']
            for (i, j), v in original.items():
                self.grid[i, j] = v
        elif self.changes['swap'] is not None:
            t1, t2 = self.changes['swap']
            self._swap_tiles(t1, t2)

    def _get_random_invalid_region(self):
        regions = np.unique(self.regions)
        np.random.shuffle(regions)
        for r in regions:
            tiles = self.regions_map[r]
            for t in tiles:
                if self._conflict_count(t) > 0:
                    return r
        return None

    def _shuffle_region(self, region):
        positions = self.regions_map[region]
        length = self.regions_lengths[region]

        original = {(i, j):self.grid[i, j] for i, j in positions}
        numbers = list(range(1, length + 1))
        np.random.shuffle(numbers)
        new_dist = {(i, j):numbers[k] for k, (i, j) in enumerate(positions)}

        for (i, j), value in new_dist.items():
            self.grid[i, j] = value

        return original
    
    def _unshuffle_region(self, original):
        for (i, j), value in original.items():
            self.grid[i, j] = value

    def _swap_tiles(self, t1, t2):
        i, j = t1
        m, n = t2
        self.grid[i, j], self.grid[m, n] = self.grid[m, n], self.grid[i, j]          
              
    def _conflict_count(self, tile):
        i, j = tile
        value = self.grid[i, j]
        n_values = np.array(self._n8_values(i, j))
        return np.count_nonzero(n_values == value)

    def _get_current_cost(self):
        cost = 0
        for (i, j), value in np.ndenumerate(self.grid):
            n8_values = np.array(self._n8_values(i, j))
            if value in n8_values:
                cost += 1
        return cost

    def _n8(self, i, j):
        moves = [(1,0),(0,1),(1,1),(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1)]
        n8s = list()
        for m, n in moves:
            nx = i + m
            ny = j + n
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                n8s.append((nx, ny))
        return n8s

    def _n8_values(self, i, j):
        return list(set([self.grid[m, n] for m, n in self.neighbours[(i, j)]]))

    def _region_values(self, region_id):
        values = set()
        positions = self.regions_map[region_id]
        for i, j in positions:
            if self.grid[i, j] != 0:
                values.add(self.grid[i, j])
        return values
    



def parse_suguru_binary(path):
    if not os.path.isfile(path):
        raise Exception('Error: invalid file path')

    # Load from file
    with open(path, 'rb') as fp:
        rows = int.from_bytes(fp.read(2))
        cols = int.from_bytes(fp.read(2))
        arr = np.fromfile(fp, dtype=np.int16).reshape(3, rows, cols)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to Suguru file')
    args = parser.parse_args()
    path = args.path

    if not os.path.isfile(path):
        print('[-] Invalid file')
        return

    grid, solution, regions = parse_suguru_binary(
        path
    )

    solver = SASolver(grid, regions)
    solved = solver.solve()
    if solved:
        print(solver.grid)
        print('[+] Solved.')

if __name__ == '__main__':
    main()