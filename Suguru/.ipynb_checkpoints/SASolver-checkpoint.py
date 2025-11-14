import math
import numpy as np
import time
import SuguruSolvers
import argparse
import os
import itertools

class SASolver(SuguruSolvers.BaseSolver):
    def __init__(self, grid, regions, delay=0, debug=False):
        super().__init__(grid, regions, delay)

        self.debug = debug
        
        self.changes = {
            'shuffle': None,
            'swap': None
        }
    
        self.diff_cost = 0
        self.neighbours = dict({(i, j):self._n8(i, j) for (i, j), _ in np.ndenumerate(self.grid)})
        self.regions_map = dict({r:list(map(tuple, np.argwhere(self.regions == r))) for r in np.unique(self.regions)})
        self.regions_lengths = dict({r:len(self.regions_map[r]) for r in np.unique(self.regions)})
        self.candidates_mapping = dict()


    def solve(self):
        # Preprocessing with deterministic engine
        if self.debug:
            print('[+] Preprocessing with Deterministic Engine...')
            
        de = SuguruSolvers.DeterministicEngine(self.grid, self.regions)
        while de._apply_rules():
            time.sleep(self.delay)
            de._update_main_grid()
            self.grid = de.grid

        # We shouldn't modify deterministically solved tiles
        self.tips = list(map(tuple, np.argwhere(self.grid != 0)))

        # Update candidates for each tile
        for region, tiles in self.regions_map.items():
            length = len(tiles)
            used = list([self.grid[i, j] for i, j in tiles if self.grid[i, j] != 0])

            for (i, j) in tiles:
                value = self.grid[i, j]
                if value == 0:
                    self.candidates_mapping[(i,j)] = set(list(range(1, length + 1))) - set(used)


        # Call to main algorithm
        self._simulated_annealing()

        return SuguruSolvers.Checker.solved(self.grid, self.regions)


    def _simulated_annealing(self) -> None:

        # Fill with random numbers
        n = np.count_nonzero(self.grid == 0)
        for region, tiles in self.regions_map.items():
            used = list()
            for (i, j) in tiles:
                if self.grid[i, j]:
                    used.append(self.grid[i, j])
                    continue

                candidates = list(self.candidates_mapping[(i, j)])
                np.random.shuffle(candidates)
                for c in candidates:
                    if c not in used:
                        used.append(c)
                        self.grid[i, j] = c
                        break



        self.live_cost = self._get_current_cost()
        cost = self.live_cost
        temperature = 1
        min_temperature = 0.1
        cooling_rate = 0.999

        iterations_per_t = math.ceil(np.sqrt(n))
        count = 0
        accepted = 0
        p = 0.05
        resets = 0
        while cost > 0:
            for i in range(iterations_per_t):
                if resets > 3:
                    cost = -1
                    break
                if temperature < min_temperature:
                    temperature *= 7
                    resets += 1
                    break
                self._do_change(p)

                new_cost = cost + self.diff_cost
                delta = new_cost - cost

                if delta <= 0 or np.random.rand() < np.exp(-delta/temperature):
                    cost = new_cost
                    accepted += 1
                else:
                    self._undo_change()

                count += 1
                if cost <= 0:
                    break

                time.sleep(self.delay)
            if self.debug:
                print(cost, temperature)
            temperature *= cooling_rate


    def _do_change(self, prob_shuffle) -> None:
        self.changes['shuffle'] = None
        self.changes['swap'] = None
        r1 = self._get_random_invalid_region()
        r2 = self._get_random_region()

        if np.random.rand() < 0.1:
            invalid_region = r2
        else:
            invalid_region = r1
        all_positions = self.regions_map[invalid_region]

        positions = [(i, j) for i, j in all_positions if (i, j) not in self.tips]

        before_cost = self._calculate_bound_cost(all_positions)
        np.random.shuffle(positions)
        if np.random.rand() < 2:
            original = self._shuffle_region(invalid_region)
            self.changes['shuffle'] = original
            self.changes['swap'] = None
        else:
            tiles = self._swap_two_random_tiles(invalid_region)
            if tiles is not None:
                t1, t2 = tiles
                self.changes['swap'] = [(t1[0], t1[1]), (t2[0], t2[1])]
                self.changes['shuffle'] = None
            else:
                original = self._shuffle_region(invalid_region)
                self.changes['shuffle'] = original
                self.changes['swap'] = None
        
        after_cost = self._calculate_bound_cost(all_positions)
        self.diff_cost = after_cost - before_cost
                    
    def _undo_change(self):
        if self.changes['shuffle'] is not None:
            original = self.changes['shuffle']
            for (i, j), v in original.items():
                self.grid[i, j] = v
        elif self.changes['swap'] is not None:
            t1, t2 = self.changes['swap']
            self._swap_tiles(t1, t2)

    def _calculate_bound_cost(self, positions) -> int:
        # Extract rows and columns
        rows = [i for i, _ in positions]
        cols = [j for _, j in positions]

        # Before cost
        top_left = (max(min(rows) - 1, 0), max(min(cols) - 1, 0))
        bottom_right = (min(max(rows) + 1, self.rows - 1), min(max(cols) + 1, self.cols - 1))

        cost = 0
        for i in range(top_left[0], bottom_right[0] + 1):
            for j in range(top_left[1], bottom_right[1] + 1):
                n8_values = self._n8_values(i, j)
                if self.grid[i, j] in n8_values:
                    cost += 1
        return cost

    def _get_random_invalid_region(self):
        regions = list(self.regions_map.keys())
        np.random.shuffle(regions)
        for r in regions:
            length = self.regions_lengths[r]
            if length < 2:
                continue
            tiles = self.regions_map[r]
            for t in tiles:
                if self._conflict_count(t) > 0:
                    return r
        return None

    def _get_random_region(self):
        regions = list(self.regions_map.keys())
        np.random.shuffle(regions)
        for r in regions:
            length = len(self.regions_map[r])
            if length < 2:
                continue
            return r
        return None

    def _swap_two_random_tiles(self, region):
        tiles = self.regions_map[region]
        tiles = [t for t in tiles if t not in self.tips]
        for t1, t2 in itertools.combinations(tiles, 2):
            i, j = t1
            m, n = t2
            v1 = self.grid[i, j]
            v2 = self.grid[m, n]
            if v1 in self.candidates_mapping[m, n] and v2 in self.candidates_mapping[i, j]:
                self._swap_tiles(t1, t2)
                return t1, t2

    def _shuffle_region(self, region) -> dict:
        # Get positions excluding tips
        positions = self.regions_map[region]

        used = list([self.grid[i, j] for i, j in positions if (i, j) in self.tips])

        positions = [(i, j) for i, j in positions if (i, j) not in self.tips]
        original = {(i, j):self.grid[i, j] for i, j in positions}

        for (i, j) in positions:
            candidates = list(self.candidates_mapping[(i, j)])
            np.random.shuffle(candidates)
            for c in candidates:
                if c not in used:
                    used.append(c)
                    self.grid[i, j] = c
                    break

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

    start = time.perf_counter()
    solver = SASolver(grid, regions)
    solved = solver.solve()
    end = time.perf_counter()
    elapsed = end - start
    print('Elapsed: ', elapsed)
    if solved:
        print(solver.grid)
        print('[+] Solved.')

if __name__ == '__main__':
    main()