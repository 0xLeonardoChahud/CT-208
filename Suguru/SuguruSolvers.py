import itertools
import time
import argparse
import numpy as np
from abc import abstractmethod, ABC

class Tile:
    def __init__(self, row, col, grid, value, region, polynomio):
        self.row = row
        self.col = col
        self.value = value
        self.region = region
        self.polynomio = polynomio
        self.grid = grid
        self.height, self.width = self.grid.shape
        self.candidates = set()

    def _set_polynomio(self, polynomio):
        self.polynomio = polynomio
        if self.value == 0:
            self.candidates = set(range(1, len(self.polynomio) + 1))

    def n8(self):
        moves = [(1, 0), (-1, 0), (-1, 1), (-1, -1),
                 (0, 1), (0, -1), (1, 1), (1, -1)
                 ]
        coords = [(self.row+i, self.col+j) for i, j in moves]
        coords = [(c1, c2) for c1, c2 in coords if 0 <= c1 < self.height and 0 <= c2 < self.width]
        n8_tiles = [self.grid[i, j] for i, j in coords]

        return n8_tiles

    def n4(self):
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        coords = [(self.row+i, self.col+j) for i, j in moves]
        coords = [(c1, c2) for c1, c2 in coords if 0 <= c1 < self.height and 0 <= c2 < self.width]
        n4_tiles = [self.grid[i, j] for i, j in coords]

        return n4_tiles

    def is_consistent(self):
        if self.value == 0:
            return False

        if self.value not in set(range(1, len(self.polynomio) + 1)):
            return False

        n8_values = [t.value for t in self.n8()]
        if self.value in n8_values:
            return False

        polynomio_values = [t.value for t in self.polynomio if t != self]
        if self.value in polynomio_values:
            return False
        return True


class BaseSolver(ABC):
    def __init__(self, grid, regions, delay=0):
        self.grid = grid.copy()
        self.regions = regions.copy()
        self.rows, self.cols = self.grid.shape

        self.tile_grid = np.full((self.rows, self.cols), None, dtype=Tile)
        self.polynomios_dic = dict()

        self.delay = delay

        self._setup_polynomios()

    def _setup_polynomios(self):
        # Polynomios dict do not exist yet. So build them first and pass None
        for i in range(self.rows):
            for j in range(self.cols):
                r = self.regions[i, j]
                v = self.grid[i, j]
                if r not in self.polynomios_dic:
                    self.polynomios_dic[r] = list()
                new_tile = Tile(i, j, self.tile_grid, v, r, None)
                self.tile_grid[i, j] = new_tile
                self.polynomios_dic[r].append(new_tile)

        # Update tiles so each one knows its own polynomio
        for region, polynomio in self.polynomios_dic.items():
            for tile in polynomio:
                tile._set_polynomio(polynomio)

    def _solved(self):
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                return False
        return True


    def _update_main_grid(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.grid[i, j] = self.tile_grid[i, j].value

    def _solved_polynomio(self, polynomio):
        for tile in polynomio:
            if not tile.is_consistent():
                return False
        return True

    @abstractmethod
    def solve(self):
        pass


class DeterministicEngine(BaseSolver):
    def __init__(self, grid, regions, delay=0):
        super().__init__(grid, regions, delay)

    def solve(self):
        while self._apply_rules():
            time.sleep(self.delay)
            continue
        self._update_main_grid()
        return Checker.solved(self.grid, self.regions)

    def _apply_rules(self):
        if self._BaseCaseRule():
            return True
        if self._ExclusionRule():
            return True
        if self._ForbiddenNeighbour():
            return True
        if self._HiddenSingle():
            return True
        if self._NakedPair():
            return True
        if self._HiddenPairs():
            return True
        if self._NakedTriples():
            return True
        if self._HiddenTriples():
            return True
        if self._ForbiddenPairs():
            return True
        if self._ForbiddenTriple():
            return True

        return False

    def _BaseCaseRule(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            for tile in polynomio:
                if tile.value == 0:
                    if len(tile.candidates) == 1:
                        tile.value = tile.candidates.pop()
                        ret = True
        return ret

    def _ExclusionRule(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1 in polynomio:
                    if t1.value != 0:
                        for t2 in polynomio:
                            if t2 != t1:
                                if t1.value in t2.candidates:
                                    t2.candidates.remove(t1.value)
                                    ret = True
        return ret

    def _ForbiddenNeighbour(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            for tile in polynomio:
                neighbours = tile.n8()
                for neighbour in neighbours:
                    if tile.value in neighbour.candidates:
                        neighbour.candidates.remove(tile.value)
                        ret = True
        return ret

    def _HiddenSingle(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1 in polynomio:
                    others = set()
                    for t2 in polynomio:
                        if t1 != t2:
                            others = others.union(t2.candidates)
                    diff = t1.candidates.difference(others)
                    if others and len(diff) == 1:
                        if (t1.candidates - others) != t1.candidates:
                            ret = True
                        t1.candidates.difference_update(others)
        return ret

    def _NakedPair(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2 in itertools.combinations(polynomio, 2):
                    if t1.candidates and t2.candidates:
                        c_union = t1.candidates.union(t2.candidates)
                        if len(c_union) == 2:
                            for t3 in polynomio:
                                if t3 not in [t1, t2]:
                                    if t3.candidates and (t3.candidates - c_union) != t3.candidates:
                                        ret = True
                                    t3.candidates.difference_update(c_union)
        return ret

    def _HiddenPairs(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2 in itertools.combinations(polynomio, 2):
                    if t1.candidates and t2.candidates:
                        c_union = t1.candidates.union(t2.candidates)
                        c_other = set()
                        for t3 in polynomio:
                            if t3 not in [t1, t2]:
                                c_other = c_other.union(t3.candidates)
                        diff = c_union.difference(c_other)
                        if c_other and len(diff) == 2:
                            for t4 in polynomio:
                                if t4 not in [t1, t2]:
                                    if (t4.candidates - diff) != t4.candidates:
                                        ret = True
                                    t4.candidates.difference_update(diff)
        return ret

    def _NakedTriples(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2, t3 in itertools.combinations(polynomio, 3):
                    if t1.candidates and t2.candidates and t3.candidates:
                        c_union = t1.candidates.union(t2.candidates.union(t3.candidates))
                        if len(c_union) == 3:
                            for t4 in polynomio:
                                if t4 not in [t1, t2, t3]:
                                    if (t4.candidates - c_union) != t4.candidates:
                                        ret = True
                                    t4.candidates.difference_update(c_union)
        return ret

    def _HiddenTriples(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2, t3 in itertools.combinations(polynomio, 3):
                    if t1.candidates and t2.candidates and t3.candidates:
                        c_union = t1.candidates.union(t2.candidates.union(t3.candidates))
                        c_other = set()
                        for t4 in polynomio:
                            if t4 not in [t1, t2, t3]:
                                c_other = c_other.union(t4.candidates)
                        diff = c_union.difference(c_other)
                        if c_other and len(diff) == 3:
                            if (t1.candidates - diff) != t1.candidates or (t2.candidates - diff) != t2.candidates or (t3.candidates - diff) != t3.candidates:
                                ret = True
                            t1.candidates -= diff
                            t2.candidates -= diff
                            t3.candidates -= diff
        return ret

    def _ForbiddenPairs(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2 in itertools.combinations(polynomio, 2):
                    if t1.candidates and t2.candidates:
                        c_union = t1.candidates.union(t2.candidates)
                        c_other_in_region = set().union(*(t3.candidates for t3 in polynomio if t3 not in [t1, t2]))
                        c_unique = c_union - c_other_in_region
                        if len(c_unique) == 2:
                            neighbours1 = set([n for n in t1.n8() if n not in polynomio])
                            neighbours2 = set([n for n in t2.n8() if n not in polynomio])

                            common = neighbours1.intersection(neighbours2)
                            if common:
                                for nt in common:
                                    if nt.candidates and nt.candidates.intersection(c_unique):
                                        ret = True
                                        nt.candidates.difference_update(c_unique)
        return ret

    def _ForbiddenTriple(self):
        ret = False
        for region, polynomio in self.polynomios_dic.items():
            if not self._solved_polynomio(polynomio):
                for t1, t2, t3 in itertools.combinations(polynomio, 3):
                    if t1.candidates and t2.candidates and t3.candidates:
                        c_union = t1.candidates.union(t2.candidates.union(t3.candidates))
                        c_other_in_region = set().union(*(t4.candidates for t4 in polynomio if t4 not in [t1, t2, t3]))
                        c_unique = c_union - c_other_in_region
                        if len(c_unique) == 3:
                            neighbours1 = set([n for n in t1.n8() if n not in polynomio])
                            neighbours2 = set([n for n in t2.n8() if n not in polynomio])
                            neighbours3 = set([n for n in t3.n8() if n not in polynomio])

                            common = neighbours1.intersection(neighbours2.intersection(neighbours3))
                            if common:
                                for nt in common:
                                    if nt.candidates and nt.candidates.intersection(c_unique):
                                        ret = True
                                        nt.candidates.difference_update(c_unique)
        return ret


class BacktrackSolver(BaseSolver):
    def __init__(self, grid, regions):
        super().__init__(grid, regions)

    def solve(self):
        self._backtrack(0, 0)
        self._update_main_grid()
        return Checker.solved(self.grid, self.regions)

    def _backtrack(self, i, j):
        if i >= self.rows:
            return True

        x, y = (i+1, 0) if (j+1 >= self.cols) else (i, j+1)
        if self.tile_grid[i, j].value != 0:
            return self._backtrack(x, y)

        polynomio = self.tile_grid[i, j].polynomio
        numbers = range(1, len(polynomio)+1)

        for number in numbers:
            self.tile_grid[i, j].value = number
            if self.tile_grid[i, j].is_consistent():
                ret = self._backtrack(x, y)
                if ret:
                    return True
            else:
                self.tile_grid[i, j].value = 0

        self.tile_grid[i, j].value = 0
        return False


class Checker:
    def __init__(self,):
        pass

    @staticmethod
    def solved(grid: np.array, regions: np.array):
        grid = grid.copy()
        regions = regions.copy()
        rows, cols = grid.shape
        for (i, j), value in np.ndenumerate(grid):
                region = regions[i, j]

                # if the tile doesn't have a value
                if value == 0:
                    return False

                # if the tile has a number which does not belong to its group
                region_length = np.count_nonzero(regions == region)
                numbers = set(range(1, region_length + 1))
                if value not in numbers:
                    return False
                
                # if the tile has conflicts with neighbour tiles
                ps = [(1, 0), (0, 1), (1, 1), (-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
                n8 = [(i + x, j + y) for x, y in ps if 0 <= (i + x) < rows and 0 <= (j + y) < cols]
                neighbour_values = set([grid[x, y] for x, y in n8])

                if value in neighbour_values:
                    return False
                
                # if the tile has a number already taken by someone in the group
                tiles = np.argwhere(regions == region)
                in_group_values = list(grid[x, y] for x, y in tiles if (x, y) != (i, j))

                if value in in_group_values:
                    return False
                
        return True

    @staticmethod
    def rot_tiles(grid: np.array, regions: np.array):
        rotten = list()
        grid = grid.copy()
        regions = regions.copy()
        rows, cols = grid.shape
        for i in range(rows):
            for j in range(cols):
                value = grid[i, j]
                region = regions[i, j]

                # if the tile doesn't have a value
                if value == 0:
                    rotten.append((i,j))
                    continue

                # if the tile has a number which does not belong to its group
                numbers = set(range(1, np.count_nonzero(regions == region) + 1))
                if value not in numbers:
                    rotten.append((i, j))
                    continue
                
                # if the tile has conflicts with neighbour tiles
                ps = [(1, 0), (0, 1), (1, 1), (-1, -1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
                n8 = [(i + x, j + y) for x, y in ps if 0 <= (i + x) < rows and 0 <= (j + y) < cols]
                neighbour_values = set([grid[x, y] for x, y in n8])

                if value in neighbour_values:
                    rotten.append((i, j))
                    continue
                
                # if the tile has a number already taken by someone in the group
                xs, ys = np.where(regions == region)
                in_group_values = list(grid[p] for p in zip(xs, ys) if p != (i, j))

                if value in in_group_values:
                    rotten.append((i, j))
                
        return rotten

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
    parser.add_argument('--solver', help='Solver to be used', choices=['de', 'backtrack'])
    parser.add_argument('--path', help='Path to target Suguru puzzle')
    args = parser.parse_args()

    solver_arg = args.solver
    path = args.path

    grid, solved, regions = Suguru.parse_suguru_binary(path)
    if solver_arg == 'de':
        solver = DeterministicEngine(grid, regions)
    elif solver_arg == 'backtrack':
        solver = BacktrackSolver(grid, regions)

    print('Original grid')
    print(grid)
    start = time.perf_counter()
    solved = solver.solve()
    elapsed = time.perf_counter() - start
    print('Final grid')
    print(solver.grid)
    print('Elapsed: {}s'.format(elapsed))

    if solved:
        print('[+] Solved')
    else:
        print('[-] Not solved')

if __name__ == '__main__':
    main()