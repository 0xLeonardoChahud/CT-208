import itertools
import time
import argparse
import numpy as np
import threading
import suguru_gui
import typing

class Cell:
    def __init__(self, row, col, side):
        self.numbers = set()
        self.numbers_copy = set()
        self.value = 0
        self.region = 0
        self.region_size = 0
        self.row = row
        self.col = col
        self.side = side

    def __repr__(self):
        return str(self.value)

    def get_index(self):
        return self.row, self.col

    def n8(self):
        aux = [(1,0),(1,1),(-1,0),(-1,-1),(0,1),(0,-1),(1,-1),(-1,1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y + self.col
            if 0 <= nx < self.side and 0 <= ny < self.side:
                ret.append((nx,ny))
        return ret

    def n4(self):
        aux = [(1,0),(-1,0),(0,1),(0,-1)]
        ret = list()
        for x, y in aux:
            nx,ny = x + self.row, y + self.col
            if 0 <= nx < self.side and 0 <= ny < self.side:
                ret.append((nx,ny))
        
        return ret

class DeterministicEngine:
    def __init__(self, grid, regions):
        self.grid = grid
        self.regions = regions

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

    def _update_grid(self):
        pass

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


class SuguruGenerator:
    def __init__(self, m, n):
        self.m = m
        self.n = n

        self.grid = np.zeros((self.m, self.n), dtype=int)
        self.regions = np.zeros((self.m, self.n), dtype=int)

        self.region_map = dict()


    def generate(self):
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

                n8v = [self.grid[i][j] for i,j in self._n8(i,j)]

                if value in n8v:
                    continue
                    
                # Setup tile
                self.grid[i][j] = value
                self.regions[i][j] = region
                
                if p in pos:
                    pos.remove(p)

                neighbours = [n for n in self._n4(i,j) if self.grid[n[0]][n[1]] == 0]
                path = path.union(neighbours)
                value += 1
                cnt += 1

                if region not in self.region_map:
                    self.region_map[region] = list()
                self.region_map[region].append(p)
                
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
                    if value not in n8tv:
                        self.regions[k][l] = r
                        self.grid[k][l] = value
                        self.region_map[r].append(p)
                        zeros_after -= 1
                        break     
            if zeros == zeros_after:
                break
        
        return self._solved()

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
             
    def _n4(self, i, j):
        moves = [(1,0),(0,1),(-1,0),(0,-1)]
        n4p = list()
        for move in moves:
            mx, my = i+move[0], j+move[1]
            if 0 <= mx < self.m and 0 <= my < self.n:
                n4p.append((mx,my))
        return n4p
    
    def _solved(self):
        if np.any(self.regions == 0) or np.any(self.grid == 0):
            return False
    
        # Group permutation check
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
    args = parser.parse_args()

    rows = int(args.rows)
    cols = int(args.cols)
    cnt = int(args.count)

    created = 0
    start_time = time.perf_counter()
    m = SuguruGenerator(rows, cols)
    while created < cnt:
        if m.generate():
            created += 1
            conc = np.vstack([m.grid, m.regions])
            with open(f'./samples/{rows}x{cols}_{created}.data', 'wb') as fp:
                fp.write(rows.to_bytes(2))
                fp.write(cols.to_bytes(2))
                conc.astype('int16').tofile(fp)
    end_time = time.perf_counter()
    print(f'Elapsed time: {end_time - start_time} seconds')  

if __name__ == '__main__':
    main()



