import time
import SuguruSolvers
import numpy as np
import SuguruGui
import threading
import os
import tkinter as tk
import json
import SASolver


class Suguru:
    def __init__(self, grid, regions, solver: SuguruSolvers.BaseSolver, delay=0):
        # Organize
        self.grid, self.regions = grid, regions
        self.rows, self.cols = self.grid.shape
        self.size = self.rows*self.cols

        self.solver = solver(self.grid, self.regions, delay)

        # Graphical control
        self.root = tk.Tk()
        self.gui = SuguruGui.SuguruGUI(self.root,
                                       self.rows, self.cols,
                                       self.grid, self.regions,
                                       cell_size=80
                                       )


    def show(self):
        self.root.mainloop()

    def solve(self):
        t = threading.Thread(target=self.solver.solve, daemon=True)
        t.start()

        self._poll_updates(10)

    def _poll_updates(self, delay):
        #self.solver._update_main_grid()
        self.grid = self.solver.grid
        self.gui.set_grid(self.grid)

        if SuguruSolvers.Checker.solved(self.grid, self.regions):
            self.gui.set_solved()
        self.root.after(delay, lambda: self._poll_updates(delay))

def parse_suguru_line_to_grid_and_regions(line):
    data = json.loads(line)
    rows, cols = data["rows"], data["cols"]

    # Create empty grid
    grid = np.zeros((rows, cols), dtype=int)
    regions = np.zeros((rows, cols), dtype=int)

    # Fill grid
    cells_str = data["cells"]
    for idx in range(rows * cols):
        val = int(cells_str[2*idx])
        row, col = divmod(idx, cols)
        grid[row, col] = val

    # Fill regions
    for region_id, group_str in enumerate(data["groups"]):
        for i in range(0, len(group_str), 2):
            cell_idx = int(group_str[i:i+2])
            r, c = divmod(cell_idx, cols)
            regions[r, c] = region_id

    return grid, regions


def parse_suguru_binary(path):
    if not os.path.isfile(path):
        raise Exception('Error: invalid file path')

    # Load from file
    with open(path, 'rb') as fp:
        rows = int.from_bytes(fp.read(2))
        cols = int.from_bytes(fp.read(2))
        arr = np.fromfile(fp, dtype=np.int16).reshape(3, rows, cols)
    return arr

def parse_json_file(path):
    path = 'puzzles.json'
    puzzles = list()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid, regions = parse_suguru_line_to_grid_and_regions(line)
            puzzles.append((grid, regions))
    return puzzles

def main():

    grid, solution, regions = parse_suguru_binary(
        './samples/9x9_1.data'
    )
    s = Suguru(grid, regions, SASolver.SASolver, 0.01)
    #s = Suguru(grid, regions, SuguruSolvers.DeterministicEngine, 0.2)
    s.solve()
    s.show()


if __name__ == '__main__':
    main()
