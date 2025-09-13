import time
import SuguruSolvers
import numpy as np
import SuguruGui
import threading
import os
import tkinter as tk
import json


class Suguru:
    def __init__(self, grid, solved, regions):
        # Organize
        self.grid, self.solved, self.regions = grid, solved, regions
        self.rows, self.cols = self.grid.shape
        self.size = self.rows*self.cols

        # Graphical control
        self.root_window = tk.Tk()
        self.gui = SuguruGui.SuguruGUI(self.root_window,
                                       self.rows, self.cols,
                                       self.grid, self.regions,
                                       cell_size=80
                                       )

        # Deterministic engine
        # self.de = DeterministicEngine(self.grid, self.regions)

    def show(self, solve=False):
        if solve:
            thread = threading.Thread(target=self._update_grid_periodically,
                                      args=(self.gui, self),
                                      daemon=True
                                      )
            thread.start()
        self.gui.root.mainloop()

    @staticmethod
    def _update_grid_periodically(gui, suguru):
        while True:
            de = SuguruSolvers.BacktrackSolver(suguru.grid, suguru.regions)
            de.solve()
            suguru.grid = de.grid

            # Schedule the GUI update on the main thread
            gui.root.after(0, gui.set_grid, suguru.grid)
            time.sleep(0.1)

            if de._solved():
                gui.set_solved()
                time.sleep(2)
                # gui.root.after(0, gui.root.quit)
                break


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


def main():

    path = 'puzzles.json'
    puzzles = list()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid, regions = parse_suguru_line_to_grid_and_regions(line)
            puzzles.append((grid, regions))

    grid, solution, regions = parse_suguru_binary(
        './unique_samples/9x9_45.data'
    )
    s = Suguru(grid, solution, regions)
    s.show(True)


if __name__ == '__main__':
    main()
