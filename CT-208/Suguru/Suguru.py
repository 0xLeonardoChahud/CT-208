import numpy as np
import SuguruGui
import threading
import os
import tkinter as tk
import argparse
import SASolver
import SuguruSolvers

class Suguru:
    def __init__(self, grid, regions, solver: SuguruSolvers.BaseSolver, delay=0):
        # Organize
        self.grid, self.regions = grid, regions
        self.rows, self.cols = self.grid.shape
        self.size = self.rows*self.cols
        self.tips = np.argwhere(self.grid != 0)

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
        self.grid = self.solver.grid.copy()
        self.tips = self.solver.tips.copy()
        self.gui.set_grid(self.grid)
        self.gui.set_tips(self.tips)

        if SuguruSolvers.Checker.solved(self.grid, self.regions):
            self.gui.set_solved()
            return
        self.root.after(delay, lambda: self._poll_updates(delay))



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

    grid, solution, regions = parse_suguru_binary(
        path
    )
    s = Suguru(grid, regions, SASolver.SASolver, 0.001)
    s.solve()
    s.show()


if __name__ == '__main__':
    main()
