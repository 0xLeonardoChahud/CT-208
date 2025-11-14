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



# Usage:  grid, solution, regions = parse_suguru_binary(some_path)
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
    parser.add_argument('--delay', help='Delay to show animation', default=0.1)
    parser.add_argument('--show', help='Show animation', default=False, action='store_true')
    parser.add_argument('--solver', help='Select the solver to use', choices=['sa', 'de', 'bt'])
    args = parser.parse_args()

    path = args.path
    delay = float(args.delay)
    solver = args.solver
    show = args.show

    grid, solution, regions = parse_suguru_binary(
        path
    )

    solved = False
    if solver == 'sa':
        print('[+] Solving with Simulated Annealing')
        if show:
            s = Suguru(grid, regions, SASolver.SASolver, delay)
            solved = s.solve()
            s.show()
        else:
            s = SASolver.SASolver(grid, regions)
            solved = s.solve()
    elif solver == 'de':
        if show:
            s = Suguru(grid, regions, SuguruSolvers.DeterministicEngine, delay)
            solved = s.solve()
            s.show()
        else:
            s = SuguruSolvers.DeterministicEngine(grid, regions)
            solved = s.solve()
    elif solver == 'bt':
        if show:
            s = Suguru(grid, regions, SuguruSolvers.BacktrackSolver, delay)
            solved = s.solve()
            s.show()
        else:
            s = SuguruSolvers.BacktrackSolver(grid, regions)
            solved = s.solve()
    
    if solved:
        print('[+] Solved')
    else:
        print('[-] Not solved')


if __name__ == '__main__':
    main()
