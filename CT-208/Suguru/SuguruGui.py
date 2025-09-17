import tkinter as tk
import argparse
import os
import numpy as np


class SuguruGUI:
    def __init__(self, root, rows, cols, grid, regions, cell_size=60):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid = grid
        self.regions = regions
        self.tips = np.argwhere(self.grid != 0)

        self.canvas = tk.Canvas(
            root,
            width=self.cols * cell_size,
            height=self.rows * cell_size,
            bg="white"
        )
        self.canvas.pack()

        self.cells = {}
        self._draw_grid()

    def _draw_grid(self):
        for i in range(self.rows):
            for j in range(self.cols):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                # Base cell rectangle
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="gray", width=1
                )

                # Draw initial number
                val = self.grid[i, j]
                text = self.canvas.create_text(
                    (x1 + x2) // 2, (y1 + y2) // 2,
                    text=str(val) if val else "",
                    font=("Arial", self.cell_size // 2)
                )
                self.cells[(i, j)] = text

        # Region bold borders
        self._draw_region_borders()

    def _draw_region_borders(self):
        regions = self.regions
        for i in range(self.rows):
            for j in range(self.cols):
                rid = regions[i, j]
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                if i == 0 or regions[i - 1, j] != rid:
                    self.canvas.create_line(x1, y1, x2, y1, width=3)
                if j == 0 or regions[i, j - 1] != rid:
                    self.canvas.create_line(x1, y1, x1, y2, width=3)
                if i == self.rows - 1 or regions[i + 1, j] != rid:
                    self.canvas.create_line(x1, y2, x2, y2, width=3)
                if j == self.cols - 1 or regions[i, j + 1] != rid:
                    self.canvas.create_line(x2, y1, x2, y2, width=3)

    def set_grid(self, grid):
        self.grid = grid.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.grid[i, j]
                values = [self.grid[x, y] for x, y in self._n8(i, j)]
                if (i, j) in self.tips:
                    self.canvas.itemconfig(self.cells[(i, j)],
                                           text=str(val) if val else "",
                                            fill='lightblue'
                                           )
                elif val in values:
                    self.canvas.itemconfig(self.cells[(i, j)],
                                           text=str(val) if val else "",
                                           fill='red'
                                           )
                else:
                    self.canvas.itemconfig(self.cells[(i, j)],
                                           text=str(val) if val else "",
                                           fill='black'
                                           )
        self.root.update_idletasks()

    def set_tips(self, tips):
        self.tips = tips.copy()

    def set_solved(self):
        for i in range(self.rows):
            for j in range(self.cols):
                v = self.grid[i, j]
                self.canvas.itemconfig(self.cells[(i, j)],
                                       text=str(v) if v else "", fill='green'
                                       )
        self.root.update_idletasks()

    def _n8(self, i, j):
        moves = [(1,0),(0,1),(1,1),(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1)]
        n8s = list()
        for m, n in moves:
            nx = i + m
            ny = j + n
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                n8s.append((nx, ny))
        return n8s

def display_suguru(rows, cols, grid, regions):
    root = tk.Tk()
    _ = SuguruGUI(root, rows, cols, grid, regions, cell_size=80)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to Suguru file')
    parser.add_argument('--solved', help='Show solved Suguru', action='store_true')

    parser.set_defaults(solved=False)

    args = parser.parse_args()

    path = args.path
    solved_flag = args.solved

    if not os.path.isfile(path):
        print('[-] Invalid file')
        return

    with open(path, 'rb') as fp:
        rows = int.from_bytes(fp.read(2))
        cols = int.from_bytes(fp.read(2))

        arr = np.fromfile(fp, dtype=np.int16).reshape(3, rows, cols)
        grid, solved, regions = arr

    if solved_flag:
        display_suguru(rows, cols, solved, regions)
    else:
        display_suguru(rows, cols, grid, regions)


if __name__ == '__main__':
    main()
