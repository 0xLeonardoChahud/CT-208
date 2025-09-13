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
        """Update all cells at once from a 2D numpy array or suguru.grid."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.grid = grid
                val = self.grid[i, j]
                self.canvas.itemconfig(self.cells[(i, j)], text=str(val) if val else "")
        self.root.update_idletasks()

    def set_solved(self):
        for k in range(self.rows):
            for l in range(self.cols):
                v = self.grid[k,l]
                self.canvas.itemconfig(self.cells[(k, l)], text=str(v) if v else "", fill='green')
        self.root.update_idletasks()        

def display_suguru(rows, cols, grid, regions):
    root = tk.Tk()
    gui = SuguruGUI(root, rows, cols, grid, regions, cell_size=80)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to Suguru file')

    args = parser.parse_args()

    path = args.path

    if not os.path.isfile(path):
        print('[-] Invalid file')
        return

    with open(path, 'rb') as fp:
        rows = int.from_bytes(fp.read(2))
        cols = int.from_bytes(fp.read(2))

        arr = np.fromfile(fp, dtype=np.int16).reshape(3, rows, cols)
        grid, solved, regions = arr
    
    display_suguru(rows, cols, grid, regions)
        
        

if __name__ == '__main__':
    main()