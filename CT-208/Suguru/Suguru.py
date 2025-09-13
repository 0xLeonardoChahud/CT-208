import time
import SuguruSolvers
import numpy as np
import suguru_gui
import threading
import os
import tkinter as tk


class Suguru:
    def __init__(self, file_path, tips=0):
        if not os.path.isfile(file_path):
            raise Exception('Error: invalid file path')
        
        # Load from file
        self.file_path = file_path
        with open(self.file_path, 'rb') as fp:
            rows = int.from_bytes(fp.read(2))
            cols = int.from_bytes(fp.read(2))
            arr = np.fromfile(fp, dtype=np.int16).reshape(3, rows, cols)

        # Organize
        self.grid, self.solved, self.regions = arr
        self.rows = rows
        self.cols = cols
        self.size = self.rows*self.cols

        # Graphical control
        self.root_window = tk.Tk()
        self.gui = suguru_gui.SuguruGUI(self.root_window, self.rows, self.cols, self.grid, self.regions, cell_size=80)

        # Deterministic engine
        #self.de = DeterministicEngine(self.grid, self.regions)


    def show(self, solve=False):
        if solve:
            thread = threading.Thread(target=self._update_grid_periodically, args=(self.gui, self), daemon=True)
            thread.start()
        self.gui.root.mainloop()

    @staticmethod
    def _update_grid_periodically(gui, suguru):
        while True:
            
            de = SuguruSolvers.DeterministicEngine(suguru.grid, suguru.regions)
            while de._apply_rules():
                de._update_main_grid()
                suguru.grid = de.grid
                # Schedule the GUI update on the main thread
                gui.root.after(0, gui.set_grid, suguru.grid)
                time.sleep(0.1)  # update every second
            if de._solved():
                gui.set_solved()
            gui.root.after(0, gui.root.quit)
            break
                 

def main():

    for sample in os.listdir('./unique_samples/'):
        path = os.path.join('./unique_samples/', sample)
        s = Suguru(path)
        s.show(solve=True)

    


if __name__ == '__main__':
    main()