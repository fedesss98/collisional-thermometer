"""
Class circuit -- useful to draw the collisional model in a quantum circuit form
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Circuit:
    def __init__(self, layers, chains):
        self.layers = layers
        self.chains = chains
        self.fig = None
        self.ax = None 

    def draw_circuit(self, aspect=(1, 1), usetex=False, show=False):
        
        self.update_rcParams(usetex)
        w, h = aspect
        fig, ax = plt.subplots(figsize=(w*(self.layers*self.chains), h*(self.chains)))

        columns = 2*self.chains*self.layers

        # Draw the system, environment, and ancillas in the first column
        elements = ['E', 'S'] + [f'$a_{i+1}$' for i in range(self.chains)]
        for i, element in enumerate(elements):
            ax.text(-0.5, self.chains+1-i, element, ha='center', va='center', fontsize=14, 
                    bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
            ax.axhline(self.chains+1-i, -1.0, columns+0.1, lw=1, c='k', zorder=0)

        # Draw the gates in the subsequent columns
        for col in range(columns):
            # System-Environment interactions
            if col % (2*self.chains) == 0:
                self.interact_system_env(ax, col)
            else:
                k = col%(2*self.chains)
                for row in range(self.chains):
                    if row == (k - 1)/2:
                        self.interact_system_ancilla(ax, col, row)
                        next_col = col + 1
                        if next_col == columns:
                            self.measure_ancilla(ax, col, row)
                        elif next_col % (2*self.chains) != 0:
                            self.interact_ancilla_ancilla(ax, col, row)

        # Add column separators
        for col in range(columns+1):
            if col % (2*self.chains) == 0 or col == columns:
                ax.axvline(col, -0.5, self.chains+1, ls='--', c='lightgray', lw=1)
            else:
                ax.axvline(col, 0, self.chains+1, ls=':', c='lightgray', lw=1)


        ax.set_ylim(-0.4, self.chains+1+0.4)
        ax.set_xlim(-0.4, columns+1)
        ax.axis('off')

        self.fig = fig
        self.ax = ax

        if show:
            plt.show()

        return fig, ax
    
    def update_rcParams(self, usetex):
        plt.rcParams.update({'text.usetex': usetex,'font.family': 'serif'})
    
    def interact_system_env(self, ax, col):
        ax.add_patch(patches.Rectangle((col+0.1, self.chains-0.12), 0.8, 1.25, 
                                       edgecolor='black', facecolor='red'))

        ax.text(col+0.5, self.chains+0.5, '$T$', ha='center', va='center', fontsize=10,)        

    def interact_system_ancilla(self, ax, col, row):
        # Compute distance from System to ancilla
        dist = (row+1.25)
        ax.add_patch(patches.Rectangle((col+0.1, self.chains-dist+0.12), 0.8, dist, 
                                       edgecolor='black', facecolor='lightgrey'))
        ax.text(col+0.5, self.chains-dist/2, '$U_c$', 
                ha='center', va='center', fontsize=10, rotation=0)        

    def interact_ancilla_ancilla(self, ax, col, row):
            ax.add_patch(patches.Rectangle((col+1.1, self.chains-row-2.12), 0.8, 1.25, 
                                        edgecolor='black', facecolor='lightgrey'))
            ax.text(col+1.5, self.chains-row-1.5, '$U_e$', 
                    ha='center', va='center', fontsize=10, rotation=0)        

    def measure_ancilla(self, ax, col, row):
            ax.add_patch(patches.Rectangle((col+1.1, -0.2), 0.8, 0.4, 
                                        edgecolor='black', facecolor='lightgrey'))
            ax.text(col+1.5, 0.0, 'M', ha='center', va='center', fontsize=10,)  
        
    def save_circuit(self, filename='./circuit'):
        if self.fig is None:
            raise Exception(
                "Cannot save circuit: no figure is drawn yet! Try running draw_circuit method first.")

        self.fig.savefig(f"{filename}.png", format='png')
        self.fig.savefig(f"{filename}.eps", format='eps')


if __name__ == "__main__":
    circuit = Circuit(4, 3)
    circuit.draw_circuit(show=False)
    