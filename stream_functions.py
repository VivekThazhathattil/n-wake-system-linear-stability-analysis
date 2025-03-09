# Module for creating stream functions for the analytical 
# linear stability analysis of n-wake systems with linearly
# varying shear profiles

import numpy as np
import sympy as sp
import copy

class StreamFunctions:
    def __init__(self, n_wakes):
        self.n_wakes = n_wakes
        self.stream_functions = []
        self.region_ids = []
        self.region_bounds = []
        self.interface_locations = []
        self.y = sp.symbols('y') # cross-stream coordinate
        self.L = sp.symbols('L') # spacing between centres of bluff bodies
        self.delta = sp.symbols('delta') # finite thickness of the shear layer
        self.D = sp.symbols('D') # bluff body diameter
        self.alpha = sp.symbols('alpha') # wavenumber
        self.c = sp.symbols('c') # phase speed
        self.s = sp.symbols('s') # for varicose(-1)/sinuous(1) modes 
        self.U_ambient = sp.symbols('U_ambient') # ambient region uniform velocity
        self.U_wake = sp.symbols('U_wake') # wake region uniform velocity
        self.rho_ambient = sp.symbols('rho_ambient') # ambient region density
        self.rho_wake = sp.symbols('rho_wake') # wake region density
        self.Lambda = sp.symbols('Lambda') # shear ratio
        self.S = sp.symbols('S') # density ratio

        # Create the piecewise regions for which stream functions will be defined
        self.create_region_ids()
        self.n_regions = len(self.region_ids)
        self.region_widths = []

        # Obtain the location of boundaries and interfaces
        self.get_interface_locations_and_region_bounds()

        # create the stream functions themselves
        self.create_stream_functions()

    def get_interface_locations_and_region_bounds(self):
        if self.n_wakes == 0:
            self.region_bounds.append([self.delta/2, np.inf])
            self.region_bounds.append([-self.delta/2, self.delta/2])
            self.region_bounds.append([-np.inf, -self.delta/2])
            self.interface_locations.append(self.delta/2)
            self.interface_locations.append(-self.delta/2)
            return

        for ii in range(1, self.n_regions-1):
            if self.region_ids[ii] == 'shear_inc':
                self.region_widths.append(self.delta)
            elif self.region_ids[ii] == 'shear_dec':
                self.region_widths.append(self.delta)
            elif self.region_ids[ii] == 'wake': 
                self.region_widths.append(self.D - self.delta)
            elif self.region_ids[ii] == 'ambient':
                self.region_widths.append(self.L - self.D - self.delta)

        self.region_widths.insert(0, np.inf) # for the first ambient region
        self.region_widths.append(np.inf) # for the last ambient region

        temp_locs = []
        for ii in range(0, self.n_regions-1):
            if ii == 0:
                temp_locs.append([0, 0])
            else:
                temp_locs.append([temp_locs[ii-1][1], temp_locs[ii-1][1] + self.region_widths[ii]])
        temp_locs.reverse()
        temp_locs = [
            [temp_locs[ii][0] - (1/2)*(temp_locs[0][1]), 
             temp_locs[ii][1] - (1/2)*(temp_locs[0][1])]  
             for ii in range(len(temp_locs))
             ]
        self.region_bounds = copy.deepcopy(temp_locs)
        self.region_bounds.pop()
        self.region_bounds.append(
            [-np.inf, self.region_bounds[len(self.region_bounds)-1][0]]
            )
        self.region_bounds.insert(0,[self.region_bounds[0][1],np.inf])
        # get interface locations
        self.interface_locations = [self.region_bounds[ii][0] for ii in range(self.n_regions - 1)]

    def create_region_ids(self):
        region_idx = 0
        num_wakes_created = 0
        if self.n_wakes == 0: # corresponds to a single shear layer
            self.region_ids = ['ambient', 'shear_inc', 'ambient']
            return
        while True:
            if region_idx == 0:
                self.region_ids.append(f'ambient')
            elif self.region_ids[region_idx-1] == 'ambient':
                # shear layer with a positive dU/dy
                self.region_ids.append(f'shear_inc')
            elif self.region_ids[region_idx-1] == 'shear_inc':
                self.region_ids.append(f'wake')
                num_wakes_created += 1
            elif self.region_ids[region_idx-1] == 'wake':
                # shear layer with a negative dU/dy
                self.region_ids.append(f'shear_dec')
            else:
                self.region_ids.append(f'ambient')
                if num_wakes_created == self.n_wakes:
                    break
            region_idx += 1

    def create_stream_functions(self):
        # create the coefficients for the stream functions
        self.A = sp.symbols(f'A1:{self.n_regions+1}')
        self.B = sp.symbols(f'B1:{self.n_regions+1}')

        for ii in range(self.n_regions):
            y_bnd = self.region_bounds[ii]
            if ii == 0:
                self.stream_functions.append(
                    self.A[ii] * sp.exp(-self.alpha * (self.y - y_bnd[0]))
                    )
            elif ii == self.n_regions - 1:
                self.stream_functions.append(
                    self.B[ii] * sp.exp(self.alpha * (self.y - y_bnd[1]))
                    )
            else:
                y_ref = 0.5 * (y_bnd[0] + y_bnd[1])
                self.stream_functions.append(
                    self.A[ii] * sp.exp(-self.alpha * (self.y - y_ref)) + 
                    self.B[ii] * sp.exp(self.alpha * (self.y - y_ref))
                )
    def get_latex_expressions(self):
        latex_expressions = []
        for ii in range(self.n_regions):
            latex_expressions.append(
                sp.latex(sp.simplify
                         (self.stream_functions[ii], rational=True)))
        return latex_expressions