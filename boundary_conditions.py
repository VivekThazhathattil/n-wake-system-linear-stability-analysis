import sympy as sp
import numpy as np
import copy
from stream_functions import StreamFunctions

class BoundaryConditions:
    def __init__(self, stream_functions):
        """
        Initializes the boundary conditions for the system.

        Parameters:
        stream_functions (list): A list of stream functions for the system.

        Attributes:
        stream_functions (list): Stores the provided stream functions.
        interfacial_equations (list): Stores the interfacial equations.
        kinematic_bc_equations (list): Stores the kinematic boundary condition equations.
        dynamic_bc_equations (list): Stores the dynamic boundary condition equations.
        M (None or matrix): The final equation coefficient matrix.
        M_non_dim (None or matrix): The final non-dimensional equation coefficient matrix.
        X (None or list): The final equation variables.
        A (None or matrix): The final A matrix for the generalized eigenvalue problem.
        A_non_dim (None or matrix): The final non-dimensional A matrix for the generalized eigenvalue problem.
        B (None or matrix): The final B matrix for the generalized eigenvalue problem.
        B_non_dim (None or matrix): The final non-dimensional B matrix for the generalized eigenvalue problem.
        n_equations (int): The number of equations.

        Methods:
        apply_interfacial_boundary_conditions: Applies the interfacial boundary conditions.
        create_matrix_M: Creates the matrix M.
        create_matrix_M_non_dim: Creates the non-dimensional matrix M.
        create_matrix_A: Creates the matrix A.
        create_matrix_A_non_dim: Creates the non-dimensional matrix A.
        create_matrix_B: Creates the matrix B.
        create_matrix_B_non_dim: Creates the non-dimensional matrix B.
        """
        self.stream_functions = stream_functions
        self.interfacial_equations = []
        self.kinematic_bc_equations = []
        self.dynamic_bc_equations = []
        self.M = None # final eq coeff matrix
        self.M_non_dim = None # final non-dim eq coeff matrix
        self.X = None # final eq variables
        self.A = None # final A matrix for gen. eig. problem
        self.A_non_dim = None # final non-dim A matrix for gen. eig. problem
        self.B = None # final B matrix for gen. eig. problem
        self.B_non_dim = None # final non-dim B matrix for gen. eig. problem
        self.n_equations = 0
        self.apply_interfacial_boundary_conditions()
        self.create_matrix_M()
        self.create_matrix_M_non_dim()
        self.create_matrix_A()
        self.create_matrix_A_non_dim()
        self.create_matrix_B()
        self.create_matrix_B_non_dim()

    def apply_interfacial_boundary_conditions(self):
        self.apply_kinematic_boundary_conditions()
        self.apply_dynamic_boundary_conditions()
        self.n_equations = len(self.interfacial_equations)

    def apply_kinematic_boundary_conditions(self):
        sf = self.stream_functions
        psi = sf.stream_functions

        # for each interface, apply kinematic boundary condition
        for ii in range(sf.n_regions-1):
            exp_lhs = psi[ii].subs(sf.y, sf.interface_locations[ii])
            exp_rhs = psi[ii+1].subs(sf.y, sf.interface_locations[ii])

            # apply varicose/sinuous modes
            if sf.n_wakes > 0:
                exp_lhs = exp_lhs.subs(sf.A[sf.n_regions//2], sf.s * sf.B[sf.n_regions//2])
                exp_rhs = exp_rhs.subs(sf.A[sf.n_regions//2], sf.s * sf.B[sf.n_regions//2])

            self.interfacial_equations.append(
                sp.Eq(exp_lhs, exp_rhs)
            )
            self.kinematic_bc_equations.append(
                sp.Eq(exp_lhs, exp_rhs)
            )

    def apply_dynamic_boundary_conditions(self):
        sf = self.stream_functions
        psi = sf.stream_functions

        # for each interface, apply dynamic boundary conditions
        for ii in range(sf.n_regions - 1):
            region_id = sf.region_ids[ii]
            c = sf.c
            if region_id == 'ambient':
                U1 = sf.U_ambient
                U2 = sf.U_ambient
                U1y = 0
                U2y = (sf.U_ambient - sf.U_wake) / sf.delta
                rho1 = sf.rho_ambient
                rho2 = sf.rho_ambient
            elif region_id == 'wake':
                U1 = sf.U_wake
                U2 = sf.U_wake
                U1y = 0
                U2y = -(sf.U_ambient - sf.U_wake) / sf.delta
                rho1 = sf.rho_wake
                rho2 = sf.rho_ambient
            elif region_id == 'shear_inc':
                U1 = sf.U_wake
                U2 = sf.U_wake
                U1y = (sf.U_ambient - sf.U_wake) / sf.delta
                U2y = 0
                rho1 = sf.rho_ambient
                rho2 = sf.rho_wake
            elif region_id == 'shear_dec':
                U1 = sf.U_ambient
                U2 = sf.U_ambient
                U1y = -(sf.U_ambient - sf.U_wake) / sf.delta
                U2y = 0
                rho1 = sf.rho_ambient
                rho2 = sf.rho_ambient

            psi1 = psi[ii]
            psi2 = psi[ii+1]
            psi1y = psi1.diff(sf.y)
            psi2y = psi2.diff(sf.y)
            
            exp_lhs = sf.delta * (rho1 * (U1 - c) * psi1y - rho1 * U1y * psi1)
            exp_rhs = sf.delta * (rho2 * (U2 - c) * psi2y - rho2 * U2y * psi2)

            # substitute for y location
            exp_lhs = exp_lhs.subs(sf.y, sf.interface_locations[ii])
            exp_rhs = exp_rhs.subs(sf.y, sf.interface_locations[ii])

            # apply varicose/sinuous modes
            if sf.n_wakes > 0:
                exp_lhs = exp_lhs.subs(sf.A[sf.n_regions//2], sf.s * sf.B[sf.n_regions//2])
                exp_rhs = exp_rhs.subs(sf.A[sf.n_regions//2], sf.s * sf.B[sf.n_regions//2])

            self.interfacial_equations.append(
                sp.Eq(exp_lhs, exp_rhs)
            )
            self.dynamic_bc_equations.append(
                sp.Eq(exp_lhs, exp_rhs)
            )

    def create_matrix_M(self):
        sf = self.stream_functions
        eqns_reqd = []
        if sf.n_wakes == 0:
            n_k_eq = len(self.kinematic_bc_equations)
            n_d_eq = len(self.dynamic_bc_equations)
        else:
            n_k_eq = len(self.kinematic_bc_equations)//2
            n_d_eq = len(self.dynamic_bc_equations)//2
        for ii in range(n_k_eq):
            eqns_reqd.append(self.kinematic_bc_equations[ii])
        for ii in range(n_d_eq):
            eqns_reqd.append(self.dynamic_bc_equations[ii])

        symbol_set = [*sf.A[0:n_k_eq], *sf.B[1:n_k_eq + 1]]
        #print(symbol_set)
        self.M, self.X = sp.linear_eq_to_matrix(eqns_reqd, symbol_set)

    def create_matrix_M_non_dim(self):
        sf = self.stream_functions
        if self.M == None:
            self.create_matrix_M()
        self.M_non_dim = self.M.subs({
            sf.rho_ambient: 1,
            sf.rho_wake: sf.S,
            sf.D: 1
        })

        if sf.n_wakes == 0:
            self.M_non_dim = self.M_non_dim.subs({
                sf.delta: 1,
                sf.U_ambient: 1,
                sf.U_wake: -1,
                sf.S: 1
            })
        else:
            self.M_non_dim = self.M_non_dim.subs({
                sf.U_ambient: 1 - sf.Lambda,
                sf.U_wake: 1 + sf.Lambda
            })

    def create_matrix_A(self):
        sf = self.stream_functions
        A_temp = self.M.subs(sf.c, 0)
        self.A = copy.deepcopy(A_temp)

    def create_matrix_A_non_dim(self):
        sf = self.stream_functions
        if self.A == None:
            self.create_matrix_A()
        self.A_non_dim = self.A.subs({
            sf.rho_ambient: 1,
            sf.rho_wake: sf.S,
            sf.D: 1,
        })

        if sf.n_wakes == 0:
            self.A_non_dim = self.A_non_dim.subs({
                sf.delta: 1,
                sf.U_ambient: 1,
                sf.U_wake: -1,
                sf.S: 1
            })
        else:
            self.A_non_dim = self.A_non_dim.subs({
                sf.U_ambient: 1 - sf.Lambda,
                sf.U_wake: 1 + sf.Lambda
            })

    def create_matrix_B(self):
        """
        Creates the matrix B by performing operations on the matrices A and M.

        This method performs the following steps:
        1. Subtracts matrix M from matrix A to create a temporary matrix B_temp.
        2. Substitutes the stream function constant (sf.c) in B_temp with 1 to create B_temp_subbed.
        3. Deep copies B_temp_subbed to the instance variable B.

        Attributes:
            self.A (Matrix): The matrix A.
            self.M (Matrix): The matrix M.
            self.stream_functions (StreamFunctions): An object containing stream functions, including the constant c.
            self.B (Matrix): The resulting matrix B after the operations.

        Returns:
            None
        """
        sf = self.stream_functions
        B_temp = self.A - self.M
        B_temp_subbed  = B_temp.subs(sf.c, 1)
        self.B = copy.deepcopy(B_temp_subbed)

    def create_matrix_B_non_dim(self):
        """
        Creates a non-dimensional version of matrix B by substituting specific 
        stream function parameters with their non-dimensional equivalents.

        This method first checks if matrix B is None and creates it if necessary. 
        Then, it substitutes the following parameters in matrix B to create B_non_dim:
            - rho_ambient: 1
            - rho_wake: S (stream function parameter)
            - D: 1
            - U_ambient: 1 - Lambda (stream function parameter)
            - U_wake: 1 + Lambda (stream function parameter)

        Attributes:
            B (sympy.Matrix): The original matrix B.
            B_non_dim (sympy.Matrix): The non-dimensional version of matrix B.
            stream_functions (object): An object containing stream function parameters.
        """
        sf = self.stream_functions
        if self.B == None:
            self.create_matrix_B()
        self.B_non_dim = self.B.subs({
            sf.rho_ambient: 1,
            sf.rho_wake: sf.S,
            sf.D: 1,
        })

        if sf.n_wakes == 0:
            self.B_non_dim = self.B_non_dim.subs({
                sf.delta: 1,
                sf.U_ambient: 1,
                sf.U_wake: -1,
                sf.S: 1
            })
        else:
            self.B_non_dim = self.B_non_dim.subs({
                sf.U_ambient: 1 - sf.Lambda,
                sf.U_wake: 1 + sf.Lambda
            })

    def get_latex_expressions(self):
        """
        Generate LaTeX expressions for the interfacial equations.

        This method simplifies each interfacial equation and converts it to a LaTeX
        formatted string.

        Returns:
            list of str: A list containing the LaTeX formatted strings of the 
            simplified interfacial equations.
        """
        latex_expressions = []
        for ii in range(self.n_equations):
            latex_expressions.append(
                sp.latex(sp.simplify
                         (self.interfacial_equations[ii], rational=True)))
        return latex_expressions

    def get_latex_matrix(self, mat):
        """
        Converts a given matrix to its LaTeX representation and prints it.

        Parameters:
        mat (sympy.Matrix): The matrix to be converted to LaTeX format.

        Returns:
        None
        """
        print(sp.latex(sp.simplify(mat, rational=True)))