import sympy as sp
import numpy as np
from scipy.linalg import eig

class SIVP:
    def __init__(self, sf, bc, max_threshold=None, val_set=None):
        self.L = bc.A_non_dim * sf.alpha 
        self.M = bc.B_non_dim
        self.L_dot = None
        self.M_dot = None
        self.L_ddot = None
        self.M_ddot = None

        self.qi_hat = sp.symbols(f'q1:{self.L.shape[0]+1}_hat')
        q_hat_mat = []
        for i in range(self.L.shape[0]):
            q_hat_mat.append(self.qi_hat[i])
        self.q_hat = sp.Matrix(q_hat_mat)

        self.q_hat_herm = self.q_hat.H
        self.J = None
        self.J11 = None
        self.J12 = None
        self.J21 = None
        self.J22 = None
        self.b = None
        self.b1 = None
        self.b2 = None
        self.val_set = val_set
        self.sf = sf
        self.bc = bc
        self.max_threshold = max_threshold

        self.set_L_M_derivs()
        self.set_Jacobian_matrix()
        self.set_b()
        #self.sub_all()

    def set_L_M_derivs(self):
        self.L_dot = self.L.diff(self.sf.alpha)
        self.M_dot = self.M.diff(self.sf.alpha)
        self.L_ddot = self.L_dot.diff(self.sf.alpha)
        self.M_ddot = self.M_dot.diff(self.sf.alpha)

    def set_Jacobian_matrix(self):
        self.J11 = self.L - self.sf.w * self.M
        self.J12 = -self.M * self.q_hat
        self.J21 = self.q_hat_herm
        self.J22 = sp.zeros(1,1)
        self.J = sp.BlockMatrix([[self.J11, self.J12], [self.J21, self.J22]])

    def set_b(self):
        self.b1 = (self.sf.w * self.M_dot - self.L_dot) * self.q_hat
        self.b2 = sp.zeros(1,1)
        self.b = sp.Matrix.vstack(self.b1, self.b2)

    def sub_all(self, val_set=None):
        if val_set is None:
            if self.val_set is None:
                raise ValueError("val_set must be provided.")
                return
            else:
                val_set = self.val_set

        self.L_sub = self.L.subs(val_set)
        self.M_sub = self.M.subs(val_set)
        self.L_dot_sub = self.L_dot.subs(val_set)
        self.M_dot_sub = self.M_dot.subs(val_set)
        self.L_ddot_sub = self.L_ddot.subs(val_set)
        self.M_ddot_sub = self.M_ddot.subs(val_set)   
        self.J11_sub = self.J11.subs(val_set)
        self.J12_sub = self.J12.subs(val_set)
        self.J21_sub = self.J21.subs(val_set)
        self.J22_sub = self.J22.subs(val_set)
        self.J_sub = self.J.subs(val_set)
        self.b1_sub = self.b1.subs(val_set)
        self.b2_sub = self.b2.subs(val_set)
        self.b_sub = self.b.subs(val_set)

    def sub_alpha(self, alpha):
        self.L_sub = self.L_sub.subs({'alpha': alpha})
        self.M_sub = self.M_sub.subs({'alpha': alpha})
        self.L_dot_sub = self.L_dot_sub.subs({'alpha': alpha})
        self.M_dot_sub = self.M_dot_sub.subs({'alpha': alpha})
        self.L_ddot_sub = self.L_ddot_sub.subs({'alpha': alpha})
        self.M_ddot_sub = self.M_ddot_sub.subs({'alpha': alpha})
        self.J11_sub = self.J11_sub.subs({'alpha': alpha})
        self.J12_sub = self.J12_sub.subs({'alpha': alpha})
        self.J21_sub = self.J21_sub.subs({'alpha': alpha})
        self.J22_sub = self.J22_sub.subs({'alpha': alpha})
        self.J_sub = self.J_sub.subs({'alpha': alpha})
        self.b1_sub = self.b1_sub.subs({'alpha': alpha})
        self.b2_sub = self.b2_sub.subs({'alpha': alpha})
        self.b_sub = self.b_sub.subs({'alpha': alpha})

    def get_ws_q_hats(self, val_set, alpha):
        L_sub1 = self.L.subs(val_set)
        M_sub1 = self.M.subs(val_set)
        L_sub2 = L_sub1.subs({'alpha': alpha})
        M_sub2 = M_sub1.subs({'alpha': alpha})
        L_num = np.array(L_sub2.evalf()).astype(np.complex128)
        M_num = np.array(M_sub2.evalf()).astype(np.complex128)
        #self.w_vals, self.q_hat_vals = eig(L_num, M_num)
        w_vals, q_hat_vals = eig(L_num, M_num)
        return w_vals, q_hat_vals

    def eval_sub(self, w, q_hat):
        m_val_set = {'w': np.squeeze(w)}
        for i in range(self.L.shape[0]):
            m_val_set[f'q{i+1}_hat'] = np.squeeze(q_hat[i])

        self.q_hat_sub = self.q_hat.subs(m_val_set)
        self.q_hat_herm_sub = self.q_hat_herm.subs(m_val_set)
        self.L_sub = self.L_sub.subs(m_val_set)
        self.M_sub = self.M_sub.subs(m_val_set)
        self.L_dot_sub = self.L_dot_sub.subs(m_val_set)
        self.M_dot_sub = self.M_dot_sub.subs(m_val_set)
        self.L_ddot_sub = self.L_ddot_sub.subs(m_val_set)
        self.M_ddot_sub = self.M_ddot_sub.subs(m_val_set)
        self.J11_sub = self.J11_sub.subs(m_val_set)
        self.J12_sub = self.J12_sub.subs(m_val_set)
        self.J21_sub = self.J21_sub.subs(m_val_set)
        self.J22_sub = self.J22_sub.subs(m_val_set)
        self.J_sub = self.J_sub.subs(m_val_set)
        self.b1_sub = self.b1_sub.subs(m_val_set)
        self.b2_sub = self.b2_sub.subs(m_val_set)
        self.b_sub = self.b_sub.subs(m_val_set)

        self.L_num = np.array(self.L_sub.evalf()).astype(np.complex128)
        self.M_num = np.array(self.M_sub.evalf()).astype(np.complex128)
        self.q_hat_num = np.array(self.q_hat_sub.evalf()).astype(np.complex128)
        self.q_hat_herm_num = np.array(self.q_hat_herm_sub.evalf()).astype(np.complex128)
        self.J_num = np.array(self.J_sub.evalf()).astype(np.complex128)
        self.b_num = np.array(self.b_sub.evalf()).astype(np.complex128)
        self.J_num_inv = np.linalg.inv(self.J_num)
        self.L_dot_num = np.array(self.L_dot_sub.evalf()).astype(np.complex128)
        self.M_dot_num = np.array(self.M_dot_sub.evalf()).astype(np.complex128)
        self.L_ddot_num = np.array(self.L_ddot_sub.evalf()).astype(np.complex128)
        self.M_ddot_num = np.array(self.M_ddot_sub.evalf()).astype(np.complex128)
    
    def eval_x2(self, x1, dk):
        x2 = np.squeeze(x1) + (self.J_num_inv @ self.b_num.ravel()) * dk
        return x2
    
    def eval_xdot(self):
        return self.J_num_inv @ self.b_num.ravel()

    def get_filtered_ws_qhats(self, ws, q_hats, max_threshold):
        filter_for_ws = [np.isfinite(ws)] and ws < max_threshold
        q_hats_filtered = []
        for i in range(q_hats.shape[1]):
            if filter_for_ws[i]:
                q_hats_filtered.append(q_hats[:,i])
        q_hats = np.array(q_hats_filtered, dtype=complex)
        q_hats = q_hats.T
        ws = ws[filter_for_ws]
        return ws, q_hats

    def two_norm_find_matching_x(self, x, ws, q_hats):
        ws, q_hats = self.get_filtered_ws_qhats(ws, q_hats, self.max_threshold)

        if q_hats.shape[0] == 0:
            print("No valid eigenvalues found.")
            return None
        
        x = np.ravel(x)
        xs = np.vstack((q_hats, ws))
        min_val = np.inf
        idx = 0
        for i in range(xs.shape[1]):
            x_i = np.ravel(xs[:,i])
            val = np.linalg.norm(x - x_i)
            if val < min_val:
                min_val = val
                idx = i
        if min_val > 10:
            print(f"Minimum value: {min_val}")
            print('xs')
            print(xs)
            print('xs[:,idx]')
            print(xs[:,idx])
            print('x')
            print(x)
            print('ws')
            print(ws)
        return xs[:,idx], min_val

    def get_matching_x(self, x1, k1, k2):
        self.sub_all(self.val_set)
        self.sub_alpha(k1)
        self.eval_sub(x1[-1], x1[:-1])
        x2 = self.eval_x2(x1, k2 - k1)
        ws, q_hats = self.get_ws_q_hats(self.val_set, k2)
        result = self.two_norm_find_matching_x(x2, ws, q_hats)
        if result is None:
            return None, None
        matching_x = result[0]
        min_val = result[1]
        return matching_x, min_val

    def get_J2_inv(self, w, qhat):
        J2_11 = self.L_num - w * self.M_num
        J2_12 = -self.M_num @ qhat.ravel()
        J2_12 = J2_12[:, np.newaxis]
        J2_21 = np.conjugate(qhat)
        J2_21 = J2_21[:, np.newaxis].transpose()
        J2_22 = np.zeros((1,1))
        J2 = np.block([[J2_11, J2_12], [J2_21, J2_22]])
        J2_inv = np.linalg.inv(J2)
        return J2_inv

    def get_b2(self, w, w_dot, qhat, qhat_dot, qhat_dot_herm):
        b2_11_term1 = 2 * ( (w * self.M_dot_num + w_dot * self.M_num - self.L_dot_num) )
        b2_11_term2 = w * self.M_ddot_num - self.L_ddot_num + 2 * w_dot * self.M_dot_num
        b2_11 = b2_11_term1 @ qhat_dot.ravel() + b2_11_term2 @ qhat.ravel()
        b2_22 = - qhat_dot_herm.ravel() @ qhat_dot.ravel()
        b2 = np.append(b2_11, b2_22)
        return b2

    def get_straight_path_k0j(self, k0j1, k0j2, step=0.1):
        path = [k0j1]
        total_dist = np.abs(k0j2 - k0j1)
        if total_dist == 0:
            return path
        num_steps = int(np.ceil(total_dist / step))
        for i in range(1, num_steps):
            frac = i / num_steps
            next_point = k0j1 + frac * (k0j2 - k0j1)
            path.append(next_point)
        path.append(k0j2)
        return path

    def get_zigzag_path_k0j(self, k0j1, k0j2, step=0.1):
        path = [k0j1]
        current = k0j1
        toggle = False
        if np.real(k0j1) > np.real(k0j2):
            inc_re_sign = -1
        else:
            inc_re_sign = 1
        if np.imag(k0j1) > np.imag(k0j2):
            inc_im_sign = -1
        else:
            inc_im_sign = 1
        num_re_inc = int(np.ceil(np.abs(np.real(k0j2) - np.real(k0j1))/step))
        num_im_inc = int(np.ceil(np.abs(np.imag(k0j2) - np.imag(k0j1))/step))
        while num_re_inc > 0 or num_im_inc > 0:
            if toggle and num_re_inc > 0:
                if np.abs(np.real(k0j2) - np.real(current)) < step:
                    next_point = np.real(k0j2) + 1j*np.imag(current)
                else:
                    next_point = current + inc_re_sign * step
                path.append(next_point)
                num_re_inc -= 1
            if not toggle and num_im_inc > 0:
                if np.abs(np.imag(k0j2) - np.imag(current)) < step:
                    next_point = np.real(current) + np.imag(k0j2)*1j
                else:
                    next_point = current + inc_im_sign * step * 1j
                path.append(next_point)
                num_im_inc -= 1
            current = next_point
            toggle = not toggle
        return path

    def pretty_print_path(self, path):
        # print path as (.,.) -> (.,.) -> etc in a single line
        path_str = ''
        for i in range(len(path) - 1):
            path_str += f'({np.real(path[i]):.2f},{np.imag(path[i]):.2f}) -> '
        path_str += f'({np.real(path[-1]):.2f},{np.imag(path[-1]):.2f})'
        print(path_str)

    def get_x_multi_SIVP(self, k0j1, k0j2, w, qhat):
        if np.abs(k0j1 - k0j2) < 1e-8:
            return w, qhat
        #Get the path to traverse from k0j1 to k0j2 using SIVP
        k0js = self.get_straight_path_k0j(k0j1, k0j2)
        self.pretty_print_path(k0js)

        # traverse the path and find the w and qhat at each step
        k_idx = 1
        k1 = k0js[0]
        x1 = np.concatenate((qhat, [w]))
        while k_idx < len(k0js):
            k2 = k0js[k_idx]
            result = self.get_matching_x(x1, k1, k2)
            if result is None:
                x2 = None
            else:
                x2 = result[0]
            if x2 is not None:
                x1 = x2
            k1 = k2
            k_idx += 1
        return x2[-1], x2[:-1]