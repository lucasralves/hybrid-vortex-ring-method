import numpy as np
import numpy.typing as npt
import typing as tp
from functools import partial
from scipy.spatial.transform import Rotation as R

from vorpy.abs.section import Section


def _create_rotation_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.array([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])

def _coeff_wrapper(cl_1: tp.Callable[[npt.NDArray], npt.NDArray], cl_2: tp.Callable[[npt.NDArray], npt.NDArray], factor: float, alpha: npt.NDArray) -> float:
    return cl_1(alpha) * (1 - factor) + cl_2(alpha) * factor

class Mesh:

    def __init__(self,
                 n_span: int,
                 rho: float,
                 refinement_type: str,
                 refinement_coef: float) -> None:

        self.n_span = n_span
        self.rho = rho

        self.vt = None          # vertices
        self.fc = None          # faces
        self.n_wake = None      # number of faces in wake direction
        self.tau = None         # circulation
        self.p_ctrl = None      # control points
        self.e1 = None          # base system
        self.e2 = None          # base system
        self.e3 = None          # base system
        self.vel_sec = None     # section velocity
        self.vel = None         # total velocity
        self.omega = None       # section angular velocity
        self.chord = None       # section chord
        self.alpha = None       # attack angle
        self.alpha_eff_1 = None # attack angle
        self.alpha_eff_2 = None # attack angle
        self.cl_call = None     # lift coefficient
        self.cd_call = None     # drag coefficeint
        self.cm_call = None     # moment coefficient
        self.cl = None          # lift coefficient
        self.cd = None          # drag coefficeint
        self.cm = None          # moment coefficient
        self.dl = None          # section lenght
        self.L = None           # lift
        self.D = None           # drag
        self.N = None           # non-steady
        self.M = None           # moment
        self.tau_prev = None    # previous circulation

        if refinement_type == 'none':
            
            self.interp_params = np.linspace(0.0, 1.0, num=n_span + 1)

        else:

            if refinement_type == 'both':
                theta_min, theta_max = 0.0, 2 * np.pi
            elif refinement_type == 'left':
                theta_min, theta_max = 0.0, np.pi / 2
            elif refinement_type == 'right':
                theta_min, theta_max = np.pi / 2, 2 * np.pi
            
            theta = np.linspace(theta_min, theta_max, num=self.n_span + 2)
            f = 1 - np.cos(theta) + refinement_coef

            interp_params = [0]
            
            for i in range(self.n_span):
                interp_params.append(interp_params[i] + f[i+1])
            
            interp_params = np.asarray(interp_params)
            self.interp_params = interp_params / np.max(interp_params)
        
        return

    def initial_state(self,
                      section_1: Section,
                      section_2: Section) -> None:

        self.vt = np.empty((2 * (self.n_span + 1), 3))
        self.fc = np.empty((self.n_span, 4), dtype=np.int64)
        self.n_span = self.n_span
        self.n_wake = 0
        self.tau = 1.0 * np.ones(self.n_span)
        self.p_ctrl = np.empty((self.n_span, 3))
        self.e1 = np.empty((self.n_span, 3))
        self.e2 = np.empty((self.n_span, 3))
        self.e3 = np.empty((self.n_span, 3))
        self.vel_sec = np.empty((self.n_span, 3))
        self.vel = np.empty((self.n_span, 3))
        self.omega = np.empty((self.n_span, 3))
        self.chord = np.empty(self.n_span)
        self.alpha = np.empty(self.n_span)
        self.alpha_eff_1 = np.empty(self.n_span)
        self.alpha_eff_2 = np.empty(self.n_span)
        self.cl_call = [partial(_coeff_wrapper, section_1.cl, section_2.cl, self.interp_params[i]) for i in range(self.n_span)]
        self.cd_call = [partial(_coeff_wrapper, section_1.cd, section_2.cd, self.interp_params[i]) for i in range(self.n_span)]
        self.cm_call = [partial(_coeff_wrapper, section_1.cm, section_2.cm, self.interp_params[i]) for i in range(self.n_span)]
        self.cl = np.empty(self.n_span)
        self.cd = np.empty(self.n_span)
        self.cm = np.empty(self.n_span)
        self.dl = np.empty(self.n_span)
        self.L = np.empty((self.n_span, 3))
        self.D = np.empty((self.n_span, 3))
        self.N = np.empty((self.n_span, 3))
        self.M = np.empty((self.n_span, 3))
        self.tau_prev = np.zeros(self.n_span)

        self.vt[:, :] = np.asarray([section_1.origin * (1 - factor) + section_2.origin * factor for factor in self.interp_params] + [(section_1.origin + section_1.chord * section_1.e1) * (1 - factor) + (section_2.origin + section_2.chord * section_2.e1) * factor for factor in self.interp_params])
        self.fc[:, :] = np.asarray([[i, 1 + i, self.n_span + 2 + i, self.n_span + 1 + i] for i in range(self.n_span)], dtype=np.int64)
        self.p_ctrl[:, :] = np.asarray([0.5 * (self.vt[i, :] + self.vt[i + 1, :]) for i in range(self.n_span)])

        rot_m_1 = _create_rotation_matrix(section_1.e1, section_1.e2, section_1.e3)
        rot_m_2 = _create_rotation_matrix(section_2.e1, section_2.e2, section_2.e3)

        rot_m_12 = np.dot(rot_m_2, np.transpose(rot_m_1))
        rot_12 = R.from_matrix(rot_m_12)
        vec = rot_12.as_rotvec()
        max_angle = np.linalg.norm(vec)
        if max_angle < 1e-6:
            axis = np.array([1., 0., 0.])
        else:
            axis = vec / np.linalg.norm(vec)

        for i in range(self.n_span):
            interp = 0.5 * (self.interp_params[i] + self.interp_params[i + 1])
            angle = max_angle * interp
            r = R.from_rotvec(angle * axis)
            self.e1[i, :] = r.apply(section_1.e1)
            self.e2[i, :] = r.apply(section_1.e2)
            self.e3[i, :] = r.apply(section_1.e3)
            self.vel_sec[i, :] = section_1.linear_velocity * (1 - interp) + section_2.linear_velocity * interp
            self.omega[i, :] = section_1.angular_velocity * (1 - interp) + section_2.angular_velocity * interp
            self.chord[i] = section_1.chord * (1 - interp) + section_2.chord * interp

        self.dl = np.asarray([np.linalg.norm(self.vt[i + 1, :] - self.vt[i, :]) for i in range(self.n_span)])

        return
    
    def next_state(self,
                   time_step: float,
                   vel_w: npt.NDArray,
                   section_1: Section,
                   section_2: Section) -> None:
        
        self.vt = np.concatenate([self.vt, np.empty((self.n_span + 1, 3))], axis=0)
        self.vt[2 * (self.n_span + 1):, :] = self.vt[(self.n_span + 1):-(self.n_span + 1), :] + time_step * vel_w
        self.vt[:2 * (self.n_span + 1), :] = np.asarray([section_1.origin * (1 - factor) + section_2.origin * factor for factor in self.interp_params] + [(section_1.origin + section_1.chord * section_1.e1) * (1 - factor) + (section_2.origin + section_2.chord * section_2.e1) * factor for factor in self.interp_params])

        self.fc = np.concatenate([self.fc, np.array([[(2 + self.n_wake) * (self.n_span + 1) + i, (1 + self.n_wake) * (self.n_span + 1) + i, (1 + self.n_wake) * (self.n_span + 1) + i + 1, (2 + self.n_wake) * (self.n_span + 1) + i + 1] for i in range(self.n_span)])], axis=0)

        self.tau = np.concatenate([self.tau[:self.n_span], self.tau])

        self.p_ctrl[:, :] = np.asarray([0.5 * (self.vt[i, :] + self.vt[i + 1, :]) for i in range(self.n_span)])

        self.tau_prev[:] = self.tau[:self.n_span]

        self.n_wake += 1

        return

    def calculate_surface_parameters(self, downwash: npt.NDArray, u_inf: npt.NDArray, time_step: float) -> None:
        
        # Velocity
        aux = np.cross(self.omega, self.e1, axis=1)
        self.vel[:, 0] = u_inf[:, 0] - (self.vel_sec[:, 0] + 0.5 * self.chord * aux[:, 0]) + downwash[:, 0]
        self.vel[:, 1] = u_inf[:, 1] - (self.vel_sec[:, 1] + 0.5 * self.chord * aux[:, 1]) + downwash[:, 1]
        self.vel[:, 2] = u_inf[:, 2] - (self.vel_sec[:, 2] + 0.5 * self.chord * aux[:, 2]) + downwash[:, 2]

        # Alpha
        self.alpha[:] = np.arctan((self.vel[:, 0] * self.e3[:, 0] + self.vel[:, 1] * self.e3[:, 1] + self.vel[:, 2] * self.e3[:, 2]) / (self.vel[:, 0] * self.e1[:, 0] + self.vel[:, 1] * self.e1[:, 1] + self.vel[:, 2] * self.e1[:, 2]))
        alpha_dot = np.zeros_like(self.alpha)
        
        k = np.empty(self.n_span)
        k[alpha_dot >= 0] = 1.0
        k[alpha_dot < 0] = 0.75

        sign = np.empty(self.n_span)
        sign[alpha_dot >= 0] = 1.0
        sign[alpha_dot < 0] = -1.0

        self.alpha_eff_1[:] = self.alpha - k * 1.0 * np.sqrt( self.chord * np.abs(alpha_dot) / (0.5 * np.linalg.norm(self.vel, axis=1)) ) * sign
        self.alpha_eff_2[:] = self.alpha - k * 0.5 * np.sqrt( self.chord * np.abs(alpha_dot) / (0.5 * np.linalg.norm(self.vel, axis=1)) ) * sign
        
        # Coefficients
        for i in range(self.n_span):
            self.cl[i] = self.cl_call[i](self.alpha_eff_1[i])
            self.cd[i] = self.cd_call[i](self.alpha_eff_1[i])
            self.cm[i] = self.cm_call[i](self.alpha_eff_2[i])

        # Forces
        k = 0.5 * self.rho * np.linalg.norm(self.vel) * self.chord * self.dl

        tau_dot = (self.tau[:self.n_span] - self.tau_prev) / time_step

        aux = np.cross(self.vel, self.e2, axis=1)
        self.L[:, 0] = k * aux[:, 0] * self.cl
        self.L[:, 1] = k * aux[:, 1] * self.cl
        self.L[:, 2] = k * aux[:, 2] * self.cl

        self.D[:, 0] = k * self.vel[:, 0] * self.cd
        self.D[:, 1] = k * self.vel[:, 1] * self.cd
        self.D[:, 2] = k * self.vel[:, 2] * self.cd

        aux = np.cross(self.omega, self.e3, axis=1)
        self.N[:, 0] = self.rho * self.chord * (tau_dot * self.e3[:, 0] + self.tau[:self.n_span] * aux[:, 0]) * self.dl
        self.N[:, 1] = self.rho * self.chord * (tau_dot * self.e3[:, 1] + self.tau[:self.n_span] * aux[:, 1]) * self.dl
        self.N[:, 2] = self.rho * self.chord * (tau_dot * self.e3[:, 2] + self.tau[:self.n_span] * aux[:, 2]) * self.dl

        aux = np.cross(self.e1, self.N, axis=1)
        self.M[:, 0] = k * np.linalg.norm(self.vel) * self.chord * self.cm * self.e2[:, 0] + 0.5 * self.chord * aux[:, 0]
        self.M[:, 1] = k * np.linalg.norm(self.vel) * self.chord * self.cm * self.e2[:, 1] + 0.5 * self.chord * aux[:, 1]
        self.M[:, 2] = k * np.linalg.norm(self.vel) * self.chord * self.cm * self.e2[:, 2] + 0.5 * self.chord * aux[:, 2]

        # Circulation
        self.tau[:self.n_span] = np.linalg.norm(self.L, axis=1) / (self.rho * np.linalg.norm(self.vel, axis=1))

        return