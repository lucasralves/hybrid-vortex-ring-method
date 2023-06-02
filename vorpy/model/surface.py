import copy
import numpy.typing as npt
import numpy as np

from vorpy.abs.section import Section
from vorpy.model.mesh import Mesh
from vorpy.utils.singularities import quadrilateral_induced_velocity


class Surface:

    def __init__(self,
                 section_1: Section,
                 section_2: Section,
                 n_span: int,
                 rho: float,
                 refinement_type: str = 'none',
                 refinement_coef: float = 0.0) -> None:
        
        assert refinement_type in ['none', 'left', 'right', 'both']
        
        self.section_1 = section_1
        self.section_2 = section_2
        self.n_span = n_span
        self.rho = rho
        self.refinement_type = refinement_type
        self.refinement_coef = refinement_coef

        self.mesh = None        # Mesh class
        self.time = None        # simulation time
        self.memory = None      # stores the previous meshes

        return
    
    def initial_state(self) -> None:

        self.memory = []
        self.mesh = Mesh(self.n_span, self.rho, self.refinement_type, self.refinement_coef)
        self.time = 0.0

        self.section_1.update(self.time)
        self.section_2.update(self.time)

        self.mesh.initial_state(self.section_1, self.section_2)

        return
    
    def next_state(self, time_step: float, vel_w: npt.NDArray) -> None:
        self.time += time_step
        self.section_1.update(self.time)
        self.section_2.update(self.time)
        self.mesh.next_state(time_step, vel_w, self.section_1, self.section_2)
        return
    
    def save(self) -> None:
        self.memory.append({'time': self.time, 'mesh': copy.deepcopy(self.mesh)})
        return

    def induced_velocity(self, x: npt.NDArray) -> npt.NDArray:
        vel = np.zeros_like(x)
        for i in range(self.mesh.fc.shape[0]):
            vel = vel + quadrilateral_induced_velocity(self.mesh.vt[self.mesh.fc[i, 0], :], self.mesh.vt[self.mesh.fc[i, 1], :], self.mesh.vt[self.mesh.fc[i, 2], :], self.mesh.vt[self.mesh.fc[i, 3], :], self.mesh.tau[i], x)
        return vel
    
    def calculate_surface_parameters(self, downwash: npt.NDArray, u_inf: npt.NDArray, time_step: float) -> None:
        self.mesh.calculate_surface_parameters(downwash, u_inf, time_step)
        return
