import sys
sys.path.append('./')

import numpy.typing as npt
import numpy as np
import typing as tp
import pandas as pd
from scipy.interpolate import interp1d

import vorpy


class Section(vorpy.abs.Section):

    def __init__(self, side: int, span: float) -> None:
        
        self._span = span
        self._side = side

        self._period = 1.0

        self._e1 = np.array([1.0, 0.0, 0.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e3 = np.array([0.0, 0.0, 1.0])
        self._chord = 1.0
        self._angular_velocity = np.array([0.0, 0.0, 0.0])
        self._angular_acceleration = np.array([0.0, 0.0, 0.0])

        data = pd.read_csv('./data/NACA0012.csv')

        self._cl = interp1d(data['Alpha'].to_numpy(), data['Cl'].to_numpy())
        self._cd = interp1d(data['Alpha'].to_numpy(), data['Cd'].to_numpy())
        self._cm = interp1d(data['Alpha'].to_numpy(), data['Cm'].to_numpy())

        return

    def update(self, time: float) -> None:
        self._origin = np.array([0.0, 0.5 * self._side * self._span, np.sin(time * 2 * np.pi / self._period)])
        self._linear_velocity = np.array([0.0, 0.0, (2 * np.pi / self._period) * np.cos(time * 2 * np.pi / self._period)])
        self._linear_acceleration = np.array([0.0, 0.0, - ((2 * np.pi / self._period) ** 2) * np.sin(time * 2 * np.pi / self._period)])
        return
    
    def cl(self, aoa: npt.NDArray) -> npt.NDArray:
        return self._cl(np.abs(aoa)) * np.sign(aoa)
    
    def cd(self, aoa: npt.NDArray) -> npt.NDArray:
        return self._cd(np.abs(aoa))
    
    def cm(self, aoa: npt.NDArray) -> npt.NDArray:
        return self._cm(np.abs(aoa)) * np.sign(aoa)

    @property
    def origin(self) -> npt.NDArray:
        return self._origin

    @property
    def e1(self) -> npt.NDArray:
        return self._e1
    
    @property
    def e2(self) -> npt.NDArray:
        return self._e2
    
    @property
    def e3(self) -> npt.NDArray:
        return self._e3
    
    @property
    def chord(self) -> float:
        return self._chord
    
    @property
    def linear_velocity(self) -> npt.NDArray:
        return self._linear_velocity
    
    @property
    def linear_acceleration(self) -> npt.NDArray:
        return self._linear_acceleration
    
    @property
    def angular_velocity(self) -> npt.NDArray:
        return self._angular_velocity
    
    @property
    def angular_acceleration(self) -> npt.NDArray:
        return self._angular_acceleration

def u_call(time: float, x: npt.NDArray) -> tp.List:
    velocity = np.zeros_like(x)
    velocity[:, 0] = 10.0
    return velocity

if __name__ == '__main__':
    
    # Environment
    rho = 1.225

    # Surface parameters
    span = 8.0
    n_span = 50
    
    # Create sections
    section_1 = Section(-1, span)
    section_2 = Section(1, span)

    # Create surface
    surface = vorpy.model.Surface(section_1, section_2, n_span, rho, refinement_type='both', refinement_coef=1.0)

    # Run case
    meshes = vorpy.run([surface], u_call, max_interaction=10, time_step=0.05)

    # Create visualization files
    vorpy.view(meshes, './example/01/')
