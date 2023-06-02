import sys
sys.path.append('./')

import numpy.typing as npt
import numpy as np
import typing as tp
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import vorpy


class Section(vorpy.abs.Section):

    def __init__(self, radius: float, chord: float, omega: float, aoa_0: float) -> None:
        
        self._radius = radius
        self._omega = omega
        self._angle = 0.0
        self._time_prev = 0.0

        self._origin = np.array([0.0, self._radius, 0.0])

        self._e1 = np.array([1.0, 0.0, 0.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e3 = np.array([0.0, 0.0, 1.0])

        r = R.from_rotvec(aoa_0 * self._e2)

        self._e1 = r.apply(self._e1)
        self._e3 = r.apply(self._e3)

        self._chord = chord

        self._linear_velocity = self._omega * np.cross(self._e3, self._origin)
        self._linear_acceleration = - self._e2 * np.linalg.norm(self._linear_velocity) / (self._radius ** 2)

        self._angular_velocity = self._e3 * self._omega
        self._angular_acceleration = np.array([0.0, 0.0, 0.0])

        data = pd.read_csv('./data/NACA0012.csv')

        self._cl = interp1d(data['Alpha'].to_numpy(), data['Cl'].to_numpy())
        self._cd = interp1d(data['Alpha'].to_numpy(), data['Cd'].to_numpy())
        self._cm = interp1d(data['Alpha'].to_numpy(), data['Cm'].to_numpy())

        return

    def update(self, time: float) -> None:

        time_step = time - self._time_prev

        r = R.from_rotvec(time_step * self._omega * self._e3)

        self._origin = r.apply(self._origin)

        self._e1 = r.apply(self._e1)
        self._e2 = r.apply(self._e2)

        self._linear_velocity = self._omega * np.cross(self._e3, self._origin)
        self._linear_acceleration = - self._e2 * np.linalg.norm(self._linear_velocity) / (self._radius ** 2)

        self._time_prev = time

        return
    
    def cl(self, aoa: float) -> float:
        sign = 1.0 if aoa > 0 else -1.0
        return float(self._cl(abs(aoa)) * sign)
    
    def cd(self, aoa: float) -> float:
        return float(self._cd(abs(aoa)))
    
    def cm(self, aoa: float) -> float:
        sign = 1.0 if aoa > 0 else -1.0
        return float(self._cm(abs(aoa)) * sign)

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
    velocity[:, 2] = 10.0
    return velocity

if __name__ == '__main__':
    
    # Surface parameters
    span = 8.0
    n_span = 10
    rps = 0.5

    # Create sections
    section_1 = Section(1.0, 1.0, rps * 2 * np.pi, np.deg2rad(20))
    section_2 = Section(10.0, 0.5, rps * 2 * np.pi, 0.0)

    # Create surface
    surface = vorpy.model.Surface(section_1, section_2, n_span)

    # Run case
    meshes = vorpy.run([surface], u_call, max_interaction=100, time_step=0.05)

    # Create visualization files
    vorpy.view(meshes, './example/02/')
