import numpy as np
import numpy.typing as npt


# Constants
FOUR_PI = 4 * np.pi
EPS = 1e-6

# Functions
# def segment_line_induced_velocity(pi: npt.NDArray, pf: npt.NDArray, tau: float, p: npt.NDArray) -> npt.NDArray:
#     """Calculate the velocity induced by a line segment"""

#     r1 = p - pi
#     r1_norm = np.linalg.norm(r1)

#     if r1_norm < EPS:
#         return np.zeros(3, dtype=np.double)

#     r2 = p - pf
#     r2_norm = np.linalg.norm(r2)

#     if r2_norm < EPS:
#         return np.zeros(3, dtype=np.double)

#     r0 = r1 - r2

#     r1_x_r2 = np.cross(r1, r2)
#     r1_x_r2_norm = np.linalg.norm(r1_x_r2)

#     if r1_x_r2_norm < EPS:
#         return np.zeros(3, dtype=np.double)

#     vel = (tau / FOUR_PI) * (r1_x_r2 / r1_x_r2_norm) * np.dot(r0, r1 / r1_norm - r2 / r2_norm)

#     return vel

def segment_line_induced_velocity(pi: npt.NDArray, pf: npt.NDArray, tau: float, p: npt.NDArray) -> npt.NDArray:
    """Calculate the velocity induced by a line segment"""

    # r1
    r1 = np.empty_like(p)
    r1[:, 0], r1[:, 1], r1[:, 2] = p[:, 0] - pi[0], p[:, 1] - pi[1], p[:, 2] - pi[2]
    r1_norm = np.linalg.norm(r1, axis=1)
    check_r1 = r1_norm < EPS

    # r2
    r2 = np.empty_like(p)
    r2[:, 0], r2[:, 1], r2[:, 2] = p[:, 0] - pf[0], p[:, 1] - pf[1], p[:, 2] - pf[2]
    r2_norm = np.linalg.norm(r2, axis=1)
    check_r2 = r2_norm < EPS

    # r0
    r0 = pf - pi

    # r1_x_r2
    r1_x_r2 = np.cross(r1, r2, axis=1)
    r1_x_r2_norm = np.linalg.norm(r1_x_r2, axis=1)
    check_r1_x_r2 = r1_x_r2_norm < EPS

    # vel
    vel = np.zeros_like(p)
    check = check_r1 | check_r2 | check_r1_x_r2
    
    k = (tau / FOUR_PI) * (r0[0] * (r1[check == False, 0] / r1_norm[check == False] - r2[check == False, 0] / r2_norm[check == False]) + r0[1] * (r1[check == False, 1] / r1_norm[check == False] - r2[check == False, 1] / r2_norm[check == False]) + r0[2] * (r1[check == False, 2] / r1_norm[check == False] - r2[check == False, 2] / r2_norm[check == False])) / r1_x_r2_norm[check == False]

    vel[check == False, 0] = k * r1_x_r2[check == False, 0]
    vel[check == False, 1] = k * r1_x_r2[check == False, 1]
    vel[check == False, 2] = k * r1_x_r2[check == False, 2]
    
    return vel

def quadrilateral_induced_velocity(p1: npt.NDArray, p2: npt.NDArray, p3: npt.NDArray, p4: npt.NDArray, tau: float, p: npt.NDArray) -> npt.NDArray:
    """Calculate the velocity induced by a quadrangular panel"""

    vel1 = segment_line_induced_velocity(p1, p2, tau, p)
    vel2 = segment_line_induced_velocity(p2, p3, tau, p)
    vel3 = segment_line_induced_velocity(p3, p4, tau, p)
    vel4 = segment_line_induced_velocity(p4, p1, tau, p)

    vel = vel1 + vel2 + vel3 + vel4

    return vel

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    pi = np.array([0.0, 0.0, 0.0])
    pf = np.array([1.0, 0.0, 0.0])

    tau = 1.0

    n = 100

    theta = np.random.random(n) * 2 * np.pi
    radius = 1.0 + 0.5 * np.random.random(n)

    p = np.zeros((n, 3))
    p[:, 0] = 0.5
    p[:, 1] = np.sin(theta) * radius
    p[:, 2] = np.cos(theta) * radius

    vel = segment_line_induced_velocity(pi, pf, tau, p)

    plt.figure()
    plt.scatter(p[:, 1], p[:, 2])
    plt.quiver(p[:, 1], p[:, 2], vel[:, 1], vel[:, 2])
    # plt.contourf(p[:, 1], p[:, 2], vel[:, 0])
    plt.axis('equal')
    plt.grid()
    plt.show()
    