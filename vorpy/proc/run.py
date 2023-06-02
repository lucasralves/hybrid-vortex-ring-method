import typing as tp
import numpy.typing as npt
import numpy as np
import tqdm

from vorpy.model import Surface


def run(surfaces: tp.List[Surface],
        u_call: tp.Callable[[float, npt.NDArray], npt.NDArray],
        max_interaction: int = 200,
        max_tau_interaction: int = 50,
        time_step: float = 1e-1,
        tau_eps: float = 1e-10):
    
    # Initialize surfaces
    for i in range(len(surfaces)):
        surfaces[i].initial_state()

    # Create wake
    for _ in tqdm.trange(max_interaction):
        
        # Calculate circulation
        tau = np.asarray([surfaces[i].mesh.tau[:surfaces[i].n_span] for i in range(len(surfaces))])

        for _ in range(max_tau_interaction):
            
            for i in range(len(surfaces)):
                downwash = np.zeros_like(surfaces[i].mesh.p_ctrl)
                
                for j in range(len(surfaces)):
                    downwash = downwash + surfaces[j].induced_velocity(surfaces[i].mesh.p_ctrl)

                u_inf = u_call(surfaces[i].time, surfaces[i].mesh.p_ctrl)
                surfaces[i].calculate_surface_parameters(downwash, u_inf, time_step)

            new_tau = np.asarray([surfaces[i].mesh.tau[:surfaces[i].n_span] for i in range(len(surfaces))])

            if np.max(np.power(tau - new_tau, 2)) < tau_eps:
                break
            else:
                for i in range(len(surfaces)):
                    surfaces[i].mesh.tau[:surfaces[i].n_span] = 0.5 * (tau[i, :] + new_tau[i, :])
                tau[:, :] = new_tau[:, :]

        # Next state
        for i in range(len(surfaces)):

            # Wake velocity
            vel_w = u_call(surfaces[i].time, surfaces[i].mesh.vt[(surfaces[i].n_span + 1):, :])
            for j in range(len(surfaces)):
                vel_w = vel_w + surfaces[j].induced_velocity(surfaces[i].mesh.vt[(surfaces[i].n_span + 1):, :])
            
            # Save mesh to memory
            surfaces[i].save()

            # Update surface and wake
            surfaces[i].next_state(time_step, vel_w)
    
    # Store meshes
    meshes = [surfaces[i].memory for i in range(len(surfaces))]

    return meshes