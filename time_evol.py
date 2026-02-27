import numpy as np

# time evolution functions
def time_evol_split_Godunov(fx, u, dt, dx, A):
    """
    Updates one time step using operator splitting:
    Diffusion (implicit) -> Advection (Godunov)
    code snippet in numerical notes from week 6
    """
    
    """1. DIFFUSION, implicit"""
    # Solve A * fx_diff = fx
    fx_diff = np.linalg.solve(A, fx)
    
    """2. ADVECTION, Godunov"""
    # j - 1
    fx_jm1 = np.roll(fx_diff, shift=1)
    # j + 1
    fx_jp1 = np.roll(fx_diff, shift=-1)
    # eq 18
    # time evolution: subtracting based on velocity sign
    # vectorized the u
    upwind_mask = u > 0  # whether a cell has upwind velocity
    fx_sub = np.empty_like(fx)  # fill in the array with appropriatte velocities
    fx_sub[upwind_mask]  = fx[upwind_mask] - fx_jm1[upwind_mask]
    fx_sub[~upwind_mask] = fx_jp1[~upwind_mask] - fx[~upwind_mask]
    # update the new fx
    fx1 = fx_sub - (u * dt / dx) * fx_sub
    
    """Boundary conditions"""
    # ran at the end of each snapshot
    # outflow BC on first and last elements
    fx1[0] = fx1[1]
    fx1[-1] = fx1[-2]
    
    return fx1


def time_evol_split_LF(fx, u, dt, dx, A):
    """
    Updates one time step using operator splitting:
    Diffusion (implicit) -> Advection (Lax-Friedrichs)
    code snippet in numerical notes from week 6
    """
    
    """1. DIFFUSION, implicit"""
    # Solve A * fx_diff = fx
    fx_diff = np.linalg.solve(A, fx)
    
    """2. ADVECTION, Lax-Friedrichs"""
    # j - 1
    fx_jm1 = np.roll(fx_diff, shift=1)
    # j + 1
    fx_jp1 = np.roll(fx_diff, shift=-1)
    # eq 18
    # time evolution: subtracting based on velocity sign
    # vectorized the u
    fx1 = 1/2 * (fx_jp1 + fx_jm1) -\
          (u * dt) / (2 * dx) * (fx_jp1 - fx_jm1)
    
    """Boundary conditions"""
    # ran at the end of each snapshot
    # outflow BC on first and last elements
    fx1[0] = fx1[1]
    fx1[-1] = fx1[-2]
    
    return fx1