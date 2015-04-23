Col_mo_spp
===
Two files for different simulations for collective motions of self-propelled particles

1. col_mo_with_ca_cp.py
The potential function is active in the decelerating range, and is normalized as a parabolic function.

2. col_mo_equ_point.py
The potential function is active in whole sensing range, and is normalized as a parabolic function with an equilibrium point.

  - The equilibrium point is (SENSING_RANGE-2*COLLISION_SHELL)/2, and leads the particle group to a quasi-lattice formation
  - The coefficient of alignment and potential force span the phase diagram and represent as follows: 
<img src="http://latex.codecogs.com/gif.latex?\theta_i(t+1)=\theta_i(t)+\arctan\frac{\alpha\sum \limits_{j\in \mathcal{N}_i(t)} v_j(t)\sin(\theta_j(t))+\beta\sum \limits_{j\in \mathcal{N}_i(t)} f_{ij}(t)\sin(\psi_j(t))}{\alpha\sum \limits_{j\in \mathcal{N}_i(t)} v_j(t)\cos(\theta_i(t))+\beta\sum \limits_{j\in \mathcal{N}_i(t)} f_{ij}(t)\cos(\psi_j(t))}" /> 
