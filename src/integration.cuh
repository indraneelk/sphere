#ifndef INTEGRATION_CUH_
#define INTEGRATION_CUH_

// integration.cuh
// Functions responsible for temporal integration

//// Choose temporal integration scheme. Uncomment only one!
//#define EULER
//#define TY2
#define TY3

__global__ void updateFixvel(const Float4* __restrict__ dev_x_sorted, const Float4* __restrict__ dev_walls_nx, Float4* __restrict__ dev_vel_sorted) 
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id
    if (idx < devC_np && devC_nw > 0) { // Condition prevents block size error
       __syncthreads();
       const Float4 x = dev_x_sorted[idx];
       const Float4 walls_nx = dev_walls_nx[0];
       Float4 vel = dev_vel_sorted[idx];
       if (abs(x.z-walls_nx.w) < 3.0*x.w) {
       	  vel.w = 1.0;
       }
       else if (x.z > 0.5*walls_nx.w) {
	  vel.w = 0.0;
       }
       __syncthreads();
       dev_vel_sorted[idx] = vel;
   }    
}														    






// Second order integration scheme based on Taylor expansion of particle kinematics. 
// Kernel executed on device, and callable from host only.
__global__ void integrate(
    const Float4* __restrict__ dev_x_sorted,
    const Float4* __restrict__ dev_vel_sorted,
    const Float4* __restrict__ dev_angvel_sorted,
    Float4* __restrict__ dev_x,
    Float4* __restrict__ dev_vel,
    Float4* __restrict__ dev_angvel,
    const Float4* __restrict__ dev_force,
    const Float4* __restrict__ dev_torque,
    Float4* __restrict__ dev_angpos,
    Float4* __restrict__ dev_acc,
    Float4* __restrict__ dev_angacc,
    Float4* __restrict__ dev_vel0,
    Float4* __restrict__ dev_angvel0,
    Float4* __restrict__ dev_xyzsum,
    const unsigned int* __restrict__ dev_gridParticleIndex,
    const unsigned int iter,
    const int* __restrict__ dev_walls_wmode,
    const Float4* __restrict__ dev_walls_mvfd,
    const Float* __restrict__ dev_walls_tau_eff_x_partial,
    const Float* __restrict__ dev_walls_tau_x,
    const Float tau_x,
    const int change_velocity_state, // 1: v *= vel_fac, -1: v /= vel_fac
    const Float velocity_factor,
    const unsigned int blocksPerGrid)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_np) { // Condition prevents block size error

        // Copy data to temporary arrays to avoid any potential
        // read-after-write, write-after-read, or write-after-write hazards. 
        __syncthreads();
        unsigned int orig_idx = dev_gridParticleIndex[idx];
        const Float4 force    = dev_force[orig_idx];
        const Float4 torque   = dev_torque[orig_idx];
        const Float4 angpos   = dev_angpos[orig_idx];
        const Float4 x        = dev_x_sorted[idx];
        const Float4 vel      = dev_vel_sorted[idx];
        const Float4 angvel   = dev_angvel_sorted[idx];
        Float4 xyzsum = dev_xyzsum[orig_idx];

        // Read the mode of the top wall
        int wall0mode;
        Float wall0mass;
        if (devC_nw > 0) {
            wall0mode = dev_walls_wmode[0];
            wall0mass = dev_walls_mvfd[0].x;
        }

        // Find the final sum of shear stresses on the top particles
        Float tau_eff_x = 0.0;
        if (devC_nw > 0 && wall0mode == 3)
            for (int i=0; i<blocksPerGrid; ++i)
                tau_eff_x += dev_walls_tau_eff_x_partial[i];

        // Get old accelerations for three-term Taylor expansion. These values
        // don't exist in the first time step
#ifdef TY3
        Float4 acc0, angacc0;
        if (iter == 0) {
            acc0 = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
            angacc0 = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        } else {
            __syncthreads();
            acc0    = dev_acc[orig_idx];
            angacc0 = dev_angacc[orig_idx];
        }
#endif

        const Float radius = x.w;

        // New values
        Float4 x_new, vel_new, angpos_new, angvel_new;

        // Coherent read from constant memory to registers
        const Float dt = devC_dt;
        const Float3 origo = MAKE_FLOAT3(
            devC_grid.origo[0],
            devC_grid.origo[1],
            devC_grid.origo[2]); 
        const Float3 L = MAKE_FLOAT3(
            devC_grid.L[0],
            devC_grid.L[1],
            devC_grid.L[2]);

        // Particle mass
        Float m = 4.0/3.0 * PI * radius*radius*radius * devC_params.rho;

        // Find the acceleration by Newton's second law
        Float4 acc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        acc.x = force.x/m + devC_params.g[0];
        acc.y = force.y/m + devC_params.g[1];
        acc.z = force.z/m + devC_params.g[2];

        // Find the angular acceleration by Newton's second law
        // (angacc = (total moment)/Intertia, intertia = 2/5*m*r^2)
        Float4 angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        angacc.x = torque.x * 1.0 / (2.0/5.0 * m * radius*radius);
        angacc.y = torque.y * 1.0 / (2.0/5.0 * m * radius*radius);
        angacc.z = torque.z * 1.0 / (2.0/5.0 * m * radius*radius);

        // Fixed shear stress BC
        if (wall0mode == 3 && vel.w > 0.0001 && x.z > devC_grid.L[2]*0.5) {

            // the force should be positive when abs(tau_x) > abs(tau_eff_x)
            const Float f_tau_x =
                (tau_x + tau_eff_x)*devC_grid.L[0]*devC_grid.L[1];

            acc.x = f_tau_x/wall0mass;  // acceleration = force/mass
            acc.y = 0.0;
            acc.z -= devC_params.g[2];

            // disable rotation
            angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        }

        // Modify the acceleration if the particle is marked as having a fixed
        // velocity. In that case, zero the horizontal acceleration so that
	// the particles move uniformly forizontally. For the z-direction,
        // there is no restriction on acceleration to account for dilation.
        else if (vel.w > 0.0001 && vel.w < 10.0) {

            acc.x = 0.0;
            acc.y = 0.0;
            // acc.z -= devC_params.g[2];

            // Zero the angular acceleration
            angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        }

        // 2d kinematic update by disabling linear acceleration along y and
        // disabling angular acceleration by everything but y.
        if (vel.w >= 10.0) {
            acc.y = 0.0;
            angacc.x = 0.0;
            angacc.z = 0.0;
        }

        if (vel.w < -0.0001) {
            acc.x = 0.0;
            acc.y = 0.0;
            acc.z = 0.0;

            // Zero the angular acceleration
            angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        }

#ifdef EULER
        // Forward Euler
        // Truncation error O(dt^2) for positions and velocities
        x_new.x = x.x + vel.x*dt;
        x_new.y = x.y + vel.y*dt;
        x_new.z = x.z + vel.z*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt;
        vel_new.y = vel.y + acc.y*dt;
        vel_new.z = vel.z + acc.z*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt;
        angpos_new.y = angpos.y + angvel.y*dt;
        angpos_new.z = angpos.z + angvel.z*dt;
        angpos_new.w = angpos.w;

        angvel_new.x = angvel.x + angacc.x*dt;
        angvel_new.y = angvel.y + angacc.y*dt;
        angvel_new.z = angvel.z + angacc.z*dt;
        angvel_new.w = angvel.w;

        // Add horizontal-displacement for this time step to the sum of
        // horizontal displacements
        xyzsum.x += vel.x*dt;
        xyzsum.y += vel.y*dt;
        xyzsum.z += vel.z*dt;
#endif

#ifdef TY2
        // Two-term Taylor expansion (TY2)
        // Truncation error O(dt^3) for positions, O(dt^2) for velocities
        x_new.x = x.x + vel.x*dt + 0.5*acc.x*dt*dt;
        x_new.y = x.y + vel.y*dt + 0.5*acc.y*dt*dt;
        x_new.z = x.z + vel.z*dt + 0.5*acc.z*dt*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt;
        vel_new.y = vel.y + acc.y*dt;
        vel_new.z = vel.z + acc.z*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt + 0.5*angacc.x*dt*dt;
        angpos_new.y = angpos.y + angvel.y*dt + 0.5*angacc.y*dt*dt;
        angpos_new.z = angpos.z + angvel.z*dt + 0.5*angacc.z*dt*dt;
        angpos_new.w = angpos.w;

        angvel_new.x = angvel.x + angacc.x*dt;
        angvel_new.y = angvel.y + angacc.y*dt;
        angvel_new.z = angvel.z + angacc.z*dt;
        angvel_new.w = angvel.w;

        // Add horizontal-displacement for this time step to the sum of
        // horizontal displacements
        xyzsum.x += vel.x*dt + 0.5*acc.x*dt*dt;
        xyzsum.y += vel.y*dt + 0.5*acc.y*dt*dt;
        xyzsum.z += vel.z*dt + 0.5*acc.z*dt*dt;
#endif

#ifdef TY3
        // Three-term Taylor expansion (TY3)
        // Truncation error O(dt^4) for positions, O(dt^3) for velocities
        // Approximate acceleration change by backwards difference:
        const Float3 dacc_dt = MAKE_FLOAT3(
            (acc.x - acc0.x)/dt,
            (acc.y - acc0.y)/dt,
            (acc.z - acc0.z)/dt);

        const Float3 dangacc_dt = MAKE_FLOAT3(
            (angacc.x - angacc0.x)/dt,
            (angacc.y - angacc0.y)/dt,
            (angacc.z - angacc0.z)/dt);

        x_new.x = x.x + vel.x*dt + 0.5*acc.x*dt*dt + 1.0/6.0*dacc_dt.x*dt*dt*dt;
        x_new.y = x.y + vel.y*dt + 0.5*acc.y*dt*dt + 1.0/6.0*dacc_dt.y*dt*dt*dt;
        x_new.z = x.z + vel.z*dt + 0.5*acc.z*dt*dt + 1.0/6.0*dacc_dt.z*dt*dt*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt + 0.5*dacc_dt.x*dt*dt;
        vel_new.y = vel.y + acc.y*dt + 0.5*dacc_dt.y*dt*dt;
        vel_new.z = vel.z + acc.z*dt + 0.5*dacc_dt.z*dt*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt + 0.5*angacc.x*dt*dt
            + 1.0/6.0*dangacc_dt.x*dt*dt*dt;
        angpos_new.y = angpos.y + angvel.y*dt + 0.5*angacc.y*dt*dt
            + 1.0/6.0*dangacc_dt.y*dt*dt*dt;
        angpos_new.z = angpos.z + angvel.z*dt + 0.5*angacc.z*dt*dt
            + 1.0/6.0*dangacc_dt.z*dt*dt*dt;
        angpos_new.w = angpos.w;

        angvel_new.x = angvel.x + angacc.x*dt + 0.5*dangacc_dt.x*dt*dt;
        angvel_new.y = angvel.y + angacc.y*dt + 0.5*dangacc_dt.y*dt*dt;
        angvel_new.z = angvel.z + angacc.z*dt + 0.5*dangacc_dt.z*dt*dt;
        angvel_new.w = angvel.w;

        // Add horizontal-displacement for this time step to the sum of
        // horizontal displacements
        xyzsum.x += vel.x*dt + 0.5*acc.x*dt*dt + 1.0/6.0*dacc_dt.x*dt*dt*dt;
        xyzsum.y += vel.y*dt + 0.5*acc.y*dt*dt + 1.0/6.0*dacc_dt.y*dt*dt*dt;
        xyzsum.z += vel.z*dt + 0.5*acc.z*dt*dt + 1.0/6.0*dacc_dt.z*dt*dt*dt;
#endif

        // Move particles outside the domain across periodic boundaries
        if (devC_grid.periodic == 1) {
            if (x_new.x < origo.x)
                x_new.x += L.x;
            if (x_new.x > L.x)
                x_new.x -= L.x;
            if (ND == 3) {
                if (x_new.y < origo.y)
                    x_new.y += L.y;
                if (x_new.y > L.y)
                    x_new.y -= L.y;
            }
        } else if (devC_grid.periodic == 2) {
            if (x_new.x < origo.x)
                x_new.x += L.x;
            if (x_new.x > L.x)
                x_new.x -= L.x;
        }

        // step-wise velocity change for rate-and-state experiments
        if (vel.w > 5.0 && vel.w < 10.0) {
            if (change_velocity_state == 1)
                vel_new.x *= velocity_factor;
            else if (change_velocity_state == -1)
                vel_new.x /= velocity_factor;
        }

        // Hold threads for coalesced write
        __syncthreads();

        // Store data in global memory at original, pre-sort positions
        dev_xyzsum[orig_idx]  = xyzsum;
        dev_acc[orig_idx]     = acc;
        dev_angacc[orig_idx]  = angacc;
        dev_angvel[orig_idx]  = angvel_new;
        dev_vel[orig_idx]     = vel_new;
        dev_angpos[orig_idx]  = angpos_new;
        dev_x[orig_idx]       = x_new;
    } 
} // End of integrate(...)


// Reduce wall force contributions from particles to a single value per wall
__global__ void summation(
    const Float* __restrict__ in,
    Float* __restrict__ out)
{
    __shared__ Float cache[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int cacheIdx = threadIdx.x;

    Float temp = 0.0f;
    while (idx < devC_np) {
        temp += in[idx];
        idx += blockDim.x * gridDim.x;
    }

    // Set the cache values
    cache[cacheIdx] = temp;

    __syncthreads();

    // For reductions, threadsPerBlock must be a power of two
    // because of the following code
    unsigned int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    // Write sum for block to global memory
    if (cacheIdx == 0)
        out[blockIdx.x] = cache[0];
}

// Update wall positions
__global__ void integrateWalls(
    Float4* __restrict__ dev_walls_nx,
    Float4* __restrict__ dev_walls_mvfd,
    const int* __restrict__ dev_walls_wmode,
    const Float* __restrict__ dev_walls_force_partial,
    Float* __restrict__ dev_walls_acc,
    const unsigned int blocksPerGrid,
    const Float t_current,
    const unsigned int iter)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_nw) { // Condition prevents block size error

        // Copy data to temporary arrays to avoid any potential
        // read-after-write, write-after-read, or write-after-write hazards. 
        Float4 w_nx   = dev_walls_nx[idx];
        Float4 w_mvfd = dev_walls_mvfd[idx];
        int wmode = dev_walls_wmode[idx];  // Wall BC, 0: fixed, 1: sigma0, 2: vel

        if (wmode != 0) { // wmode == 0: Wall fixed: do nothing

#ifdef TY3
            Float acc0;
            if (iter == 0)
                acc0 = 0.0;
            else
                acc0 = dev_walls_acc[idx];
#endif

            // Find the final sum of forces on wall
            w_mvfd.z = 0.0;
            for (int i=0; i<blocksPerGrid; ++i) {
                w_mvfd.z += dev_walls_force_partial[i];
            }

            const Float dt = devC_dt;

            // Apply sinusoidal variation if specified
            const Float sigma_0 = w_mvfd.w
                + devC_params.sigma0_A*sin(2.0*PI*devC_params.sigma0_f * t_current);

            // Normal load = Normal stress times wall surface area,
            // directed downwards.
            const Float N = -sigma_0*devC_grid.L[0]*devC_grid.L[1];

            // Calculate resulting acceleration of wall
            // (Wall mass is stored in x component of position Float4)
            Float acc = (w_mvfd.z + N)/w_mvfd.x;

            // Apply gravity
            /*if (idx == 0)
                acc += devC_params.g[2];
            else if (idx == 1)
                acc += devC_params.g[0];
            else if (idx == 2)
                acc += devC_params.g[1];*/

            // If Wall BC is controlled by velocity, it should not change
            if (wmode == 2) { 
                acc = 0.0;
            }

#ifdef EULER
            // Forward Euler tempmoral integration.

            // Update position
            w_nx.w += w_mvfd.y*dt;

            // Update velocity
            w_mvfd.y += acc*dt;
#endif

#ifdef TY2
            // Two-term Taylor expansion for tempmoral integration.
            // The truncation error is O(dt^3) for positions and O(dt^2) for
            // velocities.

            // Update position
            w_nx.w += w_mvfd.y*dt + 0.5*acc*dt*dt;

            // Update velocity
            w_mvfd.y += acc*dt;
#endif

#ifdef TY3
            // Three-term Taylor expansion for tempmoral integration.
            // The truncation error is O(dt^4) for positions and O(dt^3) for
            // velocities. The acceleration change approximated by backwards
            // central difference:
            const Float dacc_dt = (acc - acc0)/dt;

            // Update position
            w_nx.w += w_mvfd.y*dt + 0.5*acc*dt*dt + 1.0/6.0*dacc_dt*dt*dt*dt;

            // Update velocity
            w_mvfd.y += acc*dt + 0.5*dacc_dt*dt*dt;
#endif

            // Store data in global memory
            __syncthreads();
            dev_walls_nx[idx]   = w_nx;
            dev_walls_mvfd[idx] = w_mvfd;
            dev_walls_acc[idx]  = acc;
        }
    }
} // End of integrateWalls(...)


// Finds shear stress per particle adjacent to top wall (idx=0).
// The fixvel value is saved in vel.w.
// Particles who do not fulfill the criteria will have a value of 0.0 written to
// dev_walls_tau_eff_x_pp.
__global__ void findShearStressOnFixedMovingParticles(
    const Float4* __restrict__ dev_x,
    const Float4* __restrict__ dev_vel,
    const Float4* __restrict__ dev_force,
    Float* __restrict__ dev_walls_tau_eff_x_pp)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_np) { // Condition prevents block size error

        // Copy data to temporary arrays to avoid any potential
        // read-after-write, write-after-read, or write-after-write hazards. 
        __syncthreads();
        const Float z       = dev_x[idx].z;
        const Float fixvel  = dev_vel[idx].w;
        const Float force_x = dev_force[idx].x;

        // Only select fixed velocity (fixvel > 0.0, fixvel = vel.w) particles
        // at the top boundary (z > L[0]/2)
        Float f_x = 0.0;
        if (fixvel > 0.0 && z > devC_grid.L[2]*0.5)
            f_x = force_x;

        __syncthreads();
        // Convert force to shear stress and save
        dev_walls_tau_eff_x_pp[idx] = f_x/(devC_grid.L[0]*devC_grid.L[1]);
    }
}
#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
