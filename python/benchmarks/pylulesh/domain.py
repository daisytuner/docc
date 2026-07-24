# See LICENSE.md for full license.
"""
File containing the primary data structure used in LULESH.
"""
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

from constants import ZETA, ETA, XI

# Type hints for numpy arrays
IndexT = np.int32
RealT = np.float64
intarr = npt.NDArray[IndexT]
realarr = npt.NDArray[RealT]


@dataclass
class Domain:
    """
    The Domain class hosts the primary data structure for the simulation.
    """

    # Array fields
    reg_elem_size: intarr  #: Number of elements in each region (1D)
    reg_elem_list: intarr  #: Region elements (2D)
    reg_num_list: intarr  #: Region number per domain element (1D)
    ss: realarr  #: Sound speed (1D)
    elem_mass: realarr  #: Mass (1D)
    arealg: realarr  #: Element characteristic length (1D)

    x: realarr  #: x coordinate (1D)
    y: realarr  #: y coordinate (1D)
    z: realarr  #: z coordinate (1D)

    xd: realarr  #: x velocity (1D)
    yd: realarr  #: y velocity (1D)
    zd: realarr  #: z velocity (1D)

    xdd: realarr  #: x acceleration (1D)
    ydd: realarr  #: y acceleration (1D)
    zdd: realarr  #: z acceleration (1D)

    fx: realarr  #: x forces (1D)
    fy: realarr  #: y forces (1D)
    fz: realarr  #: z forces (1D)

    nodal_mass: realarr  #: Mass (1D)

    symm_x: intarr  #: Symmetry plane nodeset (1D)
    symm_y: intarr  #: Symmetry plane nodeset (1D)
    symm_z: intarr  #: Symmetry plane nodeset (1D)

    # Volumes
    v: realarr  #: Relative volume (1D)
    volo: realarr  #: Reference volume (1D)
    delv: realarr  #: m_vnew - m_v (1D)
    vdov: realarr  #: Volume derivative over volume (1D)

    e: realarr  #: Energy (1D)

    # Pressure components
    p: realarr  #: Pressure (1D)
    q: realarr  #: Artificial viscosity (1D)
    ql: realarr  #: Linear term for q (1D)
    qq: realarr  #: Quadratic term for q (1D)

    # Temporary arrays
    vnew: realarr  #: New relative volume (1D)

    # Temporary principal strains
    dxx: realarr  #: x principal strains (1D)
    dyy: realarr  #: y principal strains (1D)
    dzz: realarr  #: z principal strains (1D)

    # Temporary gradients
    delv_xi: realarr  #: Velocity gradient across xi (1D)
    delv_eta: realarr  #: Velocity gradient across eta (1D)
    delv_zeta: realarr  #: Velocity gradient across zeta (1D)
    delx_xi: realarr  #: Coordinate gradient across xi (1D)
    delx_eta: realarr  #: Coordinate gradient across xi (1D)
    delx_zeta: realarr  #: Coordinate gradient across xi (1D)

    # Region information
    nodelist: intarr  #: Element-to-node connectivity (2D, numelem x 8)
    lxim: intarr  #: Element connectivity across face (xi, m)
    lxip: intarr  #: Element connectivity across face (xi, p)
    letam: intarr  #: Element connectivity across face (eta, m)
    letap: intarr  #: Element connectivity across face (eta, p)
    lzetam: intarr  #: Element connectivity across face (zeta, m)
    lzetap: intarr  #: Element connectivity across face (zeta, p)
    elem_bc: intarr  #: Symmetry/free-surface flags for each element face (1D)

    # Parameters
    numregions: int  #: Number of regions
    numelem: int  #: Number of elements
    numnode: int  #: Number of nodes
    cost: int  #: Imbalance cost

    # Distributed parameters
    num_ranks: int  #: Number of ranks
    tp: int  #: Test processors (or number of ranks per side)
    col_loc: int  #: Column location
    row_loc: int  #: Row location
    plane_loc: int  #: Plane location

    # Time-related fields
    cycle: int  #: Current iteration
    time: float  #: Current time
    stoptime: float  #: End time for simulation
    deltatime: float  #: Variable time increment
    dtcourant: float  #: Courant constraint
    dthydro: float  #: Volume change constraint
    dtfixed: float  #: Fixed time increment
    deltatimemultlb: float  #: Lower bound for time increment
    deltatimemultub: float  #: Upper bound for time increment
    dtmax: float  #: Maximum allowable time increment

    # Constants
    qqc: float  #: Quadratic term coefficient for q
    dvovmax: float  #: Maximum allowable volume change
    qstop: float  #: Excessive q indicator

    # Cutoff constants
    e_cut: float  #: Energy tolerance
    p_cut: float  #: Pressure tolerance
    q_cut: float  #: q tolerance
    u_cut: float  #: Velocity tolerance
    v_cut: float  #: Relative volume tolerance

    # Other constants
    hgcoef: float  #: Hourglass control

    ss4o3: float  #: 4/3
    monoq_max_slope: float  #: Monotonic q maximum slope
    monoq_limiter_mult: float  #: Monotonic q limiter multiplier
    qlc_monoq: float  #: Linear term coefficient for q
    qqc_monoq: float  #: Quadratic term coefficient for q

    eos_vmin: float  #: Equation of state relative volume lower bound
    eos_vmax: float  #: Equation of state relative volume upper bound
    refdens: float  #: Reference density
    pmin: float  #: Pressure floor
    emin: float  #: Energy floor

    # Constructor
    def __init__(
        self,
        num_ranks: int,
        col_loc: IndexT,
        row_loc: IndexT,
        plane_loc: IndexT,
        nx: IndexT,
        tp: int,
        nr: int,
        balance: int,
        cost: int,
    ):
        # Initialize constants
        self.u_cut = self.q_cut = self.p_cut = self.e_cut = 1e-7
        self.v_cut = 1e-10
        self.hgcoef = 3.0
        self.ss4o3 = 4 / 3
        self.qstop = 1e12
        self.monoq_max_slope = 1.0
        self.monoq_limiter_mult = 2.0
        self.qlc_monoq = 0.5
        self.qqc_monoq = 2 / 3
        self.qqc = 2.0
        self.eos_vmax = 1e9
        self.eos_vmin = 1e-9
        self.pmin = 0
        self.emin = -1e15
        self.dvovmax = 0.1
        self.refdens = 1.0

        # Initialize parameters
        edge_elems = nx
        edge_nodes = edge_elems + 1
        self.cost = cost
        self.num_ranks = num_ranks
        self.tp = tp
        self.col_loc = col_loc
        self.row_loc = row_loc
        self.plane_loc = plane_loc

        # Initialize Sedov mesh
        self.numelem = int(edge_elems**3)
        self.numnode = int(edge_nodes**3)

        # Material indexset
        self.reg_num_list = np.zeros(self.numelem, dtype=IndexT)

        # Initialize element-centered fields
        self.allocate_elem_persistent()
        # Initialize node-centered fields
        self.allocate_node_persistent()

        self.setup_comm_buffers(edge_nodes)

        # Basic field initialization
        self.e = np.zeros([self.numelem], RealT)
        self.p = np.zeros([self.numelem], RealT)
        self.q = np.zeros([self.numelem], RealT)
        self.ss = np.zeros([self.numelem], RealT)
        self.v = np.ones([self.numelem], RealT)
        self.xd = np.zeros([self.numnode], RealT)
        self.yd = np.zeros([self.numnode], RealT)
        self.zd = np.zeros([self.numnode], RealT)
        self.xdd = np.zeros([self.numnode], RealT)
        self.ydd = np.zeros([self.numnode], RealT)
        self.zdd = np.zeros([self.numnode], RealT)
        self.nodal_mass = np.zeros([self.numnode], RealT)

        # Principal strain rates (per element). Pre-declared here so the fields
        # are part of the fixed struct layout for the docc native compiler;
        # they are (re)computed in calc_kinematics_for_elems.
        self.dxx = np.zeros([self.numelem], RealT)
        self.dyy = np.zeros([self.numelem], RealT)
        self.dzz = np.zeros([self.numelem], RealT)

        # Velocity/coordinate gradients across the element faces (per element).
        # Pre-declared here so the fields are part of the fixed struct layout for
        # the docc native compiler; they are (re)computed in
        # calc_monotonic_q_gradients_for_elems.
        self.delv_xi = np.zeros([self.numelem], RealT)
        self.delv_eta = np.zeros([self.numelem], RealT)
        self.delv_zeta = np.zeros([self.numelem], RealT)
        self.delx_xi = np.zeros([self.numelem], RealT)
        self.delx_eta = np.zeros([self.numelem], RealT)
        self.delx_zeta = np.zeros([self.numelem], RealT)

        self.build_mesh(nx, edge_nodes, edge_elems)
        self.create_region_index_sets(nr, balance)
        self.setup_symmetry_planes(edge_nodes)
        self.setup_element_connectivities(edge_elems)
        self.setup_boundary_conditions(edge_elems)

        # Setup defaults
        # NOTE from original implementation:
        # These can be changed (requires recompile) if you want to run
        # with a fixed timestep, or to a different end time, but it's
        # probably easier/better to just run a fixed number of timesteps
        # using the -i flag in 2.x
        self.dtfixed = -1e-6  # Negative means use courant condition
        self.stoptime = 1e-2  # * edge_elems*tp/45
        self.deltatimemultlb = 1.1
        self.deltatimemultub = 1.2
        self.dtcourant = 1e20
        self.dthydro = 1e20
        self.dtmax = 1e-2
        self.time = 0.0
        self.cycle = 0

        # Initialize field data
        from lulesh import calc_elem_volume  # Avoid cyclic import

        x_local = self.x[self.nodelist]
        y_local = self.y[self.nodelist]
        z_local = self.z[self.nodelist]
        volume = calc_elem_volume(x_local, y_local, z_local)
        self.volo[:] = volume
        self.elem_mass[:] = volume
        for i in range(8):
            nodelist = self.nodelist[:, i]
            self.nodal_mass[nodelist] += volume / 8

        # Deposit initial energy
        # NOTE from original implementation:
        # An energy of 3.948746e+7 is correct for a problem with
        # 45 zones along a side - we need to scale it
        ebase = 3.948746e7
        scale = (nx * tp) / 45
        einit = ebase * scale * scale * scale

        # Dump into the first zone (which we know is in the corner)
        # of the domain that sits at the origin
        if (row_loc + col_loc + plane_loc) == 0:
            self.e[0] = einit

        # Set initial dt based on analytic CFL calculation
        self.deltatime = (0.5 * np.cbrt(self.volo[0])) / np.sqrt(2 * einit)

    def allocate_elem_persistent(self):
        n = self.numelem
        self.nodelist = np.empty([n, 8], dtype=IndexT)

        self.lxim = np.empty([n], dtype=IndexT)
        self.lxip = np.empty([n], dtype=IndexT)
        self.letam = np.empty([n], dtype=IndexT)
        self.letap = np.empty([n], dtype=IndexT)
        self.lzetam = np.empty([n], dtype=IndexT)
        self.lzetap = np.empty([n], dtype=IndexT)
        self.elem_bc = np.empty([n], dtype=IndexT)

        self.e = np.empty([n], dtype=RealT)
        self.p = np.empty([n], dtype=RealT)

        self.q = np.empty([n], dtype=RealT)
        self.ql = np.empty([n], dtype=RealT)
        self.qq = np.empty([n], dtype=RealT)

        self.v = np.empty([n], dtype=RealT)
        self.volo = np.empty([n], dtype=RealT)
        self.delv = np.empty([n], dtype=RealT)
        self.vdov = np.empty([n], dtype=RealT)

        self.arealg = np.empty([n], dtype=RealT)
        self.ss = np.empty([n], dtype=RealT)
        self.elem_mass = np.empty([n], dtype=RealT)
        self.vnew = np.empty([n], dtype=RealT)

    def allocate_node_persistent(self):
        n = self.numnode

        self.x = np.empty([n], dtype=RealT)
        self.y = np.empty([n], dtype=RealT)
        self.z = np.empty([n], dtype=RealT)
        self.xd = np.empty([n], dtype=RealT)
        self.yd = np.empty([n], dtype=RealT)
        self.zd = np.empty([n], dtype=RealT)
        self.xdd = np.empty([n], dtype=RealT)
        self.ydd = np.empty([n], dtype=RealT)
        self.zdd = np.empty([n], dtype=RealT)
        self.fx = np.empty([n], dtype=RealT)
        self.fy = np.empty([n], dtype=RealT)
        self.fz = np.empty([n], dtype=RealT)

        self.nodal_mass = np.empty([n], dtype=RealT)

    def build_mesh(self, nx: int, edge_nodes: int, edge_elems: int):
        mesh_edge_elems = self.tp * nx

        # Initialize nodal coordinates
        # TODO: Can/should be done faster with arange/meshgrid
        nidx = 0
        for plane in range(edge_nodes):
            for row in range(edge_nodes):
                for col in range(edge_nodes):
                    tz = 1.125 * (self.plane_loc * nx + plane) / mesh_edge_elems
                    ty = 1.125 * (self.row_loc * nx + row) / mesh_edge_elems
                    tx = 1.125 * (self.col_loc * nx + col) / mesh_edge_elems
                    self.x[nidx] = tx
                    self.y[nidx] = ty
                    self.z[nidx] = tz
                    nidx += 1

        # Embed hexehedral elements in nodal point lattice
        # TODO: Can/should be done faster with vectorize
        zidx = 0
        nidx = 0
        for plane in range(edge_elems):
            for row in range(edge_elems):
                for col in range(edge_elems):
                    # yapf: disable
                    self.nodelist[zidx, 0] = nidx
                    self.nodelist[zidx, 1] = nidx                                      + 1
                    self.nodelist[zidx, 2] = nidx                         + edge_nodes + 1
                    self.nodelist[zidx, 3] = nidx                         + edge_nodes
                    self.nodelist[zidx, 4] = nidx + edge_nodes*edge_nodes
                    self.nodelist[zidx, 5] = nidx + edge_nodes*edge_nodes              + 1
                    self.nodelist[zidx, 6] = nidx + edge_nodes*edge_nodes + edge_nodes + 1
                    self.nodelist[zidx, 7] = nidx + edge_nodes*edge_nodes + edge_nodes
                    # yapf: enable
                    zidx += 1
                    nidx += 1
                nidx += 1
            nidx += edge_nodes

    def setup_comm_buffers(self, edge_nodes: int):
        # NOTE: In a distributed version, commmunication buffers would be set here

        # Boundary nodesets
        if self.col_loc == 0:
            self.symm_x = np.empty(edge_nodes * edge_nodes, dtype=IndexT)
        if self.row_loc == 0:
            self.symm_y = np.empty(edge_nodes * edge_nodes, dtype=IndexT)
        if self.plane_loc == 0:
            self.symm_z = np.empty(edge_nodes * edge_nodes, dtype=IndexT)

    def create_region_index_sets(self, nr: int, balance: int, my_rank: int = 0):
        # NOTE: This is not equivalent to srand(0) found in the original code,
        #       but achieves the same effect
        np.random.seed(1 + my_rank)

        self.numregions = nr
        self.reg_elem_size = np.empty(nr, dtype=IndexT)
        next_index = 0

        # Fill out the reg_num_list with material numbers, which are always
        # the region index plus one
        if nr == 1:
            # If we only have one region, just fill it
            self.reg_num_list[:] = 1
            self.reg_elem_size[0] = 0
        else:
            region_num = -1
            last_reg = -1
            runto = 0
            reg_bin_end = np.empty(self.numregions, dtype=int)

            # Determine the relative weights of all the regions.
            # This is based on the -b flag (balance).
            self.reg_elem_size[:] = 0
            # Total sum of all region weights
            region_weights = np.arange(1, self.numregions + 1) ** balance
            reg_bin_end = np.cumsum(region_weights)
            cost_denominator = np.int64(np.sum(region_weights))

            # Until all elements are assigned
            while next_index < self.numelem:
                # Make sure we don't pick the same region twice in a row
                while region_num == last_reg:
                    # Pick the region
                    region_var = np.random.randint(0, cost_denominator)
                    i = 0
                    while region_var >= reg_bin_end[i]:
                        i += 1

                    # NOTE from original implementation:
                    # rotate the regions based on MPI rank.
                    # Rotation is Rank % NumRegions this makes each domain have a
                    # different region with the highest representation
                    region_num = ((i + my_rank) % self.numregions) + 1

                bin_size = np.random.randint(0, 1000)
                if bin_size < 773:
                    elements = np.random.randint(1, 16)
                elif bin_size < 937:
                    elements = np.random.randint(16, 32)
                elif bin_size < 970:
                    elements = np.random.randint(32, 64)
                elif bin_size < 974:
                    elements = np.random.randint(64, 128)
                elif bin_size < 978:
                    elements = np.random.randint(128, 256)
                elif bin_size < 981:
                    elements = np.random.randint(256, 512)
                else:
                    elements = np.random.randint(512, 2049)
                runto = elements + next_index

                # Store the elements. If we hit the end before we run out of
                # elements then just stop.
                while next_index < runto and next_index < self.numelem:
                    self.reg_num_list[next_index] = region_num
                    next_index += 1
                last_reg = region_num

        # Convert reg_num_list to region index sets
        # First, count size of each region
        for i in range(self.numelem):
            r = self.reg_num_list[i] - 1  # Region index == regnum-1
            self.reg_elem_size[r] += 1

        # Second, allocate each region index set
        maxreg = np.max(self.reg_elem_size)
        self.reg_elem_list = np.zeros((nr, maxreg), dtype=IndexT)
        self.reg_elem_size[:] = 0

        # Third, fill index sets
        for i in range(self.numelem):
            r = self.reg_num_list[i] - 1  # Region index == regnum-1
            regndx = self.reg_elem_size[r]
            self.reg_elem_size[r] += 1
            self.reg_elem_list[r, regndx] = i

    def setup_symmetry_planes(self, edge_nodes: int):
        nidx = 0
        for i in range(edge_nodes):
            for j in range(edge_nodes):
                if self.plane_loc == 0:
                    self.symm_z[nidx] = i * edge_nodes + j
                if self.row_loc == 0:
                    self.symm_y[nidx] = i * edge_nodes * edge_nodes + j
                if self.col_loc == 0:
                    self.symm_x[nidx] = i * edge_nodes * edge_nodes + j * edge_nodes
                nidx += 1

    def setup_element_connectivities(self, edge_elems: int):
        self.lxim[0] = 0
        self.lxim[1 : self.numelem] = np.arange(self.numelem - 1, dtype=IndexT)
        self.lxip[0 : self.numelem - 1] = np.arange(1, self.numelem, dtype=IndexT)
        self.lxip[self.numelem - 1] = self.numelem - 1

        self.letam[:edge_elems] = np.arange(edge_elems, dtype=IndexT)
        self.letap[self.numelem - edge_elems :] = np.arange(
            self.numelem - edge_elems, self.numelem, dtype=IndexT
        )
        self.letam[edge_elems:] = np.arange(self.numelem - edge_elems, dtype=IndexT)
        self.letap[: self.numelem - edge_elems] = np.arange(
            edge_elems, self.numelem, dtype=IndexT
        )

        self.lzetam[: edge_elems * edge_elems] = np.arange(
            edge_elems * edge_elems, dtype=IndexT
        )
        self.lzetap[self.numelem - edge_elems * edge_elems :] = np.arange(
            self.numelem - edge_elems * edge_elems, self.numelem, dtype=IndexT
        )
        self.lzetam[edge_elems * edge_elems :] = np.arange(
            self.numelem - edge_elems * edge_elems, dtype=IndexT
        )
        self.lzetap[: self.numelem - edge_elems * edge_elems] = np.arange(
            edge_elems * edge_elems, self.numelem, dtype=IndexT
        )

    def setup_boundary_conditions(self, edge_elems):
        # Fill with INT_MIN
        ghost_idx = np.full((6,), np.iinfo(IndexT).min, dtype=IndexT)

        self.elem_bc[:] = 0

        # assume communication to 6 neighbors by default
        row_min = self.row_loc != 0
        row_max = self.row_loc != (self.tp - 1)
        col_min = self.col_loc != 0
        col_max = self.col_loc != (self.tp - 1)
        plane_min = self.plane_loc != 0
        plane_max = self.plane_loc != (self.tp - 1)

        pidx = self.numelem
        for i, cond in enumerate(
            (plane_min, plane_max, row_min, row_max, col_min, col_max)
        ):
            if cond:
                ghost_idx[i] = pidx
                pidx += edge_elems * edge_elems

        # TODO(later): Vectorizing can be more efficient
        # TODO(later): This can also be refactored as a loop
        for i in range(edge_elems):
            plane_inc = i * edge_elems * edge_elems
            row_inc = i * edge_elems
            for j in range(edge_elems):
                if not plane_min:
                    self.elem_bc[row_inc + j] |= ZETA["M"]["SYMM"]
                else:
                    self.elem_bc[row_inc + j] |= ZETA["M"]["COMM"]
                    self.lzetam[row_inc + j] = ghost_idx[0] + row_inc + j

                if not plane_max:
                    self.elem_bc[
                        row_inc + j + self.numelem - edge_elems * edge_elems
                    ] |= ZETA["P"]["FREE"]
                else:
                    self.elem_bc[
                        row_inc + j + self.numelem - edge_elems * edge_elems
                    ] |= ZETA["P"]["COMM"]
                    self.lzetap[
                        row_inc + j + self.numelem - edge_elems * edge_elems
                    ] = (ghost_idx[1] + row_inc + j)

                if not row_min:
                    self.elem_bc[plane_inc + j] |= ETA["M"]["SYMM"]
                else:
                    self.elem_bc[plane_inc + j] |= ETA["M"]["COMM"]
                    self.letam[plane_inc + j] = ghost_idx[2] + row_inc + j

                if not row_max:
                    self.elem_bc[
                        plane_inc + j + edge_elems * edge_elems - edge_elems
                    ] |= ETA["P"]["FREE"]
                else:
                    self.elem_bc[
                        plane_inc + j + edge_elems * edge_elems - edge_elems
                    ] |= ETA["P"]["COMM"]
                    self.letap[plane_inc + j + edge_elems * edge_elems - edge_elems] = (
                        ghost_idx[3] + row_inc + j
                    )

                if not col_min:
                    self.elem_bc[plane_inc + j * edge_elems] |= XI["M"]["SYMM"]
                else:
                    self.elem_bc[plane_inc + j * edge_elems] |= XI["M"]["COMM"]
                    self.lxim[plane_inc + j * edge_elems] = ghost_idx[4] + row_inc + j

                if not col_max:
                    self.elem_bc[plane_inc + j * edge_elems + edge_elems - 1] |= XI[
                        "P"
                    ]["FREE"]
                else:
                    self.elem_bc[plane_inc + j * edge_elems + edge_elems - 1] |= XI[
                        "P"
                    ]["COMM"]
                    self.lxip[plane_inc + j * edge_elems + edge_elems - 1] = (
                        ghost_idx[5] + row_inc + j
                    )
