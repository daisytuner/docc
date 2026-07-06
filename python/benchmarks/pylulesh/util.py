# See LICENSE.md for full license.
"""
Various utilities and helper functions.
"""

from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from domain import Domain


@dataclass
class CommandLineOptions:
    """
    LULESH command-line options.
    """

    its: int  #: Iterations
    nx: int  #: Problem size
    numregions: int  #: Number of regions
    progress: bool  #: Print progress
    quiet: bool  #: Suppress all outputs
    cost: int  #: Extra cost of expensive regions
    balance: int  #: Load balancing


# Exception classes
class QStopError(Exception):
    pass


class VolumeError(Exception):
    pass


def verify_and_write_final_output(
    elapsed_time: float, domain: "Domain", nx: int, num_ranks: int = 1
):
    """
    Print summary of runtime and outputs.
    """
    nx8 = nx
    grind_time_1 = ((elapsed_time * 1e6) / domain.cycle) / (nx8 * nx8 * nx8)
    grind_time_2 = ((elapsed_time * 1e6) / domain.cycle) / (nx8 * nx8 * nx8 * num_ranks)

    print("Run completed:")
    print(f"   Problem size        =  {nx}")
    print(f"   MPI tasks           =  {num_ranks}")
    print(f"   Iteration count     =  {domain.cycle}")
    print(f"   Final Origin Energy =  {domain.e[0]:12.6e}")

    max_abs_diff = 0.0
    total_abs_diff = 0.0
    max_rel_diff = 0.0

    for j in range(nx):
        for k in range(j + 1, nx):
            absdiff = abs(domain.e[j * nx + k] - domain.e[k * nx + j])
            total_abs_diff += absdiff

            if max_abs_diff < absdiff:
                max_abs_diff = absdiff

            if (
                domain.e[k * nx + j] != 0.0
            ):  # Note: Added to Python implementation to avoid warnings
                reldiff = absdiff / domain.e[k * nx + j]

                if max_rel_diff < reldiff:
                    max_rel_diff = reldiff

    # Quick symmetry check
    print("   Testing Plane 0 of Energy Array on rank 0:")
    print(f"        MaxAbsDiff   = {max_abs_diff:.6e}")
    print(f"        TotalAbsDiff = {total_abs_diff:.6e}")
    print(f"        MaxRelDiff   = {max_rel_diff:.6e}")

    # Timing information
    print()
    print(f"Elapsed time         = {elapsed_time:10.2f} (s)")
    print(
        f"Grind time (us/z/c)  = {grind_time_1:10.8f} (per dom)"
        f"  ({elapsed_time:10.8f} overall)"
    )
    print(f"FOM                  = {(1000/grind_time_2):10.8f} (z/s)")
    print()


def init_mesh_decomposition(num_ranks: int, my_rank: int) -> Tuple[int, int, int, int]:
    """
    Initializes domain decomposition for the input mesh.

    :param num_ranks: Total number of ranks.
    :param my_rank: Rank of caller.
    :return: A 4-tuple of (column, row, plane, side) of this rank's submesh.
    :note: Used for distributed execution.
    """
    test_procs = int(np.cbrt(num_ranks) + 0.5)

    if test_procs**3 != num_ranks:
        raise ValueError("Num processors must be a cube of an integer (1, 8, 27, ...)")

    dx = dy = dz = test_procs
    remainder = dx * dy * dz % num_ranks
    if my_rank < remainder:
        my_dom = my_rank * (1 + (dx * dy * dz / num_ranks))
    else:
        my_dom = remainder * (1 + (dx * dy * dz / num_ranks)) + (
            my_rank - remainder
        ) * (dx * dy * dz / num_ranks)

    col = my_dom % dx
    row = (my_dom / dx) % dy
    plane = my_dom / (dx * dy)
    side = test_procs
    return col, row, plane, side
