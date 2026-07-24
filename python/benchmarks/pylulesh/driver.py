# See LICENSE.md for full license.
"""
This file contains the entry point to the code.
"""

import argparse
from domain import Domain
from lulesh import time_increment, lagrange_leapfrog
import util
import time


def parse_command_line_options() -> util.CommandLineOptions:
    """
    Parse command-line options given as arguments.

    :return: An object with the command-line options.
    """
    parser = argparse.ArgumentParser(description="LULESH in Python.")
    parser.add_argument(
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="Quiet mode - suppress all stdout",
    )
    parser.add_argument(
        "-i",
        dest="iterations",
        type=int,
        default=9999999,
        help="Number of cycles to run",
    )
    parser.add_argument(
        "-s", dest="size", type=int, default=30, help="Length of cube mesh along side"
    )
    parser.add_argument(
        "-r", dest="numregions", type=int, default=11, help="Number of distinct regions"
    )
    parser.add_argument(
        "-b",
        dest="balance",
        type=int,
        default=1,
        help="Load balance between regions of a domain",
    )
    parser.add_argument(
        "-c",
        dest="cost",
        type=int,
        default=1,
        help="Extra cost of more expensive regions",
    )
    parser.add_argument(
        "-p",
        dest="progress",
        default=False,
        action="store_true",
        help="Print out progress",
    )
    # Visualization-based arguments (-f, -v) are skipped

    args = parser.parse_args()
    return util.CommandLineOptions(
        args.iterations,
        args.size,
        args.numregions,
        args.progress,
        args.quiet,
        args.cost,
        args.balance,
    )


def main():
    """
    Driver function.
    """

    opts = parse_command_line_options()
    num_ranks = 1  # Distributed runs unsupported
    my_rank = 0

    if not opts.quiet:
        print(f"Running problem size {opts.nx}^3 per domain until completion")
        print(f"Num processors: {num_ranks}")
        print(f"Total number of elements: {num_ranks*opts.nx*opts.nx*opts.nx} \n")
        print("To run other sizes, use -s <integer>.")
        print("To run a fixed number of iterations, use -i <integer>.")
        print("To run a more or less balanced region set, use -b <integer>.")
        print("To change the relative costs of regions, use -c <integer>.")
        print("To print out progress, use -p")
        print("See help (-h) for more options\n")

    col, row, plane, side = util.init_mesh_decomposition(num_ranks, my_rank)
    domain = Domain(
        num_ranks,
        col,
        row,
        plane,
        opts.nx,
        side,
        opts.numregions,
        opts.balance,
        opts.cost,
    )

    # NOTE: If distributed, this point would contain initial domain boundary communication

    start = time.time()

    # Time iteration
    while domain.time < domain.stoptime and domain.cycle < opts.its:
        time_increment(domain)
        lagrange_leapfrog(domain)

        if opts.progress and not opts.quiet and my_rank == 0:
            print(
                f"cycle = {domain.cycle}, time = {domain.time:e}, dt={domain.deltatime:e}"
            )

    elapsed_time = time.time() - start

    if my_rank == 0 and not opts.quiet:
        util.verify_and_write_final_output(elapsed_time, domain, opts.nx, num_ranks)


if __name__ == "__main__":
    main()
