# Python Port of the Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics (LULESH)

This is a port of LULESH 2.0 using Python and NumPy.

The project currently does not include distributed (MPI) processing, explicit
multi-threading (it is implicitly done in NumPy), or visualization.

The original LULESH code can be found at https://github.com/LLNL/LULESH

## Usage

Call the main function in the driver file as follows:

```sh
$ python driver.py
```

Command-line options can be displayed with the `-h` flag. See the original code
for more information about the arguments.

## License

The port is distributed under the New BSD license, similarly to the original
project.

The original code is licensed as follows: Copyright (c) 2010-2013. Lawrence
Livermore National Security, LLC. Produced at the Lawrence Livermore National
Laboratory. LLNL-CODE-461231. All rights reserved.

## Contributing

Any pull requests or issues are welcome!
