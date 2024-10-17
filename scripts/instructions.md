# Reproducing results from the paper

## Instructions

- Install all requirements using `pip install -r requirements.txt`
- Create the following folders in main directory (i.e. where this file is) : results/ and results/temp
- Run the 'simulation_script' using the following command:

 ```bash
 python simulation_script.py *args_to_follow*
 ```

Arguments to be provided in the following order:
| arg_no | description                                          |
|--------|------------------------------------------------------|
| arg1   | inlet velocity in x direction                        |
| arg2   | # of time steps                                      |
| arg3   | kinetic viscosity of the fluid                       |
| arg4   | size of the time step                                |
| arg5   | geometry of the immersed object (square,squares, wing, wing_incl or circle) |
| arg6   | # of qubits encoding points in x-axis                |
| arg7   | # of qubits encoding points in y-axis                |

For example, to reproduce the Karman Vortex Street simulation, use the following command:

```bash
python simulation_script.py 1.5 1500 3e-3 2e-3 circle 8 7
```

## Structure of the results

*UID* is a unique identifier generated for each simulation

- output: Log file to record standard messages about the progress, warnings and runtime errors.
- param_*UID*.pfile: Pickled dictionary that stores all the parameters of the simulations
- Re_*Re*_*dmrg2*_*UID*.mpssol: Pickled list of velocity components and pressure at all time steps.