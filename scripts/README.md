# Scripts

## Mask generation

## Time evolution

- Run the 'simulation_script' using the following command:

 ```bash
 python simulations.py *args_to_follow*
 ```

Arguments to be provided in the following order:

| arg_no | description                                          |
|--------|------------------------------------------------------|
| arg1   | inlet velocity in x direction                        |
| arg2   | # of time steps                                      |
| arg3   | kinetic viscosity of the fluid                       |
| arg4   | size of the time step                                |
| arg5   | name of mps mask (from data/mps_masks/*name*.mps)    |
| arg6   | # of qubits encoding points in x-axis                |
| arg7   | # of qubits encoding points in y-axis                |

For example, to reproduce the Karman Vortex Street simulation, use the following command:

```bash
python simulations.py 1.5 100 3e-3 2e-3 circle 8 7
```

## Structure of the results

*UID* is a unique identifier generated for each simulation

- output: Log file to record standard messages about the progress, warnings and runtime errors.
- params_*UID*.pkl: Pickled dictionary that stores all the parameters of the simulations
- results_*UID*.pkl: Pickled list of velocity components and pressure at all time steps.

## Post-processing

## Best practices
