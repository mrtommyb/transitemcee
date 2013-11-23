transitemcee
============

Some python code to perform MCMC transit fits. Its not really for consumption by others

To run, first compile the fortran code

```bash
f2py -c --opt='-O3'  tmodtom.f -m tmodtom -L/usr/lib/  -lmpi
f2py -c --opt='-O3'  claret4ppy.f -m claret4ppy -L/usr/lib/  -lmpi 
f2py -c --opt='-O3'  claretquadpy.f -m claretquadpy -L/usr/lib/  -lmpi 

```
