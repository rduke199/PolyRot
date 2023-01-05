# PolyRot
This repo contains various tools for manipulating and rotating polymers. The tools fall into three 
categories: 
* `chain_dimensions`: a tools for analytically estimating chain dimensions using the dihedral potential 
energy surfaces. 
* `central_dihedral`: tools for finding and rotating the central dihedral angle of a polymer (adapted from
https://doi.org/10.1021/ma500923r)
* `polymer_data`: a module for calculating polymer properties from a collection of master `json` files 

# Installation 
This module can be installed with `pip install` by running the following code:
```bash
pip install git+https://github.com/rduke199/PolyRot.git
```

# Modules 
## Chain Dimensions

### Generating polymers

The first step in using this tool is collecting the necessary data.  
```python
DIHED_ROT = [(0, 0), (10, 0.293835), (20, 0.967939), (30, 1.86645),
                 (40, 3.177), (50, 4.91823), (60, 6.91593), (70, 8.91844), (80, 10.5465),
                 (90, 11.4081), (100, 11.2003), (110, 10.0682), (120, 8.36782), (130, 6.51926),
                 (140, 5.079767), (150, 4.36963), (160, 4.04486), (170, 3.80474), (180, 3.82803)]
L_RING = 2.548  # length of the ring tangent
L_BOND = 1.480  # length of the inter-moiety bond
DEFLECTION = 15  # degrees
```
The PolymerRotate class requires four types of data, each one a list: 
1. List of ring lengths (`float`)
2. List of bond lengths (`float`)
3. List of deflection angles lengths (`float`), in degrees if `theta_degrees` is True (default)
4. List of dihedral angles' potential energy surfaces (`list` of `tuples` where each tuple has the format 
(`degree`, `energy`)), in degrees if `theta_degrees` is True (default)

With the specified data, the user can define a polymer object. 
```python
from PolyRot.chain_dimensions import PolymerRotate

polymer = PolymerRotate(bond_lengths=[L_RING, L_RING], ring_lengths=[L_BOND, L_BOND],
                        deflection_angles=[DEFLECTION, -DEFLECTION, -DEFLECTION, DEFLECTION],
                        dihed_energies=[DIHED_ROT, DIHED_ROT])
```

One a polymer object has been defined, the rest becomes very straightforward. Here we generate a polymer with 
50 rings. The `std_chain` function always builds a polymer with alternating angles. We then use the `draw_chain` 
function to draw the resulting chain in two dimensions.  
```python
from PolyRot.chain_dimensions import draw_chain

ch = polymer.std_chain(50)
draw_chain(ch, dim3=False)
```

Next we can randomly rotate the dihedral angles for the polymer. Each angle is rotated randomly, but weighted
according to the potential energy surface given when defining the polymer. 
We also plot the rotated polymer, in three dimensions this time. 
```python
new_ch = polymer.rotated_chain(50)
draw_chain(new_ch, dim3=True)
```

### Analysis: predicting chain dimensions 

To estimate the chain dimensions for our polymer, we first generate 10,000 iterations of the randomly rotated
polymer 
```python
polymers_many = multi_polymer(polymer, n=50, num_poly=100)
```

Next, we find the tangent-tangent correlation of the first ring with each subsequent ring. For a polymer
chain with any stiffness, the tangent-tangent correlation should be correlated (often linearly) with the 
distance of a ring from the first ring. 
We can view results of tangent-tangent correlation function by specifying `plot=Ture`. 
The slope of the tangent-tangent correlation vs distance estimates the persistence length (N<sub>p</sub>). 
```python
from PolyRot.chain_dimensions import n_p

Np = n_p(polymers_many, plot=True)
print("Np: ", Np)
```

Finally we can estimate the mean square end-to-end distance (R<sup>2</sup>) for our polymer as shown below. 
```python
from PolyRot.chain_dimensions import avg_r_2

r_2 = avg_r_2(polymers_many, in_nm_2=True)
print("R^2: ", r_2)
```



## Central Dihedral 
*In development*


## Polymer Data
*In development*