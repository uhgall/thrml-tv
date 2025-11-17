
THRML-TV: Solving the $20B FCC Spectrum Repacking Problem with Thermodynamic Computing

Using Extropic's THRML framework, we developed a solver for the type of graph coloring problem that was required to reassign TV channels broadcast stations after the FCC's $20 billion Incentive auction where some of the TV spectrum was reallocated to mobile operators.

Features:
- Full working implementation in python/JAX using THRML's energy-based modeling
- Live visualization showing constraint violations dropping to zero in real-time

While the problem was too large to find good solutions using the THRML simulator, our solver quickly found good solutions for scaled-down versions of the problem. 
A hardware-based solution might well find an even better solution than the current assignment, where 35 channels were required to ensure that 2047 TV stations could broadcast without producing interference.

The problem mapped nicely to the Potts Model supported by THRML; we used Categorical Nodes for the stations, where each possible state corresponded to a color. Our code compiled the interference constraints from the FCC data into edges in the graph such that an energy penalty was imposed when two connected stations were on the same channel. We also implemented adjacency constraints and enforced the permitted domain of channels for each station.
