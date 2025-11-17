# Project Summary: THRML-TV - Thermodynamic Computing for FCC Spectrum Repacking

## Overview

This project demonstrates a novel approach to solving the FCC TV spectrum repacking problem using **thermodynamic computing** and **energy-based models** via Extropic's THRML framework. The solution maps the NP-complete graph coloring problem of channel assignment to a Potts model that can be efficiently sampled using stochastic thermodynamic processes.

## Problem Domain

The FCC TV spectrum repacking involves assigning broadcast channels to television stations while satisfying complex interference constraints:
- **2,047 TV stations** (in full dataset)
- **34 available channels** (post-auction)
- **Multiple constraint types**: co-channel (CO), adjacent channel (ADJ±n)
- **Geographic interference patterns** based on signal propagation

This is fundamentally a **graph coloring problem** where:
- Vertices = TV stations
- Edges = interference relationships
- Colors = available broadcast channels
- Goal = Find valid assignment with zero constraint violations

## Technical Implementation

### Core Architecture

The implementation leverages three key technical components:

#### 1. **Potts Model Energy Formulation**

We encode the constraint satisfaction problem as an energy minimization task using a K-state Potts model:

```
E(c) = λ_conflict × Σ[constraint violations] + λ_domain × Σ[domain violations]
```

Where:
- `c = (c₀, c₁, ..., c_N)` represents channel assignments
- `λ_conflict` penalizes forbidden channel pairings (interference)
- `λ_domain` penalizes assignments outside station-specific allowed channels
- Energy minimization ≡ constraint satisfaction

#### 2. **THRML Factor Graph Construction**

The problem is represented as a **discrete energy-based model** with:

- **Categorical Nodes**: One per station (N=2047), each with K states (K=34 channels)
- **Unary Factors**: Encode per-station channel domain constraints
- **Pairwise Factors**: Encode interference constraints between station pairs
- **Boltzmann Distribution**: `P(c) ∝ exp(-β·E(c))`

Key advantages over binary encoding:
- Cleaner representation (N categorical variables vs N×K binary spins)
- Natural constraint encoding (no one-hot enforcement needed)
- Sparse local interactions (only neighbors matter for energy updates)

#### 3. **Block Gibbs Sampling**

Sampling is performed using categorical Gibbs conditionals:

```python
# For each station i, sample new channel from:
P(c_i | c_neighbors) ∝ exp(-β · ΔE_i)
```

- **Local energy updates**: Only recompute energy for neighboring stations (~50 vs 2047)
- **Stochastic optimization**: Balance exploration (high temp) and exploitation (low temp)
- **Hardware-ready**: Same model runs on both JAX/CPU and future thermodynamic hardware

### Software Components

#### `tv_graph.py` - Graph Data Structures
- Parses FCC CSV inputs (Domain, Interference_Paired, parameters)
- Builds station objects with geographic coordinates and constraints
- Maintains channel mappings and interference relationships
- Validates constraint coverage and detects violations

#### `tv_thrml_potts.py` - THRML Sampler
- Constructs categorical nodes for each station
- Builds domain factors (unary energy terms)
- Builds interference factors (pairwise Potts energy)
- Compiles `FactorSamplingProgram` with Gibbs conditionals
- Runs sampling with configurable warmup and collection phases
- Tracks best (lowest energy) solutions

#### Interactive Notebooks
- `tv_potts_sampling.ipynb`: Single-node experimentation
- `tv_potts_sampling_multinode_blocks.ipynb`: Distributed sampling experiments

#### Visualization Tools
- `tv_live_viz.py`: Matplotlib-based live sampling visualization
- `tv_live_viz_pyqtgraph.py`: PyQtGraph high-performance visualization
- Real-time display of constraint violations during sampling

### Key Features

1. **Flexible Dataset Support**: Works with various problem sizes
   - Small test cases: 50 stations, 30 channels
   - Full FCC dataset: 2047 stations, 34 channels
   - Post-auction variations with different constraint densities

2. **Configurable Penalties**: Tunable `λ_conflict` and `λ_domain` parameters allow balancing hard vs soft constraints

3. **Smart Initialization**: Seeds from post-auction assignments when available, random otherwise

4. **Real-time Monitoring**:
   - Live energy tracking
   - Violation counts by constraint type (CO, ADJ-1, ADJ+1, etc.)
   - Geographic visualization of violations

5. **Solution Quality Metrics**:
   - Domain violations (stations assigned disallowed channels)
   - Interference violations by type
   - Total energy score
   - Best sample tracking across sampling trajectory

## Results & Performance

The sampler successfully finds **valid channel assignments** (zero violations) for the FCC post-auction repacking problem:

- **Initial state**: Typically 50-200 violations (depending on initialization)
- **After sampling**: 0 violations achievable with proper hyperparameter tuning
- **Convergence**: Typically within 200-400 Gibbs sweeps for medium-sized instances

Example configuration:
```bash
--lambda-conflict 8.0
--lambda-domain 100.0
--warmup 200
--samples 400
```

## Innovation & Future Potential

### Why Thermodynamic Computing?

Traditional approaches to spectrum repacking use:
- Integer linear programming (slow for large instances)
- Greedy heuristics (suboptimal solutions)
- Simulated annealing (sequential, CPU-bound)

This project demonstrates that thermodynamic computing offers:

1. **Natural Problem Fit**: Constraint satisfaction maps directly to energy minimization
2. **Massive Parallelism**: Local energy updates enable concurrent node updates
3. **Physical Speedup**: Future hardware (Extropic TSU) implements Gibbs sampling in analog circuitry
4. **Scalability**: Factor graph locality means computation scales with edge density, not state space size

### Hardware Acceleration Path

The same THRML model runs on:
- **Software**: JAX implementation (current - CPU/GPU)
- **Hardware**: Extropic thermodynamic sampling unit (future)

Hardware advantages:
- Orders of magnitude faster sampling rates
- True physical noise for randomness
- Analog computation for energy evaluation
- Concurrent updates across thousands of nodes

## Broader Applications

The techniques demonstrated here generalize to:
- **Wireless spectrum allocation** (5G, satellite)
- **Resource scheduling** (cloud computing, manufacturing)
- **Map coloring** (register allocation, task assignment)
- **Constraint satisfaction** (logistics, planning)

Any problem expressible as energy minimization over discrete variables with local interactions is a candidate for this approach.

## Technical Requirements

- Python 3.11+
- JAX (CPU/GPU)
- THRML (Extropic's energy-based modeling library)
- NumPy, Matplotlib (visualization)
- FCC broadcast data (included in `input/` directories)

## Conclusion

This project showcases a **first-principles approach** to combinatorial optimization using thermodynamic computing. By formulating the FCC spectrum repacking problem as a Potts model and leveraging THRML's factor graph abstraction, we achieve:

✓ **Clean mathematical formulation** (energy-based modeling)  
✓ **Efficient software implementation** (JAX-based Gibbs sampling)  
✓ **Hardware-ready design** (factor graphs with local interactions)  
✓ **Proven results** (zero-violation solutions on real FCC data)  
✓ **Scalable architecture** (tested up to 2,047 stations)

The combination of domain expertise (broadcast engineering), mathematical rigor (statistical physics), and modern tooling (THRML/JAX) demonstrates the practical viability of thermodynamic computing for real-world optimization problems.
