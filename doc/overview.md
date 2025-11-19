# Graph Coloring with Extropic / THRML 
# The FCC's broadcast TV channel packing problem

## 1. What problem are we solving?

Standard graph coloring problem:
- A **graph** with
  - `N` vertices (nodes), e.g. TV stations or generic items
  - `M` edges, each edge `(u, v)` means “u and v conflict if they share the same resource”
- A set of **K colors** 1..K.

We want a **graph coloring**:
- Assign each vertex `i` a color `c_i ∈ {0, 1, …, K-1}`
- Such that **no edge is monochromatic**:
  - For every edge `(u, v)`, we want `c_u ≠ c_v`.

The particular problem implemented here is a variation of this problem: 
- Nodes are TV broadcast stations
-- Each station broadcasts on a particular channel;  this corresponds exactly to a color
- Edges indicate different kinds of interference constraints. There can be more than 2 edges between a pair stations. Each edge indicates: If station A is on channel c_A, then station B cannot be on channel c_B because that would cause an unacceptable amount of interference.
-- "CO"-channel interference: c_A = c_B
-- "ADJ+N constraints: c_A = c_B + N
- All constraints are symmetric, so for example for each ADJ+1 constraint, there is a corresponding ADJ-1 constraint with the station ordering reversed. 

---

## 2. Turning it into an energy minimization problem

The classic trick: turn a constraint problem into **energy minimization**.

We define an **energy function** `E(c)` over all color assignments `c = (c_0, …, c_{N-1})`.

For graph coloring, a natural choice is the **Potts energy**:

\[
E(c) = λ \sum_{(u,v) ∈ E} 1[c_u = c_v]
\]

- `1[·]` is an indicator: 1 if the condition is true, 0 otherwise.
- `λ > 0` is a penalty weight.
- For each edge `(u, v)` we add `λ` when the endpoints have the **same** color, otherwise 0.

Properties:
- Any **proper coloring** (no conflicts) has `E(c) = 0`.
- Any **improper coloring** has `E(c) ≥ λ` times the number of bad edges.

So:
- **Minimizing energy** ≡ **minimizing conflicts**.
- At temperature ~0 with large `λ`, the system strongly prefers proper colorings.

This is exactly the **K-state Potts model** (generalization of the Ising model from 2 to K states) with **antiferromagnetic** coupling (it dislikes equal states on an edge).

---

## 3. From energy to probability: Boltzmann distribution

Instead of just saying “minimize E”, we embed the problem into a **probabilistic** framework.

Define the Boltzmann distribution at inverse temperature β:

\[
P(c) = \frac{1}{Z} \exp(-β E(c))
\]

- `Z` is the normalization constant (partition function).
- Low-energy configurations get **exponentially higher** probability than high-energy ones.

Why this is useful:
- Sampling from `P(c)` (or running a Markov chain that converges to it) is a way to perform **stochastic optimization**:
  - At high temperature (small β), the chain explores widely (lots of random moves).
  - At low temperature (large β), the chain concentrates near low-energy configurations.

So the plan is:
1. Encode the graph coloring problem as `E(c)`.
2. Use a **sampler** (Gibbs or similar) to explore `P(c)`.
3. Track low-energy states as candidate solutions.

---

## 4. Discrete variables: categorical nodes instead of spins

A lot of literature uses **binary spins**: `s_i ∈ {−1, +1}` and then encodes colors with blocks of spins.

Here we take a much cleaner approach:

- One **categorical variable** per vertex:
  - `c_i ∈ {0, 1, …, K-1}`
  - K is the number of colors.

This has big advantages:
- No need to encode one-hot constraints ("exactly one color" is built in).
- The graph is much smaller in dimensionality: `N` variables, not `N × K` spins.
- The interaction structure is naturally **pairwise categorical**: each edge is a function of `(c_u, c_v)`.

Mathematically, each variable is a discrete random variable with K possible states; THRML calls these **CategoricalNodes**.

---

## 5. Local interactions: why factor graphs matter

To run this efficiently, we use a **factor graph** representation.

- Variables: `{c_0, …, c_{N-1}}`
- Factors: one factor per edge `(u, v)` encoding the Potts penalty.

Each edge factor contributes:

\[
\phi_{uv}(c_u, c_v) = \exp(-β λ 1[c_u = c_v])
\]

or in energy form:

\[
E_{uv}(c_u, c_v) = λ 1[c_u = c_v]
\]

Key ideas:
- The **global** energy `E(c)` is the sum over **local** edge energies:
  - `E(c) = Σ_{(u,v)∈E} E_{uv}(c_u, c_v)`
- This locality is crucial: we never need to recompute E from scratch if only one node changes.

If node `i` proposes a change from color `a` to `b`, the energy difference is:

\[
ΔE = \sum_{v ∈ N(i)} [E_{iv}(b, c_v) - E_{iv}(a, c_v)]
\]

- Only neighbors `N(i)` matter (typ. ~50, not 2000).
- This is what makes Gibbs sampling efficient and what Extropic’s hardware exploits.

---

## 6. What THRML is and how we use it

**THRML** is Extropic’s software library for specifying and working with energy-based models, including:

- Variables: spins, categorical nodes, etc.
- Factors: functions of subsets of variables that add to the energy.
- Sampling programs: Gibbs, block-Gibbs, etc., implemented in JAX.

### 6.1. Our modeling steps in THRML

1. **Define nodes**
   - For each vertex `i`, create a `CategoricalNode` with `K` states.

2. **Define a custom factor** for the Potts interactions
   - We implement a `PottsGraphFactor` that knows:
     - the graph’s edge list `(u, v)`
     - the penalty `λ`
     - how many colors `K` we have
   - Internally, this factor produces the correct interaction structure so that for any configuration `c`, the energy `E(c)` is exactly the Potts energy above.

3. **Combine nodes + factor into an EBM**
   - A “Discrete Energy-Based Model” object that THRML can sample from.

4. **Set up a sampling program**
   - Use THRML’s block-Gibbs machinery:
     - We define one **block** containing all categorical nodes to start.
     - We associate a **Categorical Gibbs conditional** with that block, meaning THRML knows how to resample each node from its conditional distribution given its neighbors.
   - THRML then gives us a `BlockSamplingProgram` or `FactorSamplingProgram` that we can call as a stepper: `state_{t+1} = step(state_t, rng_key)`.

At this point we have a **complete, software-level Gibbs sampler** for graph coloring.

---

## 7. Why we still need hardware if software works

From a pure software point of view:
- We know how to evaluate energy.
- We know how to do Gibbs sampling.
- So we *can* run this in JAX on CPUs/GPUs.

The issue is **scaling** and **mixing speed**:
- The state space is gigantic: `K^N` (e.g. `60^2000` ≈ `10^3500`).
- Standard MCMC steps are **sequential**:
  - pick one node
  - compute ΔE from its neighbors
  - accept/reject
  - repeat
- Even with vectorization and GPUs, at some point this is slow to explore deeply.

**Extropic’s hardware** (a thermodynamic sampling unit) aims to:
- Implement these local ΔE computations and stochastic updates in **physical hardware**, with:
  - massive parallelism (many nodes updating concurrently)
  - very fast analog / mixed-signal arithmetic
  - true physical noise as the randomness source
- That means the effective sampling rate (transitions per second) could be orders of magnitude higher than in software.

Conceptually:
- THRML is the **modeling language and simulator**.
- Extropic hardware is the **physical runtime** that runs the same model much faster and more efficiently.

---

## 8. Where our custom Potts factor fits in

Our `PottsGraphFactor` is the glue between the mathematical model and the THRML/Extropic world. It is responsible for:

1. **Encoding the graph structure**:
   - It stores the edge list `(u, v)`.
   - It tells THRML which categorical nodes interact.

2. **Encoding the interaction energy**:
   - For each edge `(u, v)`, and each pair of colors `(a, b)`, the energy contribution is:
     - `λ` if `a == b`
     - `0` otherwise
   - This can be represented as a `K × K` matrix per edge (often shared across all edges).

3. **Providing data for local conditionals**:
   - The sampler needs, for a given node `i`, the energy of each possible color `a` conditional on neighbors.
   - The factor gives exactly the pairwise tables needed to compute that.

THRML then uses this factor to:
- build the global energy function
- compute local conditionals for Gibbs updates
- run the Markov chain either in software (JAX) or, conceptually, in hardware.

---

## 9. The end-to-end workflow

### 9.1. Offline (software) workflow

1. **Prepare data**
   - Have an edge list file with `M` lines of `u v` integers.
   - Choose number of colors `K` and penalty `λ`.

2. **Load and build model**
   - Load `(N, edges)` into JAX.
   - Create `N` `CategoricalNode`s.
   - Create a `PottsGraphFactor` from `(edges, λ, nodes)`.
   - Build a `DiscreteEBM` / `FactorSamplingProgram` in THRML.

3. **Initialize state**
   - Draw random initial `colors[i] ∈ {0..K-1}`.

4. **Run sampling**
   - For `t = 1..T`:
     - Call the THRML sampling program step function.
     - Occasionally compute energy and conflict count with pure-JAX utilities.
     - Track the best (lowest energy) configuration seen.

5. **Extract solution**
   - The best configuration is a near-optimal or (if you’re lucky) perfect coloring.

### 9.2. Later: hardware-accelerated workflow

Once Extropic’s TSU hardware is in play, the idea is:

1. Use the **same THRML model** (nodes, factors, PottsGraphFactor).
2. Instead of running the sampler in JAX, point THRML’s backend to the hardware sampler.
3. The hardware then performs the same Gibbs-type transitions:
   - uses the same factor graph
   - only now, state updates occur massively in parallel and at device speeds.
4. Periodically, read back samples to host, evaluate energy / conflicts, and collect solutions.

The math does not change; only the implementation of the stochastic transitions does.

---

## 10. Why this is a good testbed

This setup is a great example problem for Extropic’s stack because:

- It’s **combinatorial** and **hard** in general (NP-complete in K, N).
- The energy is:
  - pairwise
  - sparse (only edges matter)
  - local (each node talks to its neighbors)
- It’s very naturally expressed as an EBM on discrete variables.
- We can easily vary:
  - Graph size (N)
  - Edge density (M)
  - Number of colors (K)
  - Constraint hardness (λ, temperature schedule)

So it’s almost the canonical “hello world but actually non-trivial” for a thermodynamic computing system:
- Clean math
- Clear energy landscape
- Easy correctness checks (zero conflicts)
- Clear performance metrics (time to reach a proper coloring, quality of solutions vs. time).

---

## 11. Summary

1. We encode graph coloring as a **K-state Potts model**:
   - one categorical variable per vertex
   - pairwise penalties on edges
2. We define an **energy function** whose minima correspond to proper colorings.
3. We use **THRML** to:
   - model variables and factors
   - compile them into a factor graph
   - run discrete Gibbs sampling over the configuration space.
4. Our custom **PottsGraphFactor** bridges the graph structure and the THRML EBM API.
5. In software, we sample via JAX; in the future, the **same model** can run on Extropic’s hardware for massive acceleration.

The whole point is to separate:
- **Modeling** (what is the energy?)
- **Inference / sampling** (how do we explore the energy landscape?)

THRML gives us clean modeling and sampling in software, and Extropic’s TSU aims to radically speed up the sampling part in hardware, without changing the underlying math.

