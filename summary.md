THRML-TV: Solving the $10B FCC Spectrum Repacking Problem with Thermodynamic Computing
What We Built
We used extropic's THRML framework to solve the real FCC TV spectrum repacking problem - assigning 2,047 broadcast stations to 34 channels without interference. This is an NP-complete graph coloring problem that traditionally takes hours with integer programming. Our thermodynamic sampler finds approximate solutions in minutes.
The Demo

Full working implementation in python/JAX using THRML's energy-based modeling
live visualization showing constraint violations dropping to zero in real-time
actual FCC data from the 2016 incentive auction (2,047 stations, 34 channels, complex interference constraints)
runs on CPU/GPU NOW, ready for extropic hardware LATER

Why This Matters
the FCC spent $10B+ buying back spectrum and needed to repack remaining stations without creating interference. This affects every TV broadcast in america. Traditional solvers are SLOW and struggle with constraint density.
Our approach: map the problem directly to physics. Channel assignment → potts model, interference constraints → energy penalties, finding solutions → thermodynamic sampling. 
Technical Implementation
Core Innovation: Potts Model Encoding
instead of binary spins (the usual approach), we use categorical variables - one per station with 34 possible states (channels). Way cleaner:
Energy = λ_conflict × [interference violations] + λ_domain × [disallowed channels]
minimize energy → satisfy all constraints.
THRML Factor Graph

2,047 categorical nodes (one per station)
unary factors encode which channels each station can use
pairwise factors encode interference (co-channel, adjacent channel ±1,2,3...)
block gibbs sampling with local energy updates (only check ~50 neighbors vs all 2,047)

What Makes It Fast
traditional approach: check entire state space after every move
our approach: only recompute energy for neighbors bc of sparse interference patterns
this is EXACTLY the kind of local computation that thermodynamic hardware excels at - concurrent updates across thousands of nodes using analog physics.
Results
Configuration:

2,047 stations, 34 channels
~12,000 pairwise constraints
seeded from post-auction FCC assignments

Performance:

initial violations: 50-200 (depending on seed)
convergence to 0 violations: 200-400 sampling steps
typical runtime: <5 min on CPU

hyperparameters that worked:
λ_conflict = 8.0
λ_domain = 100.0  
warmup = 200 steps
samples = 400 

Why THRML Specifically
this problem is perfect for thermodynamic computing bc:

natural energy formulation - constraints = energy penalties
sparse local interactions - each station only interferes with ~50 others
massive parallelism potential - gibbs updates are embarrassingly parallel
hardware ready - same code runs on extropic TSU when available

we're not just solving the problem, we're demonstrating the killer app for thermodynamic computing: real-world combinatorial optimization with sparse constraint graphs.
Code Structure

tv_graph.py - parses FCC data, builds constraint graph
tv_thrml_potts.py - constructs factor graph, runs sampler
tv_live_viz.py - real-time violation tracking during sampling
jupyter notebooks - experiments with different problem sizes

What's Novel Here
afaict nobody's done this before:

first application of thermodynamic computing to FCC spectrum problems
categorical potts model for channel assignment (vs typical binary encodings)
validated on REAL regulatory data at full scale
complete pipeline from CSV → factor graph → solution

Future Work
same model generalizes to:

5G spectrum allocation
satellite frequency coordination
any graph coloring problem with geographic constraints

but more importantly: this is hardware-ready. the moment extropic ships TSUs, this exact code should see orders of magnitude speedup bc the physics does the computation.

tl;dr: we mapped a real $10B infrastructure problem to thermodynamic sampling and it WORKS. zero violations on full FCC dataset. hardware-ready architecture. first demo of THRML solving real regulatory optimization at scale.