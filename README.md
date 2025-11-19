# THRML Example: Graph Coloring

This repository demonstrates how [Extropic](https://extropic.com)'s thermodynamic computing technology can tackle challenging graph coloring problems, focusing on a real-world example: a TV station channel packing problem.

This project was one of the winners at the THRML-HACK hackathon on Nov 16, 2025. Feel free to do whatever you want with this code - MIT License.

## Example problem: FCC Incentive Auction

The FCC Incentive Auction (2016–2017) was a government-run two-sided auction, in which:

- **TV stations** sold their broadcast rights.
- **Mobile carriers** purchased the newly released spectrum.

The primary engineering challenge was **channel repacking**: The auction algorithm required checking, during the bidding phase, whether subsets of TV stations could be packed into a given number of channels without violating interference constraints. This step is essentially an **enormous graph coloring problem**:

- **Stations** = nodes
- **Channels** = colors
- **Interference** = edges
- **Allowed channels** = per-node domain

The original FCC auction system solved the problem by reducing it to SAT and running it on standard SAT solvers. Even on the significant compute cluster allocated for the auction, due to the problem size (2990 nodes × 49 colors × 2.7 million edge constraints), optimal solutions could not be guaranteed. Even small improvements in coloring directly corresponded to significant public savings.

## Solution using THRML

We formulate repacking as an energy minimization problem:

- **Unary factors**: penalize assignments outside each station’s allowed channel list.
- **Pairwise factors**: penalize interfering stations assigned to the same channel.

The software implementation supports CPU and GPU execution via [Extropic's THRML SDK](https://extropic.com/). Once Extropic's hardware becomes available, it should be possible to run it directly on the hardware.

For details, see [doc/overview.md](doc/overview.md).

## Source Code 

tv_graph.py parses the input files and produces an in-memory representation of the graph for convenient access to all the data

tv_web_viz contains code to visualize the solver process while it is running. 




