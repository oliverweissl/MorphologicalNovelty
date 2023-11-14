# Thesis
## - THIS IS DEPRECATED -> If you want to use Morphological Novelty yourself look at revolve2/ci-group https://github.com/ci-group/revolve2

 
## Prepare for Simulations:
- Install Python 3.10
- Install Revolve2 -> https://ci-group.github.io/revolve2/installation/index.html
- Install Julia 1.8
- in _python_ terminal:
  - pip install julia
- in _julia_ terminal: 
  - import Pkg
  - Pkg.add("PyCall")
  -  Pkg.add("Shuffle")

## Run Simulations:
For one specific simulation:
- python3 -m run_batch --novelty \<novelty weigth or None\> --amount \<amount of simulations\> --seed \<seed\><br/>
 
For running multiple simulations in parrallel:
- sh start_batches.sh 

if novelty\_weight -> parent selection using weighted average (fitness*(1-novelty\_weight) + novelty*novelty\_weight) </br>
if None -> parent selection using penalized fitness (fitness*novelty)
