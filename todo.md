# project embh


## to do software goals
- [] create class to implement embh using fasta file and edge list
- [] initialize parameters using F81 model
- [] compute relative contribution of each pattern
- [] determine patterns that contribute to 90%, 95% and 99% of total log-likelihood score for F81 model
- [] save on computation time by reusing 
- [] implement early stopping for embh using Aitken's method
- [] perform EM on small data set comprising heaviest patterns to determine convergence threshold
- [] test Aitken's method to determine convergence threshold
- [] split c++ code into class-wise modules
- [] serial c++ code for optimizing log likelihood using patterns
- [] convert serial c++ to serial c code
- [] c code to produce pattern file
- [] parallelize c code using cuda
- [] optimize by reusing branch-specific site patterns
- [] implement ssh model to reparameterize bh at alternate root location
- [] select optimal clique for computing log-likelihood
- [] perform memory leak test

## to do analysis goals
- [] characterize pattern weight distribution for ran alignment
- [] perform embh on test data
- [] explore impact of threshold
- [] benchmark memory consumption of optimizing embh
- [] characterize clique-tree-branch specific site patterns

## completed software goals
- [x] create test data using using 1000 columns of ran alignment

## completed analysis goals
