# project embh



## to do software goals
- [] replace gap encoding of -1 with 4
- [] compute log-likelihood of F81 model using pruning algorithm with fasta input
- [] compute log-likelihood of F81 model using pruning algorithm with patterns input
- [] compute log-likelihood of F81 model using propagation algorithm with patterns input
- [] reuse message of branch pattern for propagation algorithm with patterns input and compare log-likelihood score
- [] optimize BH by reusing messages
- [] parallelize embh using cuda
- [] initialize parameters using F81 model
- [] determine patterns that contribute to 80%, 85%, 90%, 95% and 99% of total log-likelihood score of F81 model
- [] create datasets P80, P85, P90, P95 and P99 containing 80% to 95% of patterns that contribute to log-likelihood score of F81 model
- [] Determine convergence threshold using Aitken's method for P90, P95 and P99, and compare thresholds
- [] perform EM on small data set comprising heaviest patterns to determine convergence threshold
- [] test Aitken's method to determine convergence threshold
- [] implement ssh model to reparameterize bh at alternate root location
- [] select optimal clique for computing log-likelihood
- [] create memory leak test

## to do analysis goals
- [] characterize pattern weight distribution for ran alignment
- [] perform embh on test data
- [] explore impact of threshold
- [] benchmark memory consumption of optimizing embh
- [] characterize clique-tree-branch specific site patterns

## completed software goals
- [x] create class to implement embh using fasta file and edge list
- [x] create test data using using 1000 columns of ran alignment

## completed analysis goals
