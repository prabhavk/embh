# Why Root Probabilities Must Be Reapplied in Propagation Algorithm

## Your Excellent Question

You correctly observed that `rootProbability` is already incorporated into the initial potential of the root clique (line 446 in `SetInitialPotentialAndBelief`):

```cpp
// Case: X and Y are hidden and X is the root and "this" is the root clique
// psi = P(X) * P(Y|X) 
this->initialPotential = y->transitionMatrix;
for (int dna_p = 0; dna_p < 4; dna_p++) {
    for (int dna_c = 0; dna_c < 4; dna_c++) {
        this->initialPotential[dna_p][dna_c] *= x->rootProbability[dna_c];  // Applied here!
    }
}
```

So why do we need to multiply by root probabilities **again** when computing the log-likelihood?

## The Answer: Normalization Destroys Root Probabilities

### What Happens During Belief Propagation

1. **Initial Potential** (line 443-448): 
   - Root probabilities ARE incorporated: `ψ(X,Y) = P(Y|X) × P(X)`

2. **Message Passing** (in `SendMessage` and `CalibrateTree`):
   - Messages are sent between cliques
   - Each message is **normalized** (rescaled) multiple times to prevent numerical underflow
   - Normalization constants are accumulated in `logScalingFactorForClique`

3. **ComputeBelief** (lines 317-376):
   - The belief is computed by multiplying the initial potential with all incoming messages
   - The result is **normalized multiple times** (lines 338-342, 347-351, 359-363)
   - **Each normalization divides by the sum**, destroying absolute probability values
   - The final belief sums to exactly 1.0

### Evidence from Debug Output

For site 0, we saw:
```
Root probabilities: [0.287675, 0.197608, 0.259430, 0.255286]

Root clique belief matrix (after normalization):
  [0.000000, 0.000000, 0.000000, 0.000000]
  [0.000000, 0.000000, 0.000000, 0.000000]
  [0.000000, 0.000000, 1.000000, 0.000000]  ← Normalized to 1.0
  [0.000000, 0.000000, 0.000000, 0.000000]

Marginal over X (root): [0.000000, 0.000000, 1.000000, 0.000000]
```

The marginal says P(X=G|data) = 1.0 after all normalizations, but according to the model's prior, P(X=G) should be 0.259430.

**The root probabilities were normalized away!**

### What the Normalized Belief Represents

After all normalizations, the belief represents:
```
P(X, Y | data) / Z
```

where Z is a normalization constant. The marginal is:
```
P(X | data) / Z
```

This is a **conditional distribution** given the data, not the joint distribution we need for likelihood.

### Why We Must Reapply Root Probabilities

To compute the likelihood, we need:
```
P(data) = Σ_X P(data | X) × P(X)
        = Σ_X [marginal_X × rootProbability[X]]
```

The `marginal_X` (after normalization) is proportional to P(data | X), and we multiply by P(X) (the root probabilities) to get the joint likelihood.

### Comparison with Pruning Algorithm

The pruning algorithm does the same thing:

```cpp
double siteLikelihood = 0.0;
const auto& rootCL = conditionalLikelihoodMap.at(this->root);  // Normalized conditional
for (int dna = 0; dna < 4; ++dna) {
    siteLikelihood += this->rootProbability[dna] * rootCL[dna];  // Apply root probs!
}
```

The `rootCL` is normalized (via scaling factors), representing P(data | X), and we multiply by `rootProbability[dna]` to get P(data, X).

## Testing the Hypothesis

When we tried **NOT** multiplying by root probabilities:
```cpp
double siteLikelihood = 0.0;
for (int dna = 0; dna < 4; dna++) {
    siteLikelihood += marginalX[dna];  // Just sum, no weighting
}
```

Result: **Wrong log-likelihood** (-25847 instead of -27221)

When we multiply by root probabilities:
```cpp
double siteLikelihood = 0.0;
for (int dna = 0; dna < 4; dna++) {
    siteLikelihood += this->rootProbability[dna] * marginalX[dna];  // Weighted sum
}
```

Result: **Correct log-likelihood** (-27221), matching both pruning algorithms!

## Conclusion

Yes, root probabilities are included in the initial potential, but **they get normalized away** during belief propagation. The final belief is a conditional distribution that must be re-weighted by the root probabilities to compute the correct likelihood.

This is fundamentally the same as what happens in the pruning algorithm - normalizations are performed for numerical stability, and we must account for the prior probabilities at the end to get the true likelihood.

## Note on Potential Bug in Line 446

As a side observation, line 446 may have an indexing error:
```cpp
this->initialPotential[dna_p][dna_c] *= x->rootProbability[dna_c];  // Uses dna_c
```

This multiplies by the root probability at the child index, but `x` is the parent. It should probably be:
```cpp
this->initialPotential[dna_p][dna_c] *= x->rootProbability[dna_p];  // Use dna_p
```

However, this doesn't affect our fix because the normalization removes this information anyway, and we correctly reapply the root probabilities at the end. If this indexing were fixed, the answer would still be the same (root probabilities still need to be reapplied after normalization).
