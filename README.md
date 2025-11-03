# embh — EM for the Barry–Hartigan Model

**embh** implements the *expectation–maximization algorithm* (EM) for the **Barry–Hartigan** model (BH) .

## Features
- Optimizes BH using belief propagation on clique-trees
- Verifies non-identifiability of root
- Reuses messages of repeating branch-specific site patterns
- Suggests optimal location of root clique to minimize messages
- Modular C core (GPLv3) with CUDA acceleration 


