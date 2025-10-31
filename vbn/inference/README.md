| Folder             | Purpose                                                  | Typical complexity               | GPU suitability                | `do(·)` support                     |
| :----------------- | :------------------------------------------------------- | :------------------------------- | :----------------------------- | :---------------------------------- |
| **`exact/`**       | Symbolic / closed-form inference (no Monte Carlo noise)  | Low–medium (graph-wise)          | High, since all are tensor ops | ✅ direct (factor surgery)           |
| **`approximate/`** | Sampling, iterative message passing, variational methods | High (sample count / iterations) | Very high (massively parallel) | ✅ naturally (clamping/intervention) |
