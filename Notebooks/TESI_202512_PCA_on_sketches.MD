## TESI 202512 – PCA on Police Sketches: Design Notes

This document explains the main design decisions made while iterating on the notebook `TESI_202512_PCA_on_sketches.ipynb`. It is meant as instructor-facing documentation: why things look and behave the way they do, and how that supports the teaching goals.

---

## Overall pedagogical goal

- **Goal**: Provide a student-friendly, visual introduction to PCA before moving to more realistic eigenfaces (e.g. Olivetti). The core narrative is: “We secretly build faces from known parts (a sketch basis). PCA, given only the composites, can recover something very similar to those parts and reconstruct new faces from a small number of components.”
- **Constraints**:
  - The visuals must obviously look like “police sketch” faces, not abstract 8×8 blocks.
  - The relationship between *true* generating basis weights and PCA coordinates must be inspectable and explainable.
  - Interactive controls should help students explore, not confuse them (sliders, grids, labels).

---

## Basis design (Cell 2)

### From 8×8 toy blocks to 64×64 line-art faces

- **Decision**: Use 64×64 grayscale line-art images, not tiny 8×8 patterns.
- **Why**:
  - 8×8 block patterns are mathematically fine but visually too abstract; they do not read as “faces”.
  - 64×64 allows simple but expressive shapes (ellipses, arcs, lines) that strongly resemble caricature sketches.

### Specific basis elements and variants

- **Decision**: Use 12 basis images:
  - `Head outline` (1 variant)
  - `Hair` (2 variants: curly, wavy)
  - `Eyes` (3 variants)
  - `Nose` (3 variants)
  - `Mouth` (3 variants)
- **Why**:
  - Having only one variant per feature made the space too limited and visually boring.
  - Having 2–3 variants per key feature demonstrates that PCA can discover *alternative modes* within a semantic group (different hairstyles, eye shapes, etc.).

### Geometric tuning for recognisable and non-overlapping features

Several corrections were made to the shapes:

- **Head ellipse**
  - Enlarged slightly in both axes so that all features fit comfortably inside.
  - Reason: the original head outline sometimes clipped the mouth or hair; it looked wrong and distracted from PCA.

- **Hair variants**
  - Drawn along the *curve of the head ellipse*, not as horizontal “hats”.
  - Variant 1: small circles along the top for “curly” hair.
  - Variant 2: short diagonal strokes for “wavy” hair.
  - Reason: better cartoon realism and clear visual distinction between hair styles.

- **Nose**
  - Repositioned and reshaped so the “T” nose no longer collides with the eyes, and a triangular nose variant was added.
  - Reason: earlier versions overlapped eyes, making it hard to interpret which part corresponded to which basis.

- **Mouth**
  - Width reduced to fit inside the head; defined as a line or partial ellipse (smile/frown) around a mouth center.
  - Reason: previous wide mouth extended outside the head ellipse, again breaking the facial illusion.

### Displaying bases with names

- **Decision**: Show a 2×6 grid of the 12 basis images with `base_names` (“Hair 1”, “Eyes 2”, …).
- **Why**:
  - Students need to see and name the “true” basis elements before they see what PCA recovers.
  - Names are reused in later plots (weight strips, grids) to keep the mapping consistent.

---

## Two datasets: `X_vis` vs `X_pca` (Cell 3)

### Visual dataset `X_vis`

- **Decision**: Create `X_vis` as a set of discrete composite sketches:
  - Always include the head.
  - For each group (hair, eyes, nose, mouth), randomly choose exactly *one* variant and weight it as 1.0.
- **Why**:
  - This gives clean, human-readable faces where each feature is “on” or “off” in an obvious way.
  - It matches the intuitive story “this sketch is head + hair variant 2 + eyes variant 3 + …”.
  - This dataset is *only* used for visualization and interactive reconstruction, not for fitting PCA.

### PCA training dataset `X_pca`

- **Decision**: Train PCA on a *different* dataset, `X_pca`:
  - Use continuous weights in `[0, 1]` for all 12 bases.
  - Independently drop groups with specified probabilities (e.g. head keep prob ~0.9, other groups ~0.7).
  - Add small Gaussian noise and clip to `[0, 1]`.
- **Why**:
  - If we always used exactly one variant of each feature in every face, features would be highly correlated; PCA tends to mix them into global modes instead of clean part-like components.
  - Continuous weights plus occasional missing features break those correlations and make it easier for PCA to recover something close to the original part basis.
  - Separating `X_vis` (didactic, clean faces) and `X_pca` (statistically richer) lets us keep good teaching visuals without compromising PCA quality.

---

## PCA fitting and alignment (Cell 4)

### Centering with a reusable mean

- **Decision**:
  - Compute and store `X_pca_mean_val = X_pca.mean(axis=0, keepdims=True)`.
  - Fit PCA on the centered data `X_centered = X_pca - X_pca_mean_val`.
- **Why**:
  - Correct PCA requires centering; explicitly storing the mean ensures later transforms use the *same* mean, avoiding subtle reconstruction errors.
  - The mean is also reused in the reconstruction cell.

### Number of PCA components

- **Decision**: Use `n_pca_components = 2 * n_bases` (24 PCs for 12 bases).
- **Why**:
  - Using at least as many PCs as bases is necessary if we want PCA to be able to span the original basis space.
  - Extra components capture noise/minor variations and let students see how “long tails” of components add small refinements to the reconstruction.

### Aligning PCA components to basis order

- **Decision**:
  - Compute the correlation matrix `corr = C @ B.T` between PCA components `C` and flattened bases `B`.
  - For each original basis index `j`, find the best-matching component `k = argmax |corr[k, j]|`.
  - Fix the sign so that the component matches the basis orientation: multiply by `sign(corr[k, j])`.
  - Store the matched, sign-corrected images in `aligned_components[j]`.
- **Why**:
  - Raw PCA components are (1) arbitrarily ordered and (2) sign-ambiguous. Without alignment, “PC 0” might line up with “Mouth 3” with a negative sign.
  - Pedagogically, it is clearer to speak about “the PC that corresponds to Hair 2” rather than “PC 7”.
  - The alignment permutation `perm` is also reused later to map PCA coordinates to the basis order when we compare weights.

### Per-component normalization for PCA images

- **Decision**:
  - For each aligned component image used for display, rescale it to `[0, 1]` using its own min and max before plotting.
- **Why**:
  - PCA components can have small dynamic range; unnormalized, they looked almost black even though they were meaningful vectors.
  - Per-image normalization is purely for visualization; it does **not** affect the math or reconstruction, but makes the “shape” of each component visible.

---

## Interactive reconstruction design (Cell 5)

### Correct dataset and centering

- **Decisions**:
  - Use `X_vis[example_idx]` as the “original sketch” to reconstruct.
  - Use the *PCA training mean* `X_pca_mean_val` when projecting:
    - `x_centered_single = x.reshape(1, -1) - X_pca_mean_val`
    - `z_full = pca.transform(x_centered_single)`
  - When reconstructing, first obtain the centered reconstruction and then add the mean back:
    - `x_recon_centered = pca.inverse_transform(z)`
    - `x_recon = x_recon_centered + X_pca_mean_val.ravel()`
- **Why**:
  - Earlier versions accidentally used a different dataset or forgot to add the mean back, which produced strange reconstructions (e.g. eyes drifting).
  - Using the same centering for training and inference is required for PCA to behave correctly.

### Sliders: sketch index and number of PCs

- **Decision**:
  - `Sketch index` slider: ranges over all examples in `X_vis`.
  - `# of PCs` slider: ranges from `1` up to `2 * n_bases` (the number used in PCA fitting).
- **Why**:
  - The sketch index lets students browse different combinations of basis features.
  - The PC count slider shows how reconstruction quality improves as more components are included, up to the full subspace used in training.

### Truncating PCA coordinates cleanly

- **Decision**:
  - Compute the full PCA coordinates `z_full` for the centered sketch.
  - Construct a truncated version `z` where only the first `n_components` entries are kept and the rest explicitely set to zero.
  - Use `z` both for reconstruction *and* for matched basis weights.
- **Why**:
  - Students expect that “using k PCs” means “all other components are truly zero”.
  - If we used `z_full` for the weight computation, components nominally “turned off” by the slider would still contribute, causing confusing non-zero weights.

---

## Weight visualization: strips with y-ticks (Cell 5)

### Why show weight strips at all?

- **Decision**:
  - For each reconstruction, plot two vertical strips:
    - **Original basis weights**: the true coefficients used to create the sketch in the known basis (`true_weights`).
    - **Matched PC weights**: PCA coordinates, mapped back into basis order via `perm` after truncation (`matched_pc_weights`).
  - Normalize each vector by its own maximum absolute value before display so the brightest stripe corresponds to the largest magnitude in that vector.
- **Why**:
  - The main teaching point is not only that PCA recovers *images* close to the bases, but also that coordinates in PCA space relate in a structured way to the original basis coefficients.
  - Seeing these as side-by-side strips helps students compare “what truly generated the image” vs. “how PCA represents it”.

### Why add y-ticks and basis names?

- **Decision**:
  - Use `base_names` as y-tick labels on both strips.
- **Why**:
  - Earlier, the strips were unlabeled; it was unclear which row corresponded to which feature.
  - With labels, students can read specific statements such as “for this sketch, Hair 1 has high true weight and high matched PC weight; Nose 3 is zero in both”, which grounds the abstract vectors in concrete features.

### Normalization of weight vectors

- **Decision**:
  - Normalize each weight vector by its own maximum absolute value before image display (`norm_vec`), but **do not** change the actual numerical values shown in text (e.g. in the grid).
- **Why**:
  - Without normalization, a single very large weight would make all others appear nearly black, even though they are important.
  - Normalization is a purely visual scaling to make relative patterns easy to spot; numeric values are preserved for textual display where needed.

---

## Grid of significant PCA basis images (Cell 5)

### Show *PCA* basis images, not original basis images

- **Decision**:
  - The grid at the bottom shows `aligned_components[idx]` (PCA components matched, and sign-corrected, to each original basis), *not* the original basis images themselves.
- **Why**:
  - The point of the grid is to visually answer: “Which PCA basis vectors are currently most important to this reconstruction?”
  - Using aligned PCA components makes the grid directly reflect the learned PCA directions; matching to bases simply provides an intuitive name.

### Per-image normalization for visibility

- **Decision**:
  - Each component image in the grid is normalized independently to `[0, 1]` before being plotted.
- **Why**:
  - Some components have very small amplitude; without normalization they look uniformly dark.
  - Per-image normalization is the same strategy as in the main PCA-vs-basis comparison grid and is again only for visibility.

### Filtering which bases appear in the grid

The logic for deciding which tiles to show evolved through several iterations to match the intended semantics.

#### Initial ideas and their issues

1. **Filter only by PC weight magnitude**  
   - `significant = { j : |matched_pc_weights[j]| > threshold }`  
   - **Problem**: could show bases with `w = 0` (not present in the original sketch) but non-zero PC weight due to truncation or noise. This was confusing: tiles appeared for features that “should not be there”.

2. **Always show all active bases**  
   - `active = { j : true_weights[j] != 0 }` and show every active basis, sorted by |PC weight|.  
   - **Problem**: this ignored `# of PCs`—the bottom grid always had the same number of tiles. The user wanted the grid to also reflect how many PCs are currently “in play”.

#### Final behaviour

- **Decision**:
  - Define **active bases** as those actually used in the original sketch:
    - `active = { j : |true_weights[j]| > 1e-6 }`
  - For the current `n_components`:
    - Score the active bases by `|matched_pc_weights[j]|`.
    - Sort them descending by score.
    - Show **up to** `n_components` of these, i.e.  
      `k = min(n_components, len(active))` and `significant = active[order[:k]]`.
- **Why**:
  - Ensures that:
    - Only bases that truly participate in the sketch (non-zero `true_weights`) ever appear in the grid.
    - As `# of PCs` increases, more active bases can enter the grid, up to all of them.
    - When `# of PCs` is small, the grid highlights the subset of active bases that currently contribute the most according to PCA.

### Text under each grid image

- **Decision**:
  - For each tile, show the original and matched PC weights numerically, e.g. `w=1.00`, `pc=5.36`.
- **Why**:
  - The grayscale strip gives relative magnitudes, but the numbers are needed when discussing absolute differences (“PCA uses a larger/smaller coefficient than the original basis weight”).
  - This helps explain where PCA is faithful and where it “mixes” components.

---

## Flattened vector exploration (Cell 6)

- **Decision**:
  - Add an optional widget that, for a chosen basis index, displays:
    - The original basis image.
    - The matched PCA component image.
    - Their flattened 1D vectors (each normalized to [0, 1]) as thin vertical images.
- **Why**:
  - Some students (or instructors) may want to go beyond images and see that PCA is literally working on flattened vectors.
  - This tool provides a bridge between the geometric picture (“images/features”) and the algebraic one (“vectors in ℝⁿ”).

---

## Summary for instructors

- **Two datasets**: `X_vis` (clean, discrete faces for visualization) and `X_pca` (richer, continuous, partially missing features for training PCA).
- **Explicit centering**: `X_pca_mean_val` is computed once and reused for all PCA transforms and reconstructions.
- **PCA alignment**: Components are permuted and sign-corrected to match named basis elements; this mapping powers both image grids and weight comparisons.
- **Visual aids**:
  - Per-image normalization for PCA and grid images makes patterns visible despite scale differences.
  - Labeled weight strips with y-ticks connect abstract coefficients to concrete features.
  - The bottom grid only shows active bases and its size responds to `# of PCs`, so students see which learned directions matter for the current reconstruction.

These choices are all aimed at making the PCA behaviour *visually obvious* and *narratable* in terms of familiar “face parts”, while staying faithful to the underlying linear algebra.

