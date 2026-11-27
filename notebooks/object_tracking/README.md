# Object tracking

A deeper learning path for understanding how visual evidence moves through
time—and how a tracker remains confident immediately before losing the object
behind the only available tree.

This topic builds on the image processing, features, and geometry practiced in
`../classical_cv/`, but it has enough distinct machinery and failure modes to
deserve its own notebook sequence.

## Planned sequence

The order is useful, not legally binding:

1. **Motion and background subtraction** — frame differencing, background
   models, noise, shadows, and deciding which pixels actually moved.
2. **Optical flow and point tracking** — motion vectors, Lucas–Kanade
   assumptions, feature selection, lost points, and drift.
3. **Single-object tracking** — bounding boxes, appearance or correlation-based
   tracking, scale changes, occlusion, and recovery after losing the target.
4. **Prediction with a Kalman filter** — state, velocity, uncertainty, noisy
   measurements, and predicting where the object went while nobody was
   looking.
5. **Multi-object tracking** — detection-to-track association, distance costs,
   Hungarian matching, track birth and death, occlusion, and identity switches.
6. **Modern tracking, if curiosity survives** — tracking by detection,
   appearance embeddings, SORT/DeepSORT-style pipelines, and selected neural
   trackers.

Early notebooks should use controlled synthetic sequences where true motion is
known, then introduce real video as soon as the algorithms have exhausted their
easy excuses.

## Notebooks

1. [`01_motion_and_background_subtraction.ipynb`](./01_motion_and_background_subtraction.ipynb)
   — frame differencing, foreground masks, background learning, MOG2, its
   favorite failure modes, and a small real-video experiment.
2. [`02_optical_flow_and_point_tracking.ipynb`](./02_optical_flow_and_point_tracking.ipynb)
   — corner selection, Lucas–Kanade motion, pyramids, forward–backward checks,
   lost points, robot occlusion, and real-video trajectories.
3. [`03_single_object_tracking.ipynb`](./03_single_object_tracking.ipynb)
   — initialized boxes, template correlation, local versus global search,
   scale changes, occlusion, look-alikes, confidence, and real MIL tracking.
4. [`04_kalman_filter_and_motion_prediction.ipynb`](./04_kalman_filter_and_motion_prediction.ipynb)
   — noisy and missing measurements, constant-velocity prediction, correction,
   uncertainty, trust parameters, and the consequences of a robot changing its mind.

## Project organization

- Notebook-specific scene generation and plotting belong in `_helpers/`.
- Reusable tracking algorithms can move to `src/wave_part_dual/tracking/` once
  they prove useful outside one notebook.
- Tests should accompany reusable code, especially state estimation and data
  association, where a convincing animation can conceal surprisingly creative
  bookkeeping.

