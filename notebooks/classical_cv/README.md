# Classical computer vision

Practice based initially on the five reference groups under
`_references/classic-cv/`, then extended wherever curiosity leads.

The first sequence will cover:

1. thresholding, masks, morphology, connected components, and bounding boxes;
2. cameras, image acquisition, coordinates, and projection geometry;
3. image formation and geometric transformations;
4. stereo vision, disparity, depth, and 3D reconstruction;
5. patterns, handcrafted features, matching, and recognition.

Notebooks in this directory use a local two-digit sequence such as
`01_color_detection_pipeline.ipynb`.

## Possible follow-up experiments

These are candidates, not promises or homework. We can return to whichever
ones become interesting:

- filter components by area, aspect ratio, density, and shape;
- test a color-detection pipeline on real photographs under changing light;
- implement erosion and dilation with NumPy and compare them with OpenCV.
