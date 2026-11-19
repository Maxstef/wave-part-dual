# Classical computer vision

Practice based initially on the five reference groups under
`_references/classic-cv/`, then extended wherever curiosity leads.

The first sequence will cover:

1. thresholding, masks, morphology, connected components, and bounding boxes;
2. cameras, image acquisition, coordinates, and projection geometry;
3. image formation and geometric transformations;
4. stereo vision, disparity, depth, and 3D reconstruction;
5. patterns, handcrafted features, matching, and recognition.

Current notebooks:

- `01_color_detection_pipeline.ipynb` — color masks, morphology, and connected components;
- `02_camera_laboratory.ipynb` — sensors, camera settings, shutter timing, intrinsics, distortion, and distance.
- `03_coordinates_and_calibration.ipynb` — homogeneous coordinates, world-to-camera projection, pixel rays, tangential distortion, calibration, and pixel-to-ground geometry.
- `04_stereo_depth_and_point_clouds.ipynb` — stereo correspondence, block matching, disparity, metric depth, point clouds, active infrared, and sparse feature matching.
- `05_edges_hough_and_ransac.ipynb` — smoothing, edge filters, least-squares fitting, Hough voting, and robust line fitting with RANSAC.

Notebooks in this directory use a local two-digit sequence such as
`01_color_detection_pipeline.ipynb`.

## Possible follow-up experiments

These are candidates, not promises or homework. We can return to whichever
ones become interesting:

- filter components by area, aspect ratio, density, and shape;
- test a color-detection pipeline on real photographs under changing light;
- implement erosion and dilation with NumPy and compare them with OpenCV;
- stress-test stereo matching with noise, brightness changes, vertical
  misalignment, and occlusion;
- compare `StereoBM` with `StereoSGBM`;
- vary projected-dot density on a textureless surface and measure trusted
  disparity coverage;
- give depth-sensing technologies their own practice: compare active stereo,
  structured light, time-of-flight cameras, and LiDAR without pretending they
  are all the same kind of magic;
- give keypoints and descriptors their own notebook: inspect how memorable spots
  and binary fingerprints are created, matched, and occasionally fooled, then
  enforce the rectified-stereo rule, estimate sparse depth, or use them for a
  small practical task such as image alignment.
