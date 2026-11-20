# Classical computer vision

Practice based initially on the five reference groups under
`_references/classic-cv/`, then extended wherever curiosity leads.

The first sequence covered:

1. thresholding, masks, morphology, connected components, and bounding boxes;
2. cameras, image acquisition, coordinates, and projection geometry;
3. image formation and geometric transformations;
4. stereo vision, disparity, depth, and 3D reconstruction;
5. smoothing, edges, line detection, Hough voting, least squares, and RANSAC.

Current notebooks:

- `01_color_detection_pipeline.ipynb` — color masks, morphology, and connected components;
- `02_camera_laboratory.ipynb` — sensors, camera settings, shutter timing, intrinsics, distortion, and distance.
- `03_coordinates_and_calibration.ipynb` — homogeneous coordinates, world-to-camera projection, pixel rays, tangential distortion, calibration, and pixel-to-ground geometry.
- `04_stereo_depth_and_point_clouds.ipynb` — stereo correspondence, block matching, disparity, metric depth, point clouds, active infrared, and sparse feature matching.
- `05_edges_hough_and_ransac.ipynb` — smoothing, edge filters, least-squares fitting, Hough voting, and robust line fitting with RANSAC.

Notebooks in this directory use a local two-digit sequence such as
`01_color_detection_pipeline.ipynb`.

## Next foundation practices

The first five notebooks built intuition for pixels, masks, cameras, stereo,
edges, and robust geometry. The next useful steps are:

1. **Contours and shape analysis** — boundaries, area, perimeter, moments,
   centroids, orientation, polygon approximation, and convex hulls.
2. **Keypoints, descriptors, and geometric matching** — distinctive points,
   ORB-style fingerprints, match filtering, homography, RANSAC, and a small
   image-alignment or panorama experiment.

Motion, optical flow, and tracking have grown suspiciously ambitious and now
have their own plan under [`../object_tracking/`](../object_tracking/). Classical
CV concepts remain their foundation; they simply no longer have to share a
bedroom.

Image pyramids and scale should follow when objects become unwilling to remain
the same size. A visual frequency-domain practice can later connect smoothing,
edges, spatial frequency, and periodic noise without turning into compulsory
Fourier suffering.

Whenever practical, these notebooks should gradually introduce real images.
Synthetic scenes are excellent witnesses because they tell us the truth; real
photographs provide shadows, clutter, reflections, and valuable complaints.

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
- after the keypoints practice, enforce the rectified-stereo rule and estimate
  sparse depth from reliable feature matches;
- apply smoothing, edge detection, and Hough line detection to a simple
  photograph containing strong straight lines. The synthetic road has served
  bravely; reality may now submit its complaints in a separate notebook.
