# Wall-mounted AprilTag camera calibration

The AprilTag is fixed to a vertical wall and the camera moves in front of it.
The published tracking frame uses this right-handed convention when looking
toward the tag:

- `+X`: camera moves away from the wall and AprilTag;
- `+Y`: camera moves to the viewer's right;
- `+Z`: camera moves upward.

The AprilTag solver's raw axes are `[right, up, out of wall]`. Therefore the
calibrated position is:

```text
X = raw_tag_z - wall_x_origin_m
Y = raw_tag_x
Z = raw_tag_y
```

`Y=0` and `Z=0` occur when the camera optical centre is aligned with the tag
centre. `X=0` occurs at the configured perpendicular camera-to-tag distance:

```yaml
calibration.wall_x_origin_m: 0.500
```

Set this value to the desired neutral working distance measured in metres. A
practical procedure is:

1. Mount the tag vertically and measure its black-border edge for `tag.size_m`.
2. Put the camera directly in front of the tag, level it, and centre the tag in
   the image.
3. Measure the perpendicular distance from the tag plane to the camera's colour
   optical centre.
4. Write that distance to `calibration.wall_x_origin_m`.
5. Start the vision node and check:

```bash
ros2 topic echo /vision/camera_pose
```

At the neutral pose, position should be close to `(0, 0, 0)`. Moving away,
right, and up must respectively increase X, Y, and Z.

Outputs:

- `/vision/camera_pose`: calibrated pose in `camera_tracking`;
- `/vision/camera_pose_raw`: original camera-in-tag pose in `tag`.

The origin offset cancels when the runtime computes relative motion. It is
still important for making the displayed camera position meaningful and
repeatable.
