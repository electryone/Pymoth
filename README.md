# PyTrack

A framework for organising data in Multiple Object Tracking applications.

## PyTrack(img_dirs, det_paths, info_paths, gt_paths=None)

- **info:** A dictionary of info Namespaces where keys are the name specified in the info file
- **det:** A dictionary of detection Sequences where keys are the name of the sequence
- **gt:** A dictionary of ground truth Sequences where keys are the name of the sequence

# Sequence

An object to store object states throughout a video sequence

## Sequence()

- **info:** A Namespace containing information about the Sequence
- **frames:** A list of Frame objects

### load_frames(img_dir, label_paths, info)

### set_frame_paths(img_dir)

### set_frame_path(frame, path)

### new_frame(img_path=None)

### init_frames(info=None, n=None, img_dir=None)

### create_instance(frame, id, kwargs)

### add_instance(frame, instance)

### get_image(frame, width=1, scale=1, draw=False)

### get_n_frames()

### get_n_instances(frame=None, id=None)

### get_instances(frame=None, id=None)