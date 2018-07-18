# PyTrack

## PyTrack(img_dirs, det_paths, info_paths, gt_paths=None)

A framework for organising data in Multiple Object Tracking applications.

### Attributes

- **info:** A dictionary of info Namespaces where keys are the name specified in the info file
- **det:** A dictionary of detection Sequences where keys are the name of the sequence
- **gt:** A dictionary of ground truth Sequences where keys are the name of the sequence

## Sequence()

An object to store object states throughout a video sequence.

### Attributes

- **info:** A Namespace containing information about the Sequence
- **frames:** A list of Frame objects

### Methods

**load_frames(img_dir, label_paths, info)**

**init_frames(info=None, n=None, img_dir=None)**

**new_frame(img_path=None)**

**set_frame_paths(img_dir)**

**set_frame_path(frame, path)**

**create_instance(frame, kwargs)**

**add_instance(frame, instance)**

**get_images(width=1, scale=1, draw=False, show_ids=False)**

**get_n_frames()**

**get_n_ids()**

**get_n_instances(id=None)**

**get_instances(id=None)**

**get_ids()**

**get_boxes(id=None)**

**get_rects(id=None)**

**get_xywh(id=None)**

**get_conf(id=None)**

**get_appearances(id=None, shape=None)**

**show(scale=1, width=1, draw=False, show_id=False)**

**get_appearance_pairs(shape=(128, 128, 3), seed=None)**

## Frame

An object to store instances from single frame

### Attributes

- **index:** the index in the frame in the sequence
- **img_path:** the path to the image of the frame
- **instances:** list of Instance objects

### Methods

**create_instance(kwargs)**

**add_instance(instance)**

**get_image(width=1, scale=1, draw=False, show_ids=False)**

**get_n_instances()**

**get_ids()**

**get_n_ids()**

**get_boxes()**

**get_xywh()**

**get_rects()**

**get_conf()**

**get_appearances(shape=None)**

## Instance

An object to store the state of an obejct at a single point in time

### Attributes

- **id:** the unique id of the instance (-1 if not specified)
- **frame_index:** the index of the parent frame
- **img_path:** the path to the image of the frame
- **bounding_box:**
- **conf:**
- **coordinates:**
- **color:**
- **mode:**

### Methods
