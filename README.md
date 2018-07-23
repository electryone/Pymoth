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

**Instance(id_number=-1, img_path=None, frame_index=None, bounding_box=None, coordinates=None, conf=None, state=None, color=None)**

- :param id_number: int: the unique identification number of the instance
- :param img_path: str: the path to the image that contains the instance
- :param frame_index: int: the index number of the frame that contains the instance
- :param bounding_box: np.array(1, 4): the bounding box of the instance (left, top, width, height)
- :param coordinates: np.array(1, 3): the world coordinates of the instance (...)
- :param conf: int: the detection confidence of the instance (default = -1)
- :param state: str: the human-readable state of the instance
- :param color: tuple: the color used when drawing the instance bounding box

### Attributes

**Private**
- self._bounding_box:  np.array(1, 4): the bounding box of the instance (left, top, width, height)
- self._coordinates: np.array(1, 3): the world coordinates of the instance (...)
- self._id: int: the unique identification number of the instance

**Public**
- self.color: tuple: the color used when drawing the instance bounding box
- self.conf: int: the detection confidence of the instance (default = -1)
- self.frame_index:  int: the index number of the frame that contains the instance
- self.img_path: str: the path to the image that contains the instance
- self.mode: str: human readable description of the instance mode ('bounding_box' or 'world_coordinates')
- self.state: str: the human-readable state of the instance

### Methods

**set_bounding_box(bounding_box)**
> Store the instance bounding box and set the instance mode

**set_coordinates(coordinates)**
> Store the instance world coordinates and set the instance mode

**set_id(id_number)**
> Sets the instance id. If the instance color is not already set, sets the instance color based on the instance id number
> - param id_number: int: the unique identification number of the instance

**get_bounding_box()**
> Returns np.array: the instance bounding box (left, top, width, height)

**get_rect()**
> Returns: np.array(1, 4): the instance rect (left, top, right, bottom)

**get_state()**
> Returns str: the human-readable state of the instance

**get_xywh()**
> Returns: np.array(1, 4): the bounding box defined by (left, top, width, height)

**get_id()**
> Returns: int: the unique identification number of the instance

**get_appearance(shape=None, keep_aspect=True)**
> Returns: np.array: the image of the instance
> - param shape: the required shape of the output image
> - param keep_aspect: bool: whether to keep the object aspect ratio or not when resizing


**draw(image, width=1, scale=1, show_ids=False)**
> Returns: np.array: the original image with the instance drawn
> - param image: np.array: the image on which to draw the instance
> - param width: int: the line width of the instance bounding box
> - param scale: int: the scale of the drawing
> - param show_ids: bool: whether or not to draw the instance id number
