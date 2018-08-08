# PyTrack

## Installation

```
pip install PyTrack
```

## Contents

### PyTrack(directory)

A Namespace for handling entire MOT datasets, specifically designed for MOTChallenge (MOT16 onwards)

Each sequence and its corresponding labels are loaded on PyTrack initialisation. 

- **directory**: A string indicating the path to the dataset

#### Methods

**summary(tab_size=2)**

Prints a summary of the sequences loaded into the PyTrack Object

- tab_size: int: number of spaces per tab

**get()**

Return __dict__ of the Namespeace

**add(dictionary)**

Update __dict__ with a new dictionary

#### Examples

Given layout of the MOTChallenge dataset as follows:

```
    .
    └── MOTChallenge
         ├── MOT15		# Not yet supported
         ├── MOT16
         ├── MOT17
         │   ├── test
         │   │   ├── MOT16-01
         │   │   │   ├── det
         │   │   │   │   └── det.txt
         │   │   │   ├── img1
         │   │   │   │   ├── 000001.jpg
         │   │   │   │   ├── 000002.jpg
         │   │   │   │   └── ...
         │   │   │   └── seqinfo.ini
         │   │   └── ...
         │   └── train
         │        ├── MOT16-02
         │        │   ├── det
         │        │   │   └── det.txt
         │        │   ├── gt
         │        │   │   └── gt.txt
         │        │   ├── img1
         │        │   │   ├── 000001.jpg
         │        │   │   ├── 000002.jpg
         │        │   │   └── ...
         │        │   └── seqinfo.ini
         │        └── MOT16-04
         └ ...
   
```
MOT16 would be loaded with:

```
tracks = PyTrack("path/to/MOTChallenge/MOT16")
```

Ground truth data for MOT16-02 would be accessed with:

```
tracks.train.MOT16_02.gt
```

Ground truth data for MOT16-02 would be displayed as a video with:

```
tracks.train.MOT16_02.gt.show(draw=True, show_ids=True, width=2)
```

### Sequence()

An object to store object states throughout a video sequence.

#### Attributes

- **info:** A Namespace containing information about the Sequence
- **frames:** A list of Frame objects

#### Methods

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

### Frame

An object to store instances from single frame

#### Attributes

- **index:** the index in the frame in the sequence
- **img_path:** the path to the image of the frame
- **instances:** list of Instance objects

#### Methods

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

### Instance

**Instance(id_number=-1, img_path=None, frame_index=None, bounding_box=None, coordinates=None, conf=None, state=None, color=None)**

- id_number: int: the unique identification number of the instance
- img_path: str: the path to the image that contains the instance
- frame_index: int: the index number of the frame that contains the instance
- bounding_box: np.array(1, 4): the bounding box of the instance (left, top, width, height)
- coordinates: np.array(1, 3): the world coordinates of the instance (...)
- conf: int: the detection confidence of the instance (default = -1)
- state: str: the human-readable state of the instance
- color: tuple: the color used when drawing the instance bounding box

#### Attributes

**Private**

- _bounding_box:  np.array(1, 4): the bounding box of the instance (left, top, width, height)
- _coordinates: np.array(1, 3): the world coordinates of the instance (...)
- _id: int: the unique identification number of the instance

**Public**

- color: tuple: the color used when drawing the instance bounding box
- conf: int: the detection confidence of the instance (default = -1)
- frame_index:  int: the index number of the frame that contains the instance
- img_path: str: the path to the image that contains the instance
- mode: str: human readable description of the instance mode ('bounding_box' or 'world_coordinates')
- state: str: the human-readable state of the instance

#### Methods

**set_bounding_box(bounding_box)**

Store the instance bounding box and set the instance mode

**set_coordinates(coordinates)**

Store the instance world coordinates and set the instance mode

**set_id(id_number)**

Sets the instance id. If the instance color is not already set, sets the instance color based on the instance id number

- id_number: int: the unique identification number of the instance

**get_bounding_box()**

Returns np.array: the instance bounding box (left, top, width, height)

**get_rect()**

Returns: np.array(1, 4): the instance rect (left, top, right, bottom)

**get_state()**

Returns str: the human-readable state of the instance

**get_xywh()**

Returns: np.array(1, 4): the bounding box defined by (left, top, width, height)

**get_id()**

Returns: int: the unique identification number of the instance

**get_appearance(shape=None, keep_aspect=True)**

Returns: np.array: the image of the instance

- shape: the required shape of the output image
- keep_aspect: bool: whether to keep the object aspect ratio or not when resizing


**show(image=None, draw=False, width=1, scale=1, show_ids=False)**

Returns: np.array: the frame image containing the instance

- image: np.array: the image on which to draw the instance
- draw: bool: whether or not to draw the object bounding box / world coordinates
- width: int: the line width of the instance bounding box
- scale: int: the scale of the drawing
- show_ids: bool: whether or not to draw the instance id number

### Utils

**iou()**

**iou2()**

**nms()**
