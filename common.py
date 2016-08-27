'''
common.py
Sha Lai
8/4/2016

This file mostly contains the important constants for the main file. Modify only if you know what you are doing.
'''
from os import sep

# The path of input images. By default it is an empty string, so the images are
# in the same folder as the program.
GRAPH_PATH = 'graph_input' + str(sep)
TEMPLATE_PATH = 'vertex_example' + str(sep)


# Some default window names.
GRAPH = "graph"
TEMPLATE = "template"

# The input indicating that the step is complete, only used to test uesr input.
DONE = 'done'

# The replacing element for a false node in the list.
PLACE_HOLDER = (-1, -1)

# The relative position of the label of each node when displaying them. Alternatively
# if REL_POS = 'MIDDLE' is active, then the label will be placed around the middle of the frames.
# REL_POS = (15, 15)
REL_POS = 'MIDDLE'

# The starting index of the vertices or edges when they are displayed and when the user is
# providing indices.
BASE = 1

# When extracting the edges, nodes must be blocked, this number affects the size of each block,
# so it may also affect the accuracy of the process.
SCALE_PERCENTAGE = 0.6

# This constant is the threshold factor of the ditance between an end point of an edge and the
# center of a node, a value greater than 1 is recommended. Feel free to increase this value if
# there exists at least one undetected edge.
TOLERANCE_FACTOR = 1.6

# The following two constants is relavant to how to distinguish the background and the contents.
# If METHOD is 'STATIC' then the value of THRESHOLD will serve as a break point. Otherwise the
# program will use the most common color on the image to be the background color.
METHOD = 'DYNAMIC'
#METHOD = 'STATIC'
THRESHOLD = 127

# Indicating the font properties of the numbers displayed on the image.
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1 # must be an integer
FONT_SIZE = 0.4

# Indicating the label(rectangle) properties.
RECT_COLOR = (0, 0, 255)
RECT_THICKNESS = 2 # only integer allowed

# for GUI only
OUTPUT_FONT = 8
GRAPH_SIZE_MAX = (1024, 768)