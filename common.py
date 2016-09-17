'''
common.py
Sha Lai
8/30/2016

This file mostly contains the important constants for the main file.
Modify only if you know what you are doing.
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

# The place holding tuple used for some undetermined coordinate or elements
# pending to be removed.
PLACE_HOLDER = (-1, -1)

# The relative position of the label of each node when displaying them.
# Alternatively if REL_POS = 'MIDDLE' is active, then the label will be placed
# around the middle of the frames.
# REL_POS = (15, 15)
REL_POS = 'MIDDLE'

# The starting index of the vertices or edges when they are displayed and when
# the user is providing indices.
BASE = 1

# When extracting the edges, nodes must be blocked, this number affects the
# size of each block, so it may also affect the accuracy of the process.
SCALE_PERCENTAGE = 0

# This constant is the threshold factor of the ditance between an end point of
# an edge and the center of a node, a value greater than 1 is recommended. Feel
# free to increase this value if there exists some undetected edges.
TOLERANCE_FACTOR = 1.1

# The following two constants is relavant to how to distinguish the background
# and the contents. If METHOD is 'STATIC' then the value of THRESHOLD will
# serve as a break point. Otherwise the program will use the most common color
# on the image to be the background color.
METHOD = 'DYNAMIC'
#METHOD = 'STATIC'
THRESHOLD = 127

# Indicating the font properties of the numbers displayed on the image.
FONT_COLOR = (0, 0, 255)
FONT_COLOR_G = (FONT_COLOR[-1], 0, FONT_COLOR[0])
FONT_THICKNESS = 1 # must be an integer
FONT_SIZE = 0.2

# Indicating the label(rectangle) properties.
RECT_COLOR = (0, 0, 255) # in an order Blue-Green-Red, as opposed to RGB
RECT_COLOR_G = (RECT_COLOR[-1], 0, RECT_COLOR[0]) # use this one when running in GUI
RECT_THICKNESS = 2 # must be an integer

# Indicating the shape and size of the kernel when performing erosion/dilation and the times
# they will be performed, these values depend on the quality of an image.
KERNEL_SHAPE = "RECT"
KERNEL_SIZE = (5, 5)
KERNEL_STR_MAP = {'c' : "CROSS", 'C' : "CROSS", 'e' : "ELLIPSE", 'E' : "ELLIPSE", 'r' : "RECT", 'R' : "RECT"}
DILATION_ITR = 0
EROSION_ITR = 0

# Indicating how wide the normal distribution should be.
SIGMA = 1

# Indicating the max value in a binary image.
BIN_MAX = 1

# More printing statements if True.
DEBUG_MODE = False

# Indicating the default saved image's suffix.
SUFFIX = ".png"

# for GUI only
OUTPUT_FONT = 8
GRAPH_SIZE_MAX = (1024, 768)
OUTPUT_CONSOLE_HEIGHT = 200


def is_valid_type(input, function, error = "Error: invalid input!"):
   """Checks if the input is valid using a provided function. This is most
   likely to be used to test if the string contains some integer.
   Parameters
   ----------
   input : string
      The input from the user.
   function : function
      A function that will be used to test the input.
   error : string
      An customized error message that will be displayed to the user if the
      result is invalid.
   Return
   ------
   is_valid : boolean
   """
   is_valid = False
   try:
      temp = function(input)
      is_valid = True
   except:
      print(error)
   return is_valid