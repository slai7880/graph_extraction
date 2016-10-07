'''
common.py
Sha Lai
9/17/2016

This file mostly contains the important constants as well as some shared
functions. Modify with caution.
'''
###############################################################################
#                                  Imports                                    #

from os import sep, listdir
from sys import exit
import numpy as np
import imutils
import cv2
from math import sqrt, inf, fabs, exp, pi, pow
from scipy.stats import mode

#                               End of Section                                #
###############################################################################
###############################################################################
#                                 Constants                                   #

# Default directories of the input files.
GRAPH_PATH = 'graph_input' + str(sep)
TEMPLATE_PATH = 'vertex_example' + str(sep)


# Some default window names.
GRAPH = "graph"
TEMPLATE = "template"

# The input indicating that the step is complete, only used to test uesr input.
DONE = 'done'

# The place holding tuple used for some undetermined coordinate or elements
# pending to be removed or to be ignored.
PLACE_HOLDER_INT = -1
PLACE_HOLDER_COOR = (-1, -1)

# The relative position of the label of each node when displaying them.
# Alternatively if REL_POS = 'MIDDLE' is active, then the label will be placed
# around the middle of the frames.
#REL_POS = (20, 15)
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
FONT_SIZE = 0.5

# Indicating the label(rectangle) properties.
RECT_COLOR = (0, 0, 255) # in an order Blue-Green-Red, as opposed to RGB
RECT_COLOR_G = (RECT_COLOR[-1], 0, RECT_COLOR[0]) # when running in GUI
RECT_THICKNESS = 2 # must be an integer

# The default shape and size of the kernel when performing erosion/dilation and
# the times they will be performed, these values depend on the quality of an
# image.
KERNEL_SHAPE = "RECT"
KERNEL_SIZE = (5, 5)
KERNEL_STR_MAP = {'c' : "CROSS", 'C' : "CROSS", 'e' : "ELLIPSE",\
                  'E' : "ELLIPSE", 'r' : "RECTANGLE", 'R' : "RECTANGLE"}

# The default iteration amounts of each type of operations.
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

#                               End of Section                                #
###############################################################################
###############################################################################
#                                 Functions                                   #

def print_list(list):
   """Prints all the elements in the list one per line with an index.
   Parameters
   ----------
   list : list of elements
   Return
   ------
   None
   """
   for i in range(len(list)):
      print(str(i + BASE) + ". " + str(list[i]))

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


def get_binary_image(image_gray, break_point, max_value = BIN_MAX):
   """Converts a given grayscale image into a binary one.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      A grayscale image.
   break_point : int
      Any pixel with a value greater than this value will be set to max_value
      while the rest will be set to 0.
   max_value : int
      Any pixel with a value greater than break_point will be set to this
      number while the rest will be set to 0.
   Returns
   -------
   result : numpy matrix of integers
      The reversed binary image.
   """
   ret, result = cv2.threshold(image_gray.copy(), break_point, max_value,\
                                 cv2.THRESH_BINARY)
   return result


def get_binary_image_inv(image_gray, break_point, max_value = BIN_MAX):
   """Converts a given grayscale image into a binary one but the relative
   colors of the content and the background are reversed.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      A grayscale image.
   break_point : int
      Any pixel with a value lower than this value will be set to max_value
      while the rest will be set to 0.
   max_value : int
      Any pixel with a value lower than break_point will be set to this number
      while the rest will be set to 0.
   Returns
   -------
   result : numpy matrix of integers
      The reversed binary image.
   """
   ret, result = cv2.threshold(image_gray.copy(), break_point, max_value,\
                                 cv2.THRESH_BINARY_INV)
   return result


def show_binary_image(image_bin, window_name, save = False, break_point = 0,\
                        max = 255):
   """Displays a binary image, optionally saves it to the current directory.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      The binary image that is asked to be displayed.
   window_name : string
      The name of the display window.
   save : boolean
      If the value is True then the binary image will be saved as a file.
   break_point : int
      Any pixel with a value greater than this value will be set to max
      while the rest will be set to 0.
   max_value : int
      Any pixel with a value greater than break_point will be set to this
      number while the rest will be set to 0.
   Returns
   -------
   None
   """
   image_show = get_binary_image(image_bin, break_point, 255)
   cv2.startWindowThread()
   cv2.namedWindow(window_name)
   cv2.imshow(window_name, image_show)
   cv2.waitKey(1)
   if save:
      cv2.imwrite(window_name + SUFFIX, image_show)
      
#                               End of Section                                #
###############################################################################