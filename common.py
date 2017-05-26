'''
common.py
Sha Lai
9/17/2016

This file mostly contains the important constants as well as some shared
functions. Modify with caution.

Copyright 2017 Sha Lai

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
###############################################################################
#                                  Imports                                    #

from os import sep, listdir
import sys
# from sys import exit, stderr
import numpy as np
import imutils
import cv2
from math import sqrt, inf, fabs, exp, pi, pow
from scipy.stats import mode
from datetime import datetime
from copy import deepcopy
from invariants import *
from time import time

#                               End of Section                                #
###############################################################################
###############################################################################
#                             Constants Initiation                            #
#sys.stderr = open("stderr.log", 'w')
#sys.stderr.write(str(datetime.now()) + "\n")
file = open("common.cst")
for line in file:
   if len(line) > 0 and line[0] != '#':
      exec(line)


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

def toTuples(list):
    """Converts a list of lists of element pairs to a list of tuples of
    element pairs.
    Parameters
    ----------
    list : List[[Object, Object]]
    Returns
    -------
    result : List[(Object, Object)]
    """
    result = []
    for c in list:
        if len(c) > 2 or len(c) < 1:
            print("Error: " + str(c))
            sys.exit()
        result.append((c[0], c[1]))
    return result
    
def toLists(list):
    """Converts a list of tuples of element pairs to a list of lists of
    element pairs.
    Parameters
    ----------
    list : List[(Object, Object)]
    Returns
    -------
    result : List[[Object, Object]]
    """
    result = []
    for c in list:
        if len(c) > 2 or len(c) < 1:
            print("Error: " + str(c))
            sys.exit()
        result.append([c[0], c[1]])
    return result

def draw_vectors(image_bw, starting_points, vectors):
   """Draws vectors on a given black and white image.
   Parameters
   ----------
   image_bw : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      255s and background marked by 0s.
   starting_points : List[(float, float)]
      Stores the starting point of each vector.
   vectors : List[List[(float, float)]]
      Stores the vectors at each starting point.
   Returns
   -------
   None
   """
   for i in range(len(starting_points)):
      pos = (int(starting_points[i][0]), int(starting_points[i][1]))
      cv2.putText(image_bw, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255,\
                  1, cv2.LINE_AA, False)
      for j in range(len(vectors[i])):
         cv2.arrowedLine(image_bw, pos,\
                        (int(np.ceil(pos[0] + 10 * vectors[i][j][0])),\
                        int(np.ceil(pos[1] + 10 * vectors[i][j][1]))), 255)


def two_norm(vec):
   """Just a simplified funciton to compute 2-norm.
   """
   return np.linalg.norm(vec, 2)

#                               End of Section                                #
###############################################################################