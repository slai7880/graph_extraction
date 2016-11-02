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
import sys
# from sys import exit, stderr
import numpy as np
import imutils
import cv2
from math import sqrt, inf, fabs, exp, pi, pow
from scipy.stats import mode
from datetime import datetime

#                               End of Section                                #
###############################################################################
###############################################################################
#                             Constants Initiation                            #
sys.stderr = open("stderr.log", 'w')
sys.stderr.write(str(datetime.now()) + "\n")
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
      
#                               End of Section                                #
###############################################################################