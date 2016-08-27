'''
main.py
Sha Lai
8/26/2016

This program helps the user to extract a mathematical graph from a digital
image and stores the data in some data structure.

(to be continued)

'''


from sys import exit
import numpy as np
import imutils
import cv2
from common import *
from thinning import *
from math import sqrt, inf, fabs
from scipy.stats import mode
from os import listdir

###############################################################################
#                             Helper Functions                                #

# Prints a list of elements one per line with their indices.
def print_list(list):
   for i in range(len(list)):
      print(str(i + BASE) + ". " + str(list[i]))

# Given an input argument, a testing function and an error message, tests the
# argument with the function. This is most likely used to test if the user
# input is valid.
def is_valid_type(input, function, error = "Error: invalid input!"):
   is_valid = False
   try:
      temp = function(input)
      is_valid = True
   except:
      print(error)
   return is_valid

# Given a string of user_input, a prompt sentence, keep asking the user to
# provide a list of indices until a valid one(can be DONE) is entered.
def get_valid_list(user_input, prompt_sentence, list_length):
   valid = False
   while valid == False:
      while user_input == '':
         user_input = input(prompt_sentence)
      indices = user_input.split()
      valid = True
      if user_input != DONE:
         for i in indices:
            if not is_valid_type(i, int, "Invalid input detected!"):
               valid = False
               user_input = ''
            elif int(i) < BASE or int(i) >= BASE + list_length:
               print("Error: index out of bound!\n")
               valid = False
               user_input = ''
   return user_input

# Takes a string window_name, a graph image, and a list of nodes, creates a
# window with the given name, displays the graph, and optionally labels each
# node with a red rectangle and a number indicating it's index in the list.
def update_display(graph_display, elements, tW, tH, show_indices = True):
   for i in range(len(elements)):
      position = (elements[i][0] + int(tW / 3), elements[i][1] + 
         int(tH * 2 / 3))
      if not isinstance(REL_POS, str):
         x = abs(elements[i][0] + REL_POS[0])
         y = abs(elements[i][1] + REL_POS[1])
         if x >= graph_display.shape[1]:
            x -= 2 * REL_POS[0]
         if y >= graph_display.shape[0]:
            y -= 2 * REL_POS[1]
         position = (x, y)
      if show_indices:
         cv2.putText(graph_display, str(i + BASE), position, 
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS, 
            cv2.LINE_AA)
      cv2.rectangle(graph_display, elements[i], (elements[i][0] + tW, 
         elements[i][1] + tH), RECT_COLOR, RECT_THICKNESS)
   #cv2.startWindowThread()
   # cv2.imshow(window_name, graph_display)
   # cv2.waitKey(1)

# Takes a set of edges and a copy of the original image, adds the indices of
# the edges onto the image, returns the updated version.
def update_edges_display(E, edge_to_contour, graph_copy):
   contour_center = []
   for e in E:
      contour = edge_to_contour[e]
      contour_center.append(contour[int(len(contour) / 4)])
   for i in range(len(contour_center)):
      cv2.putText(graph_copy, str(i + BASE), (contour_center[i][0][0], \
         contour_center[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, \
         FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
   return graph_copy
      
# Returns the threshold of color value which is used to distinguish the
# background and the content.
def get_threshold(image_gray, show_detail = False):
   # use a set value
   if METHOD == 'STATIC':
      if show_detail:
         print("METHOD = " + METHOD)
         print(THRESHOLD + " is being used.")
      return THRESHOLD
      
   # evaluate the mode of all the pixels in the image
   elif METHOD == 'DYNAMIC':
      if show_detail:
         print("METHOD = " + METHOD)
         print('mode = ' + str(mode(mode(image_gray)[0][0])[0][0]))
      return mode(mode(image_gray)[0][0])[0][0] - 1
   else:
      exit('Cannot recognize the METHOD, please check the common file.')

# Computes and returns the Euclidean distance between two points.
def get_distance(p1, p2):
   return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

# Takes a string of user input, based on the list of indices contained in this
# input string, returns a new list of nodes where all the false vertex indices
# have been removed.
def remove_indices(user_input, nodes):
   index_remove = user_input.split()
   for i in index_remove:
      nodes[int(i) - BASE] = PLACE_HOLDER
   nodes = [element for element in nodes if element != PLACE_HOLDER]
   return nodes

# Takes a string of user input, based on the list of indices contained in this
# input string, returns a new list of edges where all the false vertex indices
# have been removed.
def remove_edges(user_input, E):
   index_remove = user_input.split()
   removing = []
   for i in index_remove:
      try:
         removing.append(int(i) - BASE)
      except:
         print("Invalid input, please try again!")
   E = [E[i] for i in range(len(E)) if not i in removing]
   return E


#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

# Processes the template in order to retrieve helpful information.
def process_template(template, break_point):
   template = cv2.Canny(template, 50, 200)
   (tH, tW) = template.shape[:2]
   
   # this will serve as a threshold of the distance from some end point of
   # an edge to the center of a vertex
   radius = sqrt(pow((1 + 2 * SCALE_PERCENTAGE) * tH, 2) + pow((1 + 
      2 * SCALE_PERCENTAGE) * tW, 2)) / 2
   return template, (tH, tW), radius
   

# Takes an integer as the desired amount of vertices that the user
# is looking for, a hidden image graph_work to work with, along
# with some other necessary parameters, locates all the pieces in
# the image that match the template.
def locate_vertices(amount, graph_work, template, tW, tH, nodes):
   # perform multi-scale template matching, contributed by Adrian Rosebrock
   # from PyImageResearch
   for i in range(amount):
      found = None
      for scale in np.linspace(0.2, 1.0, 20)[::-1]:
         resized = imutils.resize(graph_work, width = \
            int(graph_work.shape[1] * scale))
         r = graph_work.shape[1] / float(resized.shape[1])
        
         if resized.shape[0] < tH or resized.shape[1] < tW:
            break
            
         edged = cv2.Canny(resized, 50, 200)
         result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
         (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
         if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
      (_, maxLoc, r) = found
      (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
      (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

      # graph_work is not intended to be displayed
      cv2.rectangle(graph_work, (startX, startY), (endX, endY), \
         (255, 255, 225), cv2.FILLED)
      nodes.append((int(maxLoc[0] * r), int(maxLoc[1] * r)))


   
# Takes an image of graph in gray scale, a list of nodes storing their upper-
# left corners, the size values of a block, a break point which is used to
# distinguish the background and the main contents of the image, and a boolean
# value indicating if the image needs to be thinned(doing so may improve the
# accuracy of the extraction but will definitely take more time), extracts all
# the contours except for the ones surrounding the vertices.
def extract_contours(graph_gray, nodes, tW, tH, break_point, thin = True):
   print("Processing the graph, this step may take some time, please wait....")
   graph_gray_bin = graph_gray.copy()
   # Hides all the nodes on a draft version of the graph image.
   for i in range(len(nodes)):
      # The start corner and the end corner, both in the form (x, y).
      upper_left = (nodes[i][0] - int(tW * SCALE_PERCENTAGE), nodes[i][1] - 
         int(tH * SCALE_PERCENTAGE))
      bottom_right = (nodes[i][0] + int(tW * (1 + SCALE_PERCENTAGE)), \
         nodes[i][1] + int(tH * (1 + SCALE_PERCENTAGE)))
      
      # Places a block at each node.
      cv2.rectangle(graph_gray_bin, upper_left, bottom_right, \
         (255, 255, 225), cv2.FILLED)
   
   
   # Performs image thinning if neccessary.
   cv2.startWindowThread()
   #cv2.imshow("graph_gray_bin", graph_gray_bin)
   #cv2.waitKey(1)
   ret, graph_gray_bin = cv2.threshold(graph_gray_bin, break_point, 1, \
      cv2.THRESH_BINARY_INV)
   if thin:
      graph_gray_bin = thinning(graph_gray_bin)
   ret, result = cv2.threshold(graph_gray_bin, 0, 255, cv2.THRESH_BINARY)
   #cv2.startWindowThread()
   #cv2.imshow("result", result)
   #cv2.waitKey(1)
   
   '''
   # Use a simple way.
   high_thresh, thresh_im = cv2.threshold(graph_gray_bin.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   result = cv2.Canny(graph_gray_bin, high_thresh, high_thresh)
   '''
   
   # Extracts contours.
   print("Extracting contours....")
   contours_display, contours, hierarchy = cv2.findContours(result, \
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(contours_display, contours, 1, (255, 255, 255), 3)
   print("Number of contours detected: " + str(len(contours)))
   #cv2.startWindowThread()
   #cv2.imshow("contouts", contours_display)
   #cv2.waitKey(1)
   return contours

def get_edges(contours, nodes_center, radius):
   E = [] # where the outputs are stored
   edge_pos = [] # stores the estimated midpoints of edges
   minimum = [] # stores the recorded minimum distances, for debugging purpose
   # maps edges to their corresponding contours, for debugging purpose
   edge_to_contour = {} 
   for i in range(len(contours)):
      edge = contours[i]
      end1 = -1
      end2 = -1
      end1_pos = (edge[0][0][0], edge[0][0][1])
      end2_pos = (edge[int(len(edge) / 2)][0][0], edge[int(len(edge) / 2)][0][1])
      d_min1 = inf
      for i in range(len(nodes_center)):
         d_temp = get_distance(end1_pos, nodes_center[i])
         if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min1:
            d_min1 = d_temp
            end1 = i + BASE
      d_min2 = inf
      for i in range(len(nodes_center)):
         d_temp = get_distance(end2_pos, nodes_center[i])
         if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min2:
            d_min2 = d_temp
            end2 = i + BASE
      # Only records an edge if the all of the following hold:
      # 1. Both end points are connected to some vertex.
      # 2. The edge has not yet been detected.
      # 3. The end points do not point to the identical vertex.
      if (not end1 == -1) and (not end2 == -1) and\
         (not (end1, end2) in E) and (not (end2, end1) in E) and\
         (not end1 == end2):
         E.append((end1, end2))
         edge_to_contour[(end1, end2)] = edge
         minimum.append((d_min1, d_min2))
         edge_pos.append([end1_pos, end2_pos])
   return E, minimum, edge_pos, edge_to_contour
         
#                               End of Section                                #
# =========================================================================== #