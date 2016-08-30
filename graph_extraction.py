'''
main.py
Sha Lai
8/26/2016

This program helps the user to extract a mathematical graph from a digital
image and stores the data in some data structure.

The first step before using this program is to crop an example of the vertices,
and then pass the template to the function process_template so that we can get
a new version where only the edges remain together with some helpful values
which will be used in later procedures.

To begin the user needs to call the function locate_vertices with an approaximate
number of vertices, a copy of the gray-scale image, the template and its size
values obtained in the previous step, and a list of nodes(this can be empty).
The function implements the multi-scale template matching technique, that is,
the function rescales the image, for each scale, it finds a piece of the image
which matches the template the most by calling a function matchTemplate from
the OpenCV package and store the upper-left coordinate of that piece into the
provided list nodes. In order to prevent the function from finding the same
piece of the image, everytime a piece is found, a white block is placed right
on its location so that the piece will be covered up. Note that this is done on
a copy of the image so that the original one will not be damaged. This function
may need to be called multiple times in order to find all the vertices. If
there is still one or more vertices remaining on the image undetected after a
few times then the image cannot be handled by this program. Otherwise, when all
the vertices are found the user must examine the node list, indicate the
indices of false vertices, and use the function remove_indices to get rid of
them. Be careful about the impact of the BASE value. Lastly the user may also
want to call teh function get_center_pos to obtain the center coordinates of
the true vertices.

Next we want to find the edges. There are two main steps involved. Firstly we
want to extract the contours, that is, the connected and pixels that surround
a particular area usually with some color different from the background on the
image. This can be done by calling the function extract_contours. It firstly
put a block on each of the vertices so that their pixels will not be taken into
account, and then it optionally attempts to thin the image using zhang-seun's
algorithm so that only the skeletons remain. After that it calls another OpenCV
function findContours to obtain the contours that surround all the lines or
curves on the thinned image.

With the contours which are stored as lists of pixels, the user can proceed to
the next step by using the function get_edges to obtain the edges of the graph.
Even though we only have contour pixels rather than the pixels of the edges
themselves, since here a contour is merely the pixels that surround a certain
edge's skeleton, I claim that analyzing the contour pixels is accurate enough.
However due to the way in which the contours are stored, it is likely to be
challenging to determine which pixels are closed to the endpoints of an edge.
There are two ways to solve this. The first method is rather simple. By
observation, the function findContours appears to tend to store the contours in
a way such that the pixels near an endpoint are located at the beginning, and
hence also the end of the list. Then intuitively we can approximate the
coordinates by selecting the first and the middle elements in the list. This
method works well in terms of efficiency and accuracy when the image is in high
quality, while it loses the second advantage when the quality is not that high,
hence the need for the second method. The second approach locates the two 
endpoints, or to be more precisely, two pixels that are adjacent the two
endpointsone respectively one after another. The idea is straightforward: for
each contour, look for a pixel-vertex pair where the pixel is contain in the
contour and the distance between this two elements is minimized, and then look
for another pixel-vertex pair with the same properties except that the new
pixel and the new vertex are both different from the ones obtained previously.
The test to determine which vertex each endpoint is pointing to is identical in
both of the approaches: an endpoint is linked to a vertex if the distance lies
within a certain range and it is the lowest among all the endpoint-vertex pairs.
An edge can be recored if the last set of tests are passed. These tests help
clearing false edges, duplicated ones as well as self-edges.

'''


from sys import exit
import numpy as np
import imutils
import cv2
from common import *
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


# Takes an image, a list of vertices, the size values of the block obtained
# earlier, a boolean value indicating if the indices will be displayed, and
# another boolean indicating if the program is run in console, put down a
# customized block at each of the locations stored in the list on the given
# image.
def draw_vertices(graph_display, vertices, tW, tH, show_indices = True, using_console = True):
   if not using_console:
      rect_color = RECT_COLOR_G
      font_color = FONT_COLOR_G
   else:
      rect_color = RECT_COLOR
      font_color = FONT_COLOR
   for i in range(len(vertices)):
      position = (vertices[i][0] + int(tW / 3), vertices[i][1] + 
         int(tH * 2 / 3))
      if not isinstance(REL_POS, str):
         x = abs(vertices[i][0] + REL_POS[0])
         y = abs(vertices[i][1] + REL_POS[1])
         if x >= graph_display.shape[1]:
            x -= 2 * REL_POS[0]
         if y >= graph_display.shape[0]:
            y -= 2 * REL_POS[1]
         position = (x, y)
      if show_indices:
         cv2.putText(graph_display, str(i + BASE), position, 
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, font_color, FONT_THICKNESS, 
            cv2.LINE_AA)
      cv2.rectangle(graph_display, vertices[i], (vertices[i][0] + tW, 
         vertices[i][1] + tH), rect_color, RECT_THICKNESS)

# Takes a set of edges and a copy of the original image, adds the indices of
# the edges onto the image, returns the updated version.
def draw_edges(graph_copy, edges_center, using_console = True):
   if not using_console:
      font_color = FONT_COLOR_G
   else:
      font_color = FONT_COLOR
   for i in range(len(edges_center)):
      cv2.putText(graph_copy, str(i + BASE), (edges_center[i][0],\
                  edges_center[i][1]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, \
                  font_color, FONT_THICKNESS, cv2.LINE_AA)
      
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

# Takes a list of nodes which stores the upper-left coordinates of the nodes,
# returns a list which stores the middle coordinates of the nodes under the
# same order in the given list.
def get_center_pos(nodes, tW, tH):
   nodes_center = []
   for node in nodes:
      nodes_center.append((int(node[0] + tW / 2), int(node[1] + tH / 2)))
   return nodes_center

# Takes a string of user input, based on the list of indices contained in this
# input string, returns a new list of edges where all the false vertex indices
# have been removed.
def remove_edges(user_input, E, edges_center):
   index_remove = user_input.split()
   removing = []
   for i in index_remove:
      try:
         removing.append(int(i) - BASE)
      except:
         print("Invalid input, please try again!")
   E = [E[i] for i in range(len(E)) if not i in removing]
   edges_center = [edges_center[i] for i in range(len(edges_center))\
                     if not i in removing]
   return E, edges_center

# The following four functions implement zhang-seun's image thinning algorithm.
# Testing if the current pixel satisfies the required statement group #1.
def test1(p, A, B):
    return p[0] == 1 and \
           B >= 2 and B <= 6 and \
           A == 1 and \
           (p[1] * p[3] * p[5] == 0) and \
           (p[3] * p[5] * p[7] == 0)

# Testing if the current pixel satisfies the required statement group #2.
def test2(p, A, B):
    return p[0] == 1 and \
           B >= 2 and B <= 6 and \
           A == 1 and \
           (p[1] * p[3] * p[7] == 0) and \
           (p[1] * p[5] * p[7] == 0)

# Examines all the pixels in a given images, except for the outer ones, using
# a provided test to collect all the satisfied pixels and returns the list.
def examine(mat, test):
    output = []
    for i in range(1, mat.shape[0] - 1):
        for j in range(1, mat.shape[1] - 1):
            p = (int(mat[i, j]), int(mat[i - 1, j]), int(mat[i - 1, j + 1]), \
                  int(mat[i, j + 1]), int(mat[i + 1, j + 1]), \
                  int(mat[i + 1, j]), int(mat[i + 1, j - 1]), \
                  int(mat[i, j - 1]), int(mat[i - 1, j - 1]))
            A = 0
            B = p[1]
            for k in range(2, len(p)):
                if p[k] == 1 and p[k - 1] == 0:
                    A += 1
                B += p[k]
            if p[1] == 1 and p[-1] == 0:
                A += 1
            if test(p, A, B):
                output.append((i, j))
    return output

# Takes a binay image as input, assuming that the main contents are represented
# as 1s while the rest are 0s, and then thins the contents so that only a
# skeleton of 1 pixel wide is remained.
def thinning(image):
   mat = image.copy()
   if mat.shape[0] < 3 or mat.shape[1] < 3:
      sys.exit("Invalid image input: " + str(mat.shape))
   keep_itr = True
   while keep_itr:
      output1 = examine(mat, test1)
      for coor in output1:
         mat[coor[0], coor[1]] = 0
      output2 = examine(mat, test2)
      for coor in output2:
         mat[coor[0], coor[1]] = 0
      if len(output1) + len(output2) == 0:
         keep_itr = False
   return mat

#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

# Extracts all the edges in the template, returns the new template along with
# the size values and the radius which will be used in later procedures.
def process_template(template):
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
      # rescale the image, for each scale, find the piece that has the best
      # matching score
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
   ret, graph_gray_bin = cv2.threshold(graph_gray_bin, break_point, 1, \
      cv2.THRESH_BINARY_INV) # obtain a binary image
   if thin:
      graph_gray_bin = thinning(graph_gray_bin)
   ret, result = cv2.threshold(graph_gray_bin, 0, 255, cv2.THRESH_BINARY)
   
   '''
   # Use a simple way.
   high_thresh, thresh_im = cv2.threshold(graph_gray_bin.copy(), 0, 255, \
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   result = cv2.Canny(graph_gray_bin, high_thresh, high_thresh)
   '''
   
   # Extracts contours.
   print("Extracting contours....")
   contours_display, contours, hierarchy = cv2.findContours(result, \
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(contours_display, contours, 1, (255, 255, 255), 3)
   print("Number of contours detected: " + str(len(contours)))
   return contours

# Takes a list of contours, a list of coordinates of the nodes' center and the
# radius values, returns the output list E and a list of middle points of the
# edges.
def get_edges(contours, nodes_center, radius, method = 2):
   E = [] # where the outputs are stored
   edges_center = [] # stores the estimated midpoints of edges, for debugging
   minimum = [] # stores the recorded minimum distances, for debugging
   # maps edges to their corresponding contours, for debugging purpose
   edge_to_contour = {} 
   for i in range(len(contours)):
      end1 = -1
      end2 = -1
      d_min1 = inf
      d_min2 = inf
      contour = contours[i]
      if method == 1:
         end1_pos = (contour[0][0][0], contour[0][0][1])
         end2_pos = (contour[int(len(contour) / 2)][0][0],\
                     contour[int(len(contour) / 2)][0][1])
         for i in range(len(nodes_center)):
            d_temp = get_distance(end1_pos, nodes_center[i])
            if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min1:
               d_min1 = d_temp
               end1 = i + BASE
         for i in range(len(nodes_center)):
            d_temp = get_distance(end2_pos, nodes_center[i])
            if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min2:
               d_min2 = d_temp
               end2 = i + BASE
      else:
         # first, find one of the two endpoints
         end1_pos = PLACE_HOLDER
         for j in range(len(contour)): # for each pixel in one contour
            for k in range(len(nodes_center)): # for each vertex
               d_temp = get_distance(contour[j][0], nodes_center[k])
               if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min1:
                  d_min1 = d_temp
                  end1_pos = contour[j][0]
                  end1 = k + BASE
      
         # now that one endpoint is found, need to find the other one
         end2_pos = PLACE_HOLDER
         for j in range(len(contour)): # for each pixel in one contour
            for k in range(len(nodes_center)): # for each vertex
               d_temp = get_distance(contour[j][0], nodes_center[k])
               if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min2 and\
                  (not end1_pos[0] == contour[j][0][0]) and\
                  (not end1_pos[1] == contour[j][0][1]) and\
                  k != end1 - BASE:
                  
                  
                  d_min2 = d_temp
                  end2_pos = contour[j][0]
                  end2 = k + BASE
               
      # Only records an edge if the all of the following hold:
      # 1. Both end points are connected to some vertex.
      # 2. The edge has not yet been detected.
      # 3. The end points do not point to the identical vertex.
      if (not end1 == -1) and (not end2 == -1) and\
         (not (end1, end2) in E) and (not (end2, end1) in E) and\
         (not end1 == end2):
         E.append((end1, end2))
         edge_to_contour[(end1, end2)] = contour
         minimum.append((d_min1, d_min2))
         edges_center.append([int((end1_pos[0] + end2_pos[0]) / 2),\
                              int((end1_pos[1] + end2_pos[1]) / 2)])
   return E, edges_center
         
#                               End of Section                                #
# =========================================================================== #