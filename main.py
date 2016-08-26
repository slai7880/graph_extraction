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
   
# Takes a directory path and a keyword string as parameters, lists all the
# files inside the given directory, asks the user to indicate the desired image
# file to be processed, and returns the image as well as the gray version of
# the image.
def get_image(dir_path, keyword):
   image = None
   image_gray = None
   response = ''
   valid = False
   while response == '' or valid == False:
      input_dir = listdir(dir_path)
      print("Files in the input directory:")
      print_list(input_dir)
      response = input("Please provide the file by index of the " + keyword + 
         ": ")
      if is_valid_type(response, int, "Please provide an integer!"):
         index = int(response)
         if index >= 0 + BASE and index < len(input_dir) + BASE:
            try:
               image = cv2.imread(dir_path + input_dir[index - BASE])
               image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               valid = True
               print("Selected " + keyword + " file: " + 
                  str(input_dir[index - BASE]))
            except:
               print("Error: the " + keyword + " file is invalid or \
                  cannot be processed.")
               response = ''
         else:
            print("Error: index out of bound!\n")
   return image, image_gray

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
'''
Even though these functions are not likely to be used multiple times, they are
organized in this way for better readability.
'''

# Part I: Core Functions

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

def get_edges(contours, nodes_center):
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
         
         
# =========================================================================== #
# Part II: User Interacting Functions

# Asks the user to provide the names of the images, returns the opencv images
# of the graph and the template.      
def get_images(show_graph = False, show_template = False):
   graph = None
   graph_gray = None
   template = None
   template_gray = None
   
   # involve user interaction
   graph, graph_gray = get_image(GRAPH_PATH, "graph") 
   template, template_gray = get_image(TEMPLATE_PATH, "vertex example")
   
   if show_graph:
      cv2.startWindowThread()
      cv2.imshow(GRAPH, graph)
      cv2.waitKey(1)
   if show_template:
      cv2.startWindowThread()
      cv2.imshow(TEMPLATE, template)
      cv2.waitKey(1)
   break_point = get_threshold(graph_gray)
   return graph, graph_gray, template_gray, break_point

# Takes three parameters, a graph_display that is used to present to the user
# a graph_work that is used to perform algorithms, and a template that stores
# an example of a node. This function first asks the user to give an
# approximate amount, say n, of vertices in a graph, and then it will find the
# first n pieces in the graph image that match the template. This step will
# keep going untill the user enter a 0 as input. Next it will label all the
# found verteices and ask the user to point out which one(s) may be false, the
# user must either give a sequence of ingeters indicating the indices of the
# vertices that they do not want, or type "done" to proceed to the next step.
def find_vertices(graph_display, graph_work, template, tW, tH):
   graph_display2 = graph_display.copy() # will be used in the removing part
   nodes = [] # stores the upper-left coordinates of the vertices
   nodes_center = [] # stores the center coordinates of the vertices
   user_input = 1
   while user_input > 0:
      user_input = input("How many vertices are you looking for?(0 means " + 
         "done) ")
      try:
         user_input = int(user_input)
      except:
         user_input = -1
         print("Cannot recognize the input, please provide a number.")
      while user_input < 0:
         user_input = input("How many vertices are you looking for?(0 means" + 
            " done) ")
         try:
            user_input = int(user_input)
         except:
            print("\nCannot recognize the input, please provide a number.")
            
      locate_vertices(user_input, graph_work, template, tW, tH, nodes)
      update_display(graph_display, nodes, tW, tH, False)
      cv2.startWindowThread()
      cv2.imshow("Vertices", graph_display)
      cv2.waitKey(1)
      print("Current vertices:")
      print_list(nodes)
      
   cv2.destroyWindow(GRAPH)
   cv2.destroyWindow("Vertices")
   
   # attempts to remove all the false vertices
   user_input = ''
   while not user_input == DONE:
      graph_display3 = graph_display2.copy()
      update_display(graph_display3, nodes, tW, tH)
      cv2.startWindowThread()
      cv2.imshow("Vertices with Labels", graph_display3)
      cv2.waitKey(1)
      user_input = get_valid_list(user_input, "Indicate non-vertex element " +
                                    "in the list in a sequence of indices " +
                                    "or \"done\" to proceed to next step:\n",\
                                    len(nodes))
      if user_input != DONE:
         nodes = remove_indices(user_input, nodes)
         print("Current vertices:")
         print_list(nodes)
         user_input = ''
   for node in nodes:
      nodes_center.append((int(node[0] + tW / 2), int(node[1] + tH / 2)))
   print("Current vertices:")
   print_list(nodes)
   return nodes, nodes_center

# Takes a list of vertices as the parameter, allows the user to select
# their prefered method to correct the index of each vertex. When entering
# the base, the input must be either 0 or 1.
# Note: the second of sorting is not safe for large size graph since if
# the user's input, when sorted in order, is not an arithmetic sequence,
# then the program will be broken.
def sort_vertices(nodes):
   answer = ''
   while answer == '':
      answer = input("Do you want to sort the vertices? (y/n): ")
      if answer[0] == 'y' or answer[0] == 'Y':
         result = [(0, 0)] * len(nodes)
         
         sorting_option = 0
         while sorting_option == 0:
            print("Please indicate the method by index you want to help" + 
               " sorting:")
            print("1. One-by-one,")
            print("2. Once-for-all.")
            response = input("Your choice? ")
            if is_valid_type(response, int, "The input is not an integer."):
               sorting_option = int(response)
               if sorting_option < 1 or sorting_option > 2:
                  print("Please provide a valid integer!")
                  sorting_option = 0
         
         if sorting_option == 1:
            index_list = []
            for i in range(len(nodes)):
               valid = False
               while valid == False:
                  index = input("What's the correct index value of the " + 
                     "vertex " + str(BASE + i) + ". " + str(nodes[i]) + "? ")
                  if is_valid_type(index, int, "Please provide a valid integer."):
                     index = int(index)
                     if index < BASE or index >= BASE + len(nodes):
                        print("Error: index out of bound!\n")
                     elif index in index_list:
                        print("Duplicate index detected, please provide " + 
                           "another one.")
                     else:
                        valid = True
                        index_list.append(index)
         elif sorting_option == 2:
            valid = False
            while valid == False:
               index_list = []
               indices = input("Please provide a sequence of correct indices" + 
                  " for each vertex or \"done\" to proceed to next step:\n")
               try:
                  indices = indices.split()
                  for i in range(len(indices)):
                     if int(indices[i]) in index_list:
                        print("Duplicate index detected, please provide " + 
                           "another one.")
                        break
                     else:
                        index_list.append(int(indices[i]))
                  if not len(index_list) == len(nodes):
                     print("Not enough integers or too many of them, please " + 
                        "try again.")
                  elif not max(index_list) + 1 - BASE == len(index_list):
                     print("The given input is not a valid arithmetic sequence!")
                  else:
                     valid = True
               except:
                  print("Please provide a sequence of valid integers.")
         else:
            print("Cannot sort the vertices, check the method indicating value.")
            exit(1)
         for i in range(len(index_list)):
            result[index_list[i] - BASE] = nodes[i]
         nodes = result
         print("Updated list of vertices:")
         print_list(nodes)
      elif answer[0] == 'n' or answer[0] == 'N':
         break
      else:
         answer = ''
         print("Please answer with y/n.")





# From the contours extracts all edges. For each contour, select the first and
# the middle element in the list to be the end points of an edge segment, this
# is because whatever stored in a "contour" is essentially all the pixels that
# surrounds an edge and luckily the starting element seems to be very close to
# one of the end points. Then for all the vertices examines the distance from
# their center to the end points, choosees the vertex whose distance is within
# the tolerance and is the smallest compared with the rest.
def extract_edges(nodes, nodes_center, radius, graph, graph_gray, tW, tH, \
                  break_point):
   contours = extract_contours(graph_gray, nodes, tW, tH, break_point)
   print("Retrieving edge data....")
   E, minimum, edge_pos, edge_to_contour = get_edges(contours, nodes_center)
   # ask the user to remove redundancy
   user_input = ''
   while not user_input == DONE:
      print("Number of edges detected: " + str(len(E)))
      edges_display = update_edges_display(E, edge_to_contour, graph.copy())
      cv2.startWindowThread()
      cv2.imshow("Edges with Labels", edges_display)
      cv2.waitKey(1)
      user_input = get_valid_list(user_input, "Indicate non-vertex element " +
                                    "in the list in a sequence of indices " +
                                    "or \"done\" to proceed to next step:\n",\
                                    len(E))
      if user_input != DONE:
         E = remove_edges(user_input, E)
         user_input = ''
      
   #print("Goal distance = " + str(radius * TOLERANCE_FACTOR))
   #print("Minimum:")
   #print_list(minimum)
   return E

#                               End of Section                                #
###############################################################################
###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":
   # Obtain the files.
   graph, graph_gray, template, break_point = get_images(True)
   
   # Process the template.
   template, (tH, tW), radius = process_template(template, break_point)
   
   # Find all the vertices. In particular variable nodes stores a list of
   # nodes' upper-right corner.
   nodes, nodes_center = find_vertices(graph.copy(), graph_gray.copy(), \
      template, tW, tH)
   
   # If neccesary, sort the vertices such that the order matches the given one.
   sort_vertices(nodes)
   
   # Gets the edges of the graph.
   E = extract_edges(nodes, nodes_center, radius, graph, graph_gray, tW, tH, \
                     break_point)

   print("Number of edges detected: " + str(len(E)))
   print("Edges:")
   print(E)
   halt = input("HALT (press enter to end)")