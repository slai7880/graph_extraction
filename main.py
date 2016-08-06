'''
main.py
Sha Lai
8/5/2016

This program helps the user to extract a mathematical graph from a digital
image and stores the data in some structure.

The procedure consists of a few steps. I present part of them here since these
are related to user input.

1. Initialization
Before executing this program, the user must put the graph image in the
directory called graph_input. And then they must crop an example of the
vertices from the original image and put it in the folder vertex_example.
Although the user is allowed to modify the file common where a number of
constants are defined. However do so at your own risk!

After execution, the program first reads all the files in the input directory
for graph images. It lists all the files out and asks the user to indicate the
desired one by index. Same for selecting the vertex example.

2. Locating Vertices
Next the program will ask the user to provide the number of vertices they want
from the image. This number may not be accurate, and the request will be
repeated untill the user enter 0, implying that there is no vertex left unfound.
Due to the accuracy restriction of template matching which is the main technique
being used here, there is no gurantee that all the detected vertices are true
ones. However we still ask the user to keep staying at this step till all the
vertices are marked. Depending on the input images, there may be some vertices
in the image remain unmarked no matter how many times the user repeats the
finding procedure. In this case, I apologize that my program cannot handle the
given image.

After the user enter 0, the program will label all the vertices on the image,
and asks the user to give a sequence of indices of false vertices. The input
must be a sequence of valid integers separated by space, however a single
integer is allowed as well.

Next the user will be asked if they want to correct the order of the vertices.
The user may answer yes if they want the indices to match the orginal labels on
the image. There are two ways to correct. The first method, one-by-one, allows
the user to correct the labels one by one as the name implies; the other method,
once-for-all, allows the user to provide a sequence of correct labels, for
example, let v1, v2, v3, ..., vn be a list of detect vertices with some labels
assigned during the finding step, the user may enter a sequence of integers like
2, 1, 4, 6, ... to provide the correct indices for each vertex. Apparently the
second method is better if the graph size is rather small.

3. Detecting Edges
The program now should start to process the image. When it completes, it will
again display all the detected edges with labels and ask the user to provide a
sequence of indices indicating the false edges just like the find-vertices step.

At the end a list of edges will be printed to the console.

Minor Notes:
1. When interacting with the user, there is a chance that the user may enter
   some invalid input, some are dangerous enough to break the entire program.
   To handle this type of issue, I hardcode some codes to check the inputs and
   in most of the cases the program will keep asking the user till an acceptable
   one is given.
   
2. There is no undo or redo here, as much as it sounds unreasonable in an image
   processing program.

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
def input_check(input, function, error = "Error: invalid input!"):
   is_valid = False
   try:
      temp = function(input)
      is_valid = True
   except:
      print(error)
   return is_valid
   
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
      if input_check(response, int, "Please provide an integer!"):
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

# Takes a string windwo_name, a graph image, and a list of nodes, creates a
# window with the given name, displays the graph, and labels each node with a
# red rectangle and a number indicating it's index in the list.
def display_nodes(window_name, graph_display, elements, tW, tH):
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
      cv2.putText(graph_display, str(i + BASE), position, 
         cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS, 
         cv2.LINE_AA)
      cv2.rectangle(graph_display, elements[i], (elements[i][0] + tW, 
         elements[i][1] + tH), RECT_COLOR, 1)
      cv2.imshow(window_name, graph_display)
      cv2.waitKey(1)
      
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

#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #
'''
Even though these functions are not likely to be used multiple times, they are
organized in this way for better readability.
'''

# Asks the user to provide the names of the images, returns the opencv images
# of the graph and the template.      
def get_images(show_graph = False, show_template = False):
   graph = None
   graph_gray = None
   template = None
   template_gray = None
   graph, graph_gray = get_image(GRAPH_PATH, "graph")
   template, template_gray = get_image(TEMPLATE_PATH, "vertex example")
   if show_graph:
      cv2.imshow("graph", graph)
      cv2.waitKey(1)
   if show_template:
      cv2.imshow("template", template)
      cv2.waitKey(1)
   break_point = get_threshold(graph_gray)
   return graph, graph_gray, template_gray, break_point

# Processes the template in order to retrieve helpful information.
def process_template(template, break_point):
   template = cv2.Canny(template, 50, 200)
   (tH, tW) = template.shape[:2]
   radius = sqrt(pow((1 + 2 * SCALE_PERCENTAGE) * tH, 2) + pow((1 + 
      2 * SCALE_PERCENTAGE) * tW, 2)) / 2
   return template, (tH, tW), radius

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
   user_input = 1
   nodes = []
   nodes_center = []
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
            
      # perform multi-scale template matching, contributed by Adrian Rosebrock
      # from PyImageResearch
      for i in range(user_input):
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
       
         cv2.rectangle(graph_display, (startX, startY), (endX, endY), \
            RECT_COLOR, 2)
         cv2.imshow("Vertices", graph_display)
         cv2.waitKey(1)
         # graph_work is not intended to be displayed
         cv2.rectangle(graph_work, (startX, startY), (endX, endY), \
            (255, 255, 225), cv2.FILLED)
         nodes.append((int(maxLoc[0] * r), int(maxLoc[1] * r)))
      print("Current vertices:")
      print_list(nodes)
   user_input = ''
   while not user_input == DONE:
      graph_display3 = graph_display2.copy()
      display_nodes("Vertices with Labels", graph_display3, nodes, tW, tH)
      valid = False
      while valid == False:
         while user_input == '':
            user_input = input("Indicate non-vertex element in the list in a" +
               " sequence of indices or \"done\" to proceed to next step:\n")
         indices = user_input.split()
         valid = True
         if user_input != DONE:
            for i in indices:
               if not input_check(i, int, "Invalid input detected!"):
                  valid = False
                  user_input = ''
               elif int(i) < BASE or int(i) >= BASE + len(nodes):
                  print("Error: index out of bound!\n")
                  valid = False
                  user_input = ''
         
      if not user_input == DONE:
         index_remove = user_input.split()
         for i in index_remove:
            nodes[int(i) - BASE] = PLACE_HOLDER
         nodes = [element for element in nodes if element != PLACE_HOLDER]
         for node in nodes:
            nodes_center.append(node)
         user_input = ''
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
            if input_check(response, int, "The input is not an integer."):
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
                  if input_check(index, int, "Please provide a valid integer."):
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



# Extracts all the contours in the image.
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
      cv2.THRESH_BINARY_INV)
   if thin:
      graph_gray_bin = thinning(graph_gray_bin)
   ret, result = cv2.threshold(graph_gray_bin, 0, 255, cv2.THRESH_BINARY)
   
   # Extracts contours.
   print("Extracting contours....")
   contours_display, contours, hierarchy = cv2.findContours(result, \
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(contours_display, contours, -1, (0, 255, 0), 3)
   print("Number of contours detected: " + str(len(contours)))
   return contours

# From the contours extracts all edges. For each contour, select the first and
# the middle element in the list to be the end points of an edge segment, this
# is because whatever stored in a "contour" is essentially all the pixels that
# surrounds an edge and luckily the starting element seems to be very close to
# one of the end points. Then for all the vertices examines the distance from
# their center to the end points, choosees the vertex whose distance is within
# the tolerance and is the smallest compared with the rest.
def extract_edges(contours, nodes_center, radius, graph):
   print("Retrieving graph data....")
   E = []
   minimum = []
   edge_pos = []
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
            end1 = i
      d_min2 = inf
      for i in range(len(nodes_center)):
         d_temp = get_distance(end2_pos, nodes_center[i])
         if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min2:
            d_min2 = d_temp
            end2 = i
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
      
   # ask the user to remove redundancy
   user_input = ''
   while not user_input == DONE:
      print("Number of edges detected: " + str(len(E)))
      contour_center = []
      for e in E:
         contour = edge_to_contour[e]
         contour_center.append(contour[int(len(contour) / 4)])
      edges_display = graph.copy()
      for i in range(len(contour_center)):
         cv2.putText(edges_display, str(i + BASE), (contour_center[i][0][0], \
            contour_center[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, \
            FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
         cv2.imshow("Edges with Labels", edges_display)
         cv2.waitKey(1)
      valid = False
      while valid == False:
         while user_input == '':
            user_input = input("Indicate non-edge element in the list in a " +
               "sequence of indices or \"done\" to proceed to next step:\n")
         indices = user_input.split()
         valid = True
         if user_input != DONE:
            for i in indices:
               if not input_check(i, int, "Invalid input detected!"):
                  valid = False
                  user_input = ''
               elif int(i) < BASE or int(i) >= BASE + len(nodes):
                  print("Error: index out of bound!\n")
                  valid = False
                  user_input = ''
      if user_input != DONE:
         index_remove = user_input.split()
         removing = []
         for i in index_remove:
            try:
               removing.append(int(i) - BASE)
            except:
               print("Invalid input, please try again!")
         E = [E[i] for i in range(len(E)) if not i in removing]
         user_input = ''
      
   #print("Goal distance = " + str(radius * TOLERANCE_FACTOR))
   #print("Minimum:")
   #print_list(minimum)
   print("Number of edges detected: " + str(len(E)))
   print("Edges:")
   print(E)
   return E

#                               End of Section                                #
###############################################################################
###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":
   # Obtain the files.
   graph, graph_gray, template, break_point = get_images()
   
   # Process the template.
   template, (tH, tW), radius = process_template(template, break_point)
   
   # Find all the vertices. In particular variable nodes stores a list of
   # nodes' upper-right corner.
   nodes, nodes_center = find_vertices(graph.copy(), graph_gray.copy(), \
      template, tW, tH)
   
   sort_vertices(nodes)
   
   contours = extract_contours(graph_gray, nodes, tW, tH, break_point)
   
   E = extract_edges(contours, nodes_center, radius, graph)

   halt = input("HALT (press enter to end)")
   