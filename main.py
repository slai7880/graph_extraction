'''
main.py
This program helps the user to extract a mathematical graph from a
digital image and stores the data in some structure.

The procedure consists of a few steps. I present part of them here
since these are related to user input.

1. Initialization
Before executing this program, the user is allowed to modify the file common
where a few constants are defined.

At first the program asks the user to provide the path of the image
as well as that of an example of the vertices. The user must crop the
example from the original image with some tool before using this program.
If the path cannot be understood or the file does not exist, the program
will keep asking the user to provide a correct one.

2. Locating Vertices
Next the program will ask the user to provide the number of vertices
they want from the image. This number may not be accurate, and the
request will be repeated untill the user enter 0, implying that there
is no vertex left unfound. Due to the accuracy restriction of template
matching which is the main technique being used here, there is no gurantee
that all the detected vertices are true ones. However we still ask the user
to keep staying at this step till all the vertices are marked.

After the user enter 0, the program will label all the vertices on the image,
and asks the user to give a sequence of indices of false vertices. The input
must be a sequence of valid integers separated by space, however a single integer
is allowed as well.

Next the user will be asked if they want to correct the order of the vertices.
The user may answer yes if they want the indices to match the orginal labels on
the image. There are two ways to correct, the option can be chosen in the common
file.

3. Detecting Edges
The program now should start to process the image. When it completes, it will again
display all the detected edges with labels and ask the user to provide a sequence of
indices indicating the false edges.

At the end a list of edges will be printed to the console.

'''


import sys
import numpy as np
import imutils
import cv2
from common import *
from thinning import *
from math import sqrt, inf, fabs

# Prints a list of elements one per line with their indices.
def print_list(list):
   for i in range(len(list)):
      print(str(i) + ". " + str(list[i]))

# Given an input argument, a testing function and an error message, tests
# the argument with the function. This is most likely used to test if the
# user input is valid.
def input_check(input, function, error):
   is_valid = False
   try:
      temp = function(input)
      is_valid = True
   except:
      print(error)
   return is_valid

# Takes a string windwo_name, a graph image, and a list of nodes, creates
# a window with the given name, displays the graph, and labels each node
# with a red rectangle and a number indicating it's index in the list.
def display_nodes(window_name, graph_display, elements):
   for i in range(len(elements)):
      position = (elements[i][0] + int(tW / 3), elements[i][1] + int(tH * 2 / 3))
      if not isinstance(REL_POS, str):
         position = (elements[i][0] + REL_POS[0], elements[i][1] + REL_POS[1])
      cv2.putText(graph_display, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.rectangle(graph_display, elements[i], (elements[i][0] + tW, elements[i][1] + tH), (0, 0, 225), 1)
      cv2.imshow(window_name, graph_display)
      cv2.waitKey(1)

# Takes three parameters, a graph_display that is used to present to the
# user, a graph_work that is used to perform algorithms, and a template
# that stores an example of a node. This function first asks the user to
# give an approximate amount, say n, of vertices in a graph, and then it
# will find the first n pieces in the graph image that match the template.
# This step will keep going untill the user enter a 0 as input. Next it will
# label all the found verteices and ask the user to point out which one(s)
# may be false, the user must either give a sequence of ingeters indicating
# the indices of the vertices that they do not want, or type "done" to proceed
# to the next step.
def find_vertices(graph_display, graph_work, template):
   graph_display2 = graph_display.copy() # will be used in the removing part
   user_input = 1
   nodes = []
   while user_input > 0:
      user_input = input("How many vertices are you looking for?(0 means done) ")
      try:
         user_input = int(user_input)
      except:
         user_input = -1
         print("\nCannot recognize the input, please provide a number.")
      while user_input < 0:
         user_input = input("How many vertices are you looking for?(0 means done) ")
         try:
            user_input = int(user_input)
         except:
            print("\nCannot recognize the input, please provide a number.")
            
      # perform multi-scale template matching, contributed by Adrian Rosebrock from PyImageResearch
      for i in range(user_input):
         found = None
         for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(graph_work, width = int(graph_work.shape[1] * scale))
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
       
         cv2.rectangle(graph_display, (startX, startY), (endX, endY), (0, 0, 225), 2)
         cv2.imshow("Vertices", graph_display)
         cv2.waitKey(1)
         cv2.rectangle(graph_work, (startX, startY), (endX, endY), (255, 255, 225), cv2.FILLED)
         nodes.append((int(maxLoc[0] * r), int(maxLoc[1] * r)))
      print("Current vertices:")
      print_list(nodes)
   user_input = ''
   while not user_input == DONE:
      graph_display3 = graph_display2.copy()
      display_nodes("Vertices with Labels", graph_display3, nodes)
      while user_input == '':
         user_input = input("Indicate non-vertex element in the list in a sequence of index:\n")
      if user_input != DONE:
         index_remove = user_input.split()
         for i in index_remove:
            nodes[int(i)] = PLACE_HOLDER
         nodes = [element for element in nodes if element != PLACE_HOLDER]
         user_input = ''
   print("Current vertices:")
   print_list(nodes)
   return nodes

# Takes a list of vertices as the parameter, allows the user to select
# their prefered method to correct the index of each vertex.
def sort_vertices(nodes):
   result = [(0, 0)] * len(nodes)
   
   if SORT == 1:
      index_list = []
      for i in range(len(nodes)):
         valid = False
         while valid == False:
            index = input("What's the correct index value of the vertex " + str(i) + ". " + str(nodes[i]) + "? ")
            if input_check(index, int, "Please provide a valid integer."):
               index = int(index)
               if index < INDEX_START or index >= INDEX_START + len(nodes):
                  print("Index out of bound!")
               elif index in index_list:
                  print("Duplicate index detected, please provide another one.")
               else:
                  valid = True
                  index_list.append(index)
         result[index] = nodes[i]
   elif SORT == 2:
      valid = False
      while valid == False:
         index_list = []
         indices = input("Please provide a sequence of correct indices for each vertex:\n")
         try:
            indices = indices.split()
            for i in range(len(indices)):
               if int(indices[i]) in index_list:
                  print("Duplicate index detected, please provide another one.")
                  break
               else:
                  index_list.append(int(indices[i]))
            if len(index_list) == len(nodes):
               valid = True
            else:
               print("Not enough inputs, please try again.")
         except:
            print("Please provide a sequence of valid integers.")
   else:
      print("Cannot sort the vertices, check the method indicating value.")
      sys.exit(1)
      
   for i in range(len(index_list)):
      result[index_list[i]] = nodes[i]
   return result

# Computes and returns the Euclidean distance between two points.
def get_distance(p1, p2):
   return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

# Computes the slope of the line marked by two given points.
def get_slope(p1, p2):
   if p1[0] - p2[0] == 0:
      return inf
   return (p1[1] - p2[1]) / (p1[0] - p2[0])

if __name__ == "__main__":
   # Obtain the files.
   graph_filename = ''
   while graph_filename == '':
      graph_filename = input("Please provide the filename of the graph: ")
      try:
         graph = cv2.imread(GRAPH_PATH + graph_filename)
         graph_gray = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)
      except:
         print("Error: the graph file is not found.")
         graph_filename = ''
   template_filename = ''
   while template_filename == '':
      template_filename = input("Please provide the file name of the node example: ")
      try:
         template = cv2.imread(TEMPLATE_PATH + template_filename)
         template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
      except:
         print("Error: the node example file is not found.")
         template_filename = ''
   
   # Preprocess the files.
   
   
   template = cv2.Canny(template, 50, 200)
   (tH, tW) = template.shape[:2]
   radius = sqrt(pow((1 + 2 * SCALE_PERCENTAGE) * tH, 2) + pow((1 + 2 * SCALE_PERCENTAGE) * tW, 2)) / 2
   cv2.imshow("graph", graph)
   cv2.waitKey(1)
   
   # Find all the vertices.
   nodes = find_vertices(graph.copy(), graph_gray.copy(), template)
   
   # Ask if the user wants to sort the vertices.
   answer = ''
   while answer == '':
      answer = input("Do you want to sort the vertices? (y/n): ")
      if answer[0] == 'y' or answer[0] == 'Y':
         nodes = sort_vertices(nodes)
         print("Updated list of vertices:")
         print_list(nodes)
      elif answer[0] == 'n' or answer[0] == 'N':
         break
      else:
         answer = ''
         print("Please answer with y/n.")
   #cv2.destroyAllWindows()
   
   
   
   print("Processing the graph, this step may take some time, please wait....")
   
   # Hides all the nodes on a draft version of the graph image.
   node_center = []
   graph_gray_bin = graph_gray.copy()
   for i in range(len(nodes)):
      # The start corner and the end corner, both in the form (x, y).
      upper_left = (nodes[i][0] - int(tW * SCALE_PERCENTAGE), nodes[i][1] - int(tH * SCALE_PERCENTAGE))
      bottom_right = (nodes[i][0] + int(tW * (1 + SCALE_PERCENTAGE)), nodes[i][1] + int(tH * (1 + SCALE_PERCENTAGE)))
      
      # Records the center of each node.
      node_center.append((int(nodes[i][0] + tW / 2), int(nodes[i][1] + tH / 2)))
      
      # Places a block at each node.
      cv2.rectangle(graph_gray_bin, upper_left, bottom_right, (255, 255, 225), cv2.FILLED)
      
   # Performs image thinning.
   ret, graph_gray_bin = cv2.threshold(graph_gray_bin, 127, 1, cv2.THRESH_BINARY_INV)
   graph_gray_bin = thinning(graph_gray_bin)
   ret, result = cv2.threshold(graph_gray_bin, 0, 255, cv2.THRESH_BINARY)
   
   # Extracts contours.
   print("Extracting edges....")
   contours_display, contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   # Asks the user to remove false edges.
   print("Number of edges detected: " + str(len(contours)))
   print(contours[0].shape)
   user_input = ''
   while not user_input == DONE:
      contour_center = []
      for c in contours:
         contour_center.append(c[int(len(c) / 4)])
      edges_display = graph.copy()
      for i in range(len(contour_center)):
         cv2.putText(edges_display, str(i), (contour_center[i][0][0], contour_center[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
         cv2.imshow("Edges with Labels", edges_display)
         cv2.waitKey(1)
      while user_input == '':
         user_input = input("Indicate non-edge element in the list in a sequence of index:\n")
      if user_input != DONE:
         index_remove = user_input.split()
         removing = []
         for i in index_remove:
            removing.append(int(i))
         contours = [contours[i] for i in range(len(contours)) if not i in removing]
         user_input = ''
   
   
   # From the contours extracts all edges. For each contour, select the
   # first and the middle element to be the end points of an edge segment,
   # then for all the vertices examines the distance from their center to
   # the end points, choosees the vertex whose distance is within the tolerance
   # and is the smallest compared with the rest.
   print("Retrieving graph data....")
   E = []
   minimum = []
   edge_pos = []
   edge_to_contour = []
   for i in range(len(contours)):
      edge = contours[i]
      end1 = -1
      end2 = -1
      end1_pos = (edge[0][0][0], edge[0][0][1])
      end2_pos = (edge[int(len(edge) / 2)][0][0], edge[int(len(edge) / 2)][0][1])
      d_min1 = inf
      for i in range(len(node_center)):
         d_temp = get_distance(end1_pos, node_center[i])
         if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min1:
            d_min1 = d_temp
            end1 = i
      d_min2 = inf
      for i in range(len(node_center)):
         d_temp = get_distance(end2_pos, node_center[i])
         if d_temp < radius * TOLERANCE_FACTOR and d_temp < d_min2:
            d_min2 = d_temp
            end2 = i
      E.append((end1, end2))
      edge_to_contour.append(i)
      minimum.append((d_min1, d_min2))
      edge_pos.append([end1_pos, end2_pos])
   print("Edges:")
   print(E)

   temp = input("Halt")
   