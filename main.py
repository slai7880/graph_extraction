'''
main.py
Sha Lai
8/26/2016

This program interacts with the user in console to use the graph_extraction
program to retrieve a mathematical graph from a digital image.


'''

from graph_extraction import *
from common import *

###############################################################################
#                              User Interaction                               #

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
   E, minimum, edge_pos, edge_to_contour = get_edges(contours, nodes_center, radius)
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