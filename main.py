'''
main.py
Sha Lai
9/17/2016

This program interacts with the user in console to use the graph_extraction
program to retrieve a mathematical graph from a digital image.


'''

from graph_extraction import *
from common import *

###############################################################################
#                              User Interaction                               #

def get_image(dir_path, keyword):
   """Interacts with the user to read an image from a provided directory.
   Parameters
   ----------
   dir_path : string
      The path of the directory which the program should be looking into.
   keyword : string
      Specifies the type of content(graph vs template). For printing only.
   Returns
   -------
   image : numpy matrix of integers
      The original image.
   image_gray : numpy matrix of integers
      The grayscale version of the image.
   """
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
   """Interacts with the user to read the image of the graph as well as that of
   the template used to locate the vertices.
   Parameters
   ----------
   show_graph and show_template : boolean
      When set to True, the original graph image or the template will be shown
      to the user.
   Returns
   -------
   graph and graph_gray : numpy matrix of integers
      The original graph image and it's grayscale version.
   template_gray : numpy matrix of integers
      The grayscale version of the template.
   break_point : int
      The value that is used to distinguish the background and the content in
      the original graph image.
   """
   graph = None
   graph_gray = None
   template = None
   template_gray = None

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

# Given a string of user input, a prompt sentence, keep asking the user to
# provide a list of indices until a valid one(can be DONE) is entered.
def get_valid_list(user_input, prompt_sentence, list_length):
   """Keep asking the user to provide a list of indices until a valid one(can
      be DONE) is entered.
   Parameters
   ----------
   user_input : string
      The input from the user.
   promt_sentence : string
      A string that is used to ask the user.
   list_length : int
      The length of the list that is being processed.
   """
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

def hide_vertices(image, nodes_center, radius, color = 0):
   """Puts a circle filled with the background color on each vertex.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is being studied.
   nodes_center : list of 
   """
   for n in nodes_center:
      cv2.circle(image, n, int(radius), color, cv2.FILLED)



# Takes a string indicating the shape of the kernel with its first letter, returns
# the corresponding cv2 constant.
def get_kernel_shape(shape):
   if shape[0] == 'r' or shape[0] == 'R':
      return cv2.MORPH_RECT
   elif shape[0] == 'e' or shape[0] == 'E':
      return cv2.MORPH_ELLIPSE
   elif shape[0] == 'c' or shape[0] == 'C':
      return cv2.MORPH_CROSS
   else:
      print("Invalid shape string.")
      return None


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
            user_input = -1
            print("\nCannot recognize the input, please provide a number.")
            
      locate_vertices(user_input, graph_work, template, tW, tH, nodes)
      draw_vertices(graph_display, nodes, tW, tH, False)
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
      draw_vertices(graph_display3, nodes, tW, tH)
      cv2.startWindowThread()
      cv2.imshow("Vertices with Labels", graph_display3)
      cv2.waitKey(1)
      user_input = ''
      user_input = get_valid_list(user_input, "Indicate non-vertex elements " +
                                    "in the list in a sequence of indices " +
                                    "or \"done\" to proceed to next step:\n",\
                                    len(nodes))
      if user_input != DONE:
         nodes = remove_nodes(user_input, nodes)
         print("Current vertices:")
         print_list(nodes)
         user_input = ''
   
   print("Current vertices:")
   print_list(nodes)
   return nodes

# Takes a list of vertices as the parameter, allows the user to select
# their prefered method to correct the index of each vertex. When entering
# the base, the input must be either 0 or 1.
# Note: the second of sorting is not safe for large size graph since if
# the user's input, when sorted in order, is not an arithmetic sequence,
# then the program will be broken.
def sort_vertices(nodes, graph_display):
   answer = ''
   while answer == '':
      answer = input("Do you want to sort the vertices? (y/n): ")
      if len(answer) > 0:
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
                     if not len(indices) == len(nodes):
                        print("Not enough integers or too many of them, please " + 
                           "try again.")
                     else:
                        for i in range(len(indices)):
                           if int(indices[i]) in index_list:
                              print("Duplicate index detected, please provide " + 
                                 "another one.")
                              break
                           else:
                              index_list.append(int(indices[i]))
                        if len(index_list) == len(nodes):
                           if not max(index_list) + 1 - BASE == len(index_list):
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
            draw_vertices(graph_display, nodes, tW, tH)
            cv2.startWindowThread()
            cv2.destroyWindow("Vertices with Labels")
            cv2.imshow("Vertices with Labels", graph_display)
            cv2.waitKey(1)
            print("Updated list of vertices:")
            print_list(nodes)
         elif answer[0] == 'n' or answer[0] == 'N':
            break
         else:
            answer = ''
            print("Please answer with y/n.")
      else:
         answer = ''
         print("Please answer with y/n.")
   return nodes

# Takes a gray scale image, a breakpoing value, a list of coordinates of the
# center of the vertices, and a radius value, interacts with the user to
# process the image so that noise can be reduced.
def noise_reduction(image_gray, break_point, nodes_center, radius):
   image_bin_inv = get_binary_image_inv(image_gray, break_point)
   show_binary_image(image_bin_inv, "Noise Reduction")
   image_stack = []
   message_stack = []
   image_stack.append(image_bin_inv)
   message_stack.append('')
   kernel_shape_str = KERNEL_SHAPE
   kernel_shape = get_kernel_shape(kernel_shape_str)
   kernel_size = KERNEL_SIZE
   kernel = cv2.getStructuringElement(kernel_shape,kernel_size)
   response = ''
   last_step = ''
   result = image_bin_inv
   while len(response) == 0:
      print("Current kernel shape and size is: " + kernel_shape_str + str(kernel_size))
      print("Pelease indicate which operation to perform or adjust the kernel size:")
      print("(a)djust kernel(this cannot be undone)")
      print("(c)over vertices")
      print("(d)ilation")
      print("(e)rosion")
      print("(p)roceed")
      print("(t)hin")
      print("(u)ndo" + message_stack[-1])
      response = input("Your choice is: ")
      if len(response) > 0:
         if response[0] == 'a':
            new_shape_str = ''
            while new_shape_str == '':
               print("Please provide the desired shape of the kernel:")
               print("(C)ross")
               print("(E)llipse")
               print("(R)ectangle")
               new_shape_str = input("Your choice is: ")
               if len(new_shape_str) > 0:
                  if new_shape_str[0] in ['r', 'R', 'e', 'E', 'c', 'C']:
                     kernel_shape = get_kernel_shape(new_shape_str)
                     kernel_shape_str = KERNEL_STR_MAP[new_shape_str[0]]
                  else:
                     print("Please provide a valid string!")
                     new_shape_str = ''
            new_size = ''
            while new_size == '':
               new_size = input("Enter the new kernel size as a single integer: ")
               if is_valid_type(new_size, int):
                  new_size = int(new_size)
                  if new_size > min(image_gray.shape[0], image_gray.shape[1]) or new_size <= 0:
                     print("Please provide a valid integer!")
                     new_size = ''
                  else:
                     kernel_size = (new_size, new_size)
                     print("Kernel size is now " + str(kernel_size))
               else:
                  print("Please provide a valid integer!")
                  new_size = ''
            response = ''
            kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            print("Updated kernel:")
            print(kernel)
         elif response[0] in ['c', 'd', 'e', 't']:
            image_temp = image_stack[-1].copy()
            if response[0] == 'c':
               print("Covering vertices....")
               hide_vertices(image_temp, nodes_center, radius)
               last_step = " (last step was (c)over vertices)"
            elif response[0] == 'd':
               print("Performing dilation....")
               image_temp = denoise(image_temp, cv2.dilate, kernel, 1)
               last_step = " (last step was (d)ilation)"
            elif response[0] == 'e':
               print("Performing erosion....")
               image_temp = denoise(image_temp, cv2.erode, kernel, 1)
               last_step = " (last step was (e)rosion)"
            elif response[0] == 't':
               print("Performing image thinning....")
               image_temp = thin(image_temp)
               last_step = " (last step was (t)hin)"
            image_stack.append(image_temp)
            message_stack.append(last_step)
            show_binary_image(image_temp, "Noise Reduction")
            print("Operation complete.")
            response = ''
         elif response[0] == 'u':
            if len(image_stack) > 1:
               image_stack.pop()
               message_stack.pop()
               image_temp = image_stack[-1]
               show_binary_image(image_temp, "Noise Reduction")
            else:
               print("There is no last step.")
            response = ''
         elif response[0] == 'p':
            result = image_stack[-1]
         else:
            print("Invalid input, please try again!")
            response = ''
   return result
            

# From the contours extracts all edges. For each contour, select the first and
# the middle element in the list to be the end points of an edge segment, this
# is because whatever stored in a "contour" is essentially all the pixels that
# surrounds an edge and luckily the starting element seems to be very close to
# one of the end points. Then for all the vertices examines the distance from
# their center to the end points, choosees the vertex whose distance is within
# the tolerance and is the smallest compared with the rest.
def extract_edges(graph_work, nodes_center, radius):
   E = []
   endpoints = get_endpoints(graph_work, nodes_center, radius)
   trails = []

   for i in range(len(endpoints)):
      for j in range(len(endpoints[i])):
         edge, trail = get_edge(graph_work, endpoints[i][j], [], [], nodes_center, i, radius)
         if edge != PLACE_HOLDER and not edge in E and not (edge[1], edge[0]) in E:
            E.append(edge)
            trails.append(trail)
   image_temp = np.zeros(graph_work.shape, np.uint8)
   
   
   '''
   edge, trail = get_edge(graph_work, endpoints[0][0], [], [], nodes_center, 0, endpoints, radius)
   print(edge)
   print(len(trail))
   
   
   image_temp = np.zeros(graph_work.shape, np.uint8)
   for p in trail:
      image_temp[p[1], p[0]] = 1
   show_binary_image(image_temp, "trails", True)
   '''
   
   return E, endpoints

def display_edges(E):
   print("Displaying edges....")
   print_list(E)

#                               End of Section                                #
###############################################################################
###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":
   # Obtain the files.
   graph, graph_gray, template, break_point = get_images(True)
   
   # Process the template.
   template, (tH, tW), radius = process_template(template)
   
   # Find all the vertices. In particular variable nodes stores a list of
   # nodes' upper-right corner.
   nodes = find_vertices(graph.copy(), graph_gray.copy(), \
      template, tW, tH)
   
   # If neccesary, sort the vertices such that the order matches the given one.
   nodes = sort_vertices(nodes, graph.copy())
   
   nodes_center = get_center_pos(nodes, tW, tH)
   
   complete = False
   while complete == False:
      graph_work = noise_reduction(graph_gray, break_point, nodes_center, radius)
      E, endpoints = extract_edges(graph_work, nodes_center, radius)
      display_edges(E)
      user_input = ''
      while len(user_input) == 0:
         user_input = input("Is it necessary to attempt again?(y/n)")
         if len(user_input) > 0:
            if user_input[0] == 'n' or user_input[0] == 'N':
               complete = True
            elif user_input[0] != 'y' and user_input[0] != 'Y':
               print("Invalud input, please try again!")
               user_input = ''
   '''
   # Gets the edges of the graph.
   E, endpoints = extract_edges(nodes, nodes_center, radius, graph, graph_gray, tW, tH, \
                     break_point, 1, 1, 1)

   print("Number of edges detected: " + str(len(E)))
   print("Edges:")
   print(E)
   print_list(endpoints)
   '''
   
   halt = input("HALT (press enter to end)")