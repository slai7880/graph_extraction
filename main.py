'''
main.py
Sha Lai
9/17/2016

This program interacts with the user in console to use the graph_extraction
program to retrieve a mathematical graph from a digital image.

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

from graph_extraction import *
from common import *

###############################################################################
#                               Helper Functions                              #
def get_kernel_shape(shape_str):
   """Returns the kernel shape constant in OpenCV package.
   Parameters
   ----------
   shape_str : string
      A string starting with r/R(ectangle), e/E(llipse), or c/C(ross).
   Returns
   -------
   OpenCV shape indicator or None if the parameter is invalid.
   """
   if shape_str[0] == 'r' or shape_str[0] == 'R':
      return cv2.MORPH_RECT
   elif shape_str[0] == 'e' or shape_str[0] == 'E':
      return cv2.MORPH_ELLIPSE
   elif shape_str[0] == 'c' or shape_str[0] == 'C':
      return cv2.MORPH_CROSS
   else:
      print("Invalid shape string.")
      return None







####################### GUI Event Handler Subsection ##########################
# crop an example of the vertices
def crop(event, x, y, flags, image):
   global ix, iy, drawing, cont, image_c, template_num, image_init
   if event == cv2.EVENT_LBUTTONDOWN:
      ix, iy = x, y
      drawing = True
      image_init = image.copy()
   elif event == cv2.EVENT_MOUSEMOVE:
      if drawing == True:
         image_c = image_init.copy()
         cv2.rectangle(image_c, (ix, iy), (x, y), RECT_COLOR, RECT_THICKNESS)
   elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      cv2.rectangle(image_c, (ix, iy), (x, y), RECT_COLOR, RECT_THICKNESS)
      tW = abs(ix - x)
      tH = abs(iy - y)
      template_num = ((min(iy, y), min(iy, y) + tH), (min(ix, x), min(ix, x) + tW))
   elif event == cv2.EVENT_RBUTTONUP:
      image_c = image_init.copy()
      template_num = None

# remove false vertices
def remove_vertices(event, x, y, flags, image):
   global image_c, nodes, nodes_center, image_original, removed_stack,\
            image_stack, tW, tH, ref_pos, rel_pos, font_size, font_thickness
   if event == cv2.EVENT_LBUTTONDOWN:
      if len(nodes) > 0:
         indices = []
         for i in range(len(nodes)):
            center = nodes_center[i]
            if x >= center[0] - tW / 2 and x <= center[0] + tW / 2 and\
               y >= center[1] - tH / 2 and y <= center[1] + tH / 2:
               indices.append(i)
         for i in indices:
            removed_stack.append((nodes[i], nodes_center[i], ref_pos[i], i))
         temp1 = []
         temp2 = []
         temp3 = []
         for i in range(len(nodes)):
            if not i in indices:
               temp1.append(nodes[i])
               temp2.append(nodes_center[i])
               temp3.append(ref_pos[i])
         nodes = temp1
         nodes_center = temp2
         ref_pos = temp3
         '''
         print("(x, y) = " + str((x, y)))
         print("Removing e = " + str(result) + "  dist_min = " + str(dist_min))
         print("temp = " + str(temp))
         '''
         image_stack.append(image_c.copy())
         image_c = image_original.copy()
         highlight_vertices(image_c, nodes, tW, tH)
         label_vertices(image_c, ref_pos, rel_pos, font_size, font_thickness)
   elif event == cv2.EVENT_RBUTTONDOWN:
      if len(removed_stack) > 0:
         top = removed_stack.pop()
         i = top[3]
         # nodes.append(top[0])
         nodes = nodes[: i] + top[0] + nodes[i :]
         # nodes_center.append(top[1])
         nodes_center = nodes_center[: i] + top[1] + nodes_center[i :]
         # ref_pos.append(top[2])
         ref_pos = ref_pos[: i] + top[2] + ref_pos[i :]
         image_c = image_stack.pop()

# locate the remaining vertices   
def select(event, x, y, flags, image):
   global cont, image_c, nodes, tW, tH
   if event == cv2.EVENT_MOUSEMOVE:
      image_c = image.copy()
      x_start, y_start = x - int(tW / 2), y - int(tH / 2)
      x_end, y_end = x - int(tW / 2) + tW, y - int(tH / 2) + tH
      cv2.rectangle(image_c, (x_start, y_start), (x_end, y_end), RECT_COLOR,\
                     RECT_THICKNESS)
   elif event == cv2.EVENT_LBUTTONDOWN:
      cv2.rectangle(image, (x - int(tW / 2), y - int(tH / 2)),
         (x - int(tW / 2) + tW, y - int(tH / 2) + tH), RECT_COLOR, RECT_THICKNESS)
      nodes.append((x - int(tW / 2), y - int(tH / 2)))
   elif event == cv2.EVENT_RBUTTONUP:
      cont = False

# remove false edges      
def remove(event, x, y, flags, image):
   global image_c, E, nodes_center, image_original, removed_stack, image_stack
   if event == cv2.EVENT_LBUTTONDOWN:
      if len(E) > 0:
         result = [-1, -1]
         dist_min = image_c.shape[0] + image_c.shape[1]
         P = [x, y]
         temp = [0, 0]
         for e in E:
            A = nodes_center[e[0]]
            B = nodes_center[e[1]]
            AP = get_vector(A, P)
            AB = get_vector(A, B)
            AB_unit = np.multiply(AB, 1.0 / two_norm(AB))
            scalar_proj = np.inner(AP, AB) / two_norm(AB)
            AB_len = get_distance(A, B)
            dist = min(get_distance(A, P), get_distance(B, P))
            if scalar_proj > 0 and AB_len > scalar_proj:
               vec_proj = np.multiply(AB_unit, scalar_proj)
               dist = two_norm(np.subtract(vec_proj, AP))
            if dist < dist_min:
               dist_min = dist
               result = e
               temp[0] = AB_len
               temp[1] = scalar_proj
         '''
         print("(x, y) = " + str((x, y)))
         print("Removing e = " + str(result) + "  dist_min = " + str(dist_min))
         print("temp = " + str(temp))
         '''
         E.remove(result)
         removed_stack.append(result)
         image_stack.append(image_c.copy())
         image_c = image_original.copy()
         for e in E:
            des1 = e[0]
            des2 = e[1]
            cv2.line(image_c, nodes_center[des1], nodes_center[des2], (0, 0, 255), 2)
   elif event == cv2.EVENT_RBUTTONDOWN:
      if len(removed_stack) > 0:
         E.append(removed_stack.pop())
         image_c = image_stack.pop()

# add undetected edges
def add(event, x, y, flags, image):
   global image_c, E, nodes_center, image_original, image_stack, start_linking, i1, i2
   if event == cv2.EVENT_LBUTTONDOWN:
      start_linking = True
      dist_min = image_original.shape[0] + image_original.shape[1]
      for i in range(len(nodes_center)):
         dist = get_distance([x, y], nodes_center[i])
         if dist < dist_min:
            dist_min = dist
            i1 = i
   elif event == cv2.EVENT_MOUSEMOVE:
      if start_linking:
         image_c = image.copy()
         cv2.line(image_c, nodes_center[i1], (x, y), (0, 0, 255), 2)
   elif event == cv2.EVENT_LBUTTONUP:
      if start_linking:
         start_linking = False
         dist_min = image_original.shape[0] + image_original.shape[1]
         for i in range(len(nodes_center)):
            dist = get_distance([x, y], nodes_center[i])
            if i != i1 and dist < dist_min:
               dist_min = dist
               i2 = i
         if [i1, i2] not in E and [i2, i1] not in E:
            cv2.line(image, nodes_center[i1], nodes_center[i2], (0, 0, 255), 2)
            image_stack.append(image_c.copy())
            image_c = image.copy()
            E.append([i1, i2])
            print(len(E))
   elif event == cv2.EVENT_RBUTTONDOWN:
      if len(image_stack) > 0:
         E.pop()
         image_c = image_stack.pop()

# threshold the image in a dynamic way
def filter(image_gray, trackbar_name, window_name):
   global graph_bin
   break_point = cv2.getTrackbarPos(TRACKBAR_THRESHOLD, window_name)
   image_bin2 = get_binary_image_inv(image_gray, break_point, 255)
   cv2.imshow(window_name, image_bin2)
   cv2.waitKey(1)
   graph_bin = image_bin2

# set the factor for best result
def set_rfactor(image_work, trackbar_name, window_name, radius):
   global nodes, nodes_real, nodes_unreal, endpoints
   factor = cv2.getTrackbarPos(TRACKBAR_RFACTOR, NODES)
   nodes_real, nodes_unreal = construct_network3(image_work, nodes_center,\
                                                   endpoints)
   nodes = merge_nodes(nodes_real, nodes_unreal, radius * (0.5 + 0.1 * factor),\
                        image_work)

############################  END OF SUBSECTION  ##############################

def shift_indices(E, base = BASE):
   """Shifts each index in an edge (v1, v2) by some value. Originally the
   indices should be 0-base, but natually people prefer 1-base when they study
   a graph.
   Parameters
   ----------
   E : List[[int, int]]
      The edge list.
   base : int
      The shift value.
   Returns
   -------
   result : List[[int, int]]
      The updated edge list.
   """
   result = []
   for e in E:
      result.append((e[0] + base, e[1] + base))
   return result

#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

def get_image(show_graph = False):
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
   
   response = ''
   valid = False
   while response == '' or valid == False:
      input_dir = listdir(GRAPH_PATH)
      print("Files in the input directory:")
      print_list(input_dir)
      response = input("Please provide the file by index of the graph: ")
      if is_valid_type(response, int, "Please provide an integer!"):
         index = int(response)
         if index >= 0 + BASE and index < len(input_dir) + BASE:
            try:
               graph = cv2.imread(GRAPH_PATH + input_dir[index - BASE], -1)
               # change all the transparent pixels to background ones
               if (not type(graph[0][0]) is np.uint8):
                  if len(graph[0][0]) == 4:
                     for i in range(graph.shape[0]):
                        for j in range(graph.shape[1]):
                           if graph[i][j][3] == 0:
                              graph[i][j][0] = 255
                              graph[i][j][1] = 255
                              graph[i][j][2] = 255
                  graph_gray = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)
               else:
                  graph_gray = graph.copy()
               valid = True
               print("Selected graph file: " + str(input_dir[index - BASE]))
            except:
               print("Error: the graph file is invalid or cannot be processed.")
               response = ''
         else:
            print("Error: index out of bound!\n")
   
   if show_graph:
      cv2.startWindowThread()
      cv2.namedWindow(GRAPH)
      cv2.imshow(GRAPH, graph)
      cv2.waitKey()
      cv2.destroyWindow(GRAPH)

   break_point = get_threshold(graph_gray)
   return graph, graph_gray, break_point, input_dir[index - BASE]
   
def load(filename):
   """Loads a graph data from a previous work.
   Parameters
   ----------
   filename : String
      The file name withough suffix, used in saving.
   Returns
   -------
   image : numpy matrix of integers
      Image data.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      Half of the length of the diagonal of the template.
   """
   image = np.load(DATA_PATH + filename + '.npy')
   file = open(DATA_PATH + filename + '.dat')
   lines = []
   for line in file:
      if len(line) > 0 and line[0] != '#':
         lines.append(str(line))
   file.close()
   nodes_center = eval(lines[0])
   radius = eval(lines[1])
   return image, nodes_center, radius

def initiate_UI(image, window_name, function, message):
   """Makes a certain window interactable for user to perform some task.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that will be interacted by the user.
   window_name : string
      The name of the window.
   function : python function
      This should be one of the functions defined in this subsection.
   message : String
      The message displayed to the user.
   Returns
   -------
   None
   """
   global cont, image_c
   cont = True
   image_c = image.copy()
   print(message)
   cv2.namedWindow(window_name)
   cv2.setMouseCallback(window_name, function, image_c)
   
   while cont:
      cv2.imshow(window_name, image_c)
      key = cv2.waitKey(1)
      if key & 0xFF == 13:
         cont = False
      
def get_graph_bin(image_gray, trackbar_name, window_name):
   """Allows the user to manually adjust the threshold to obtain a decent
   binary image.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      Image data in grayscale.
   trackbar_name: string
   window_name : string
   Returns
   -------
   graph_bin : numpy matrix of integers
      Image data in desired binary form.
   """
   global graph_bin
   graph_bin = image_gray.copy()
   cv2.imshow(window_name, get_binary_image_inv(image_gray, 0, 255))
   cv2.waitKey(1)
   graph_bin = get_binary_image_inv(image_gray.copy(), THRESHOLD_INIT, 255)
   cv2.createTrackbar(trackbar_name, window_name, THRESHOLD_INIT, THRESHOLD_MAX,\
                     lambda x: filter(image_gray, trackbar_name, window_name))
   print("Slide for a desired threshold value, and press Return to proceed.")
   while (1):
      k = cv2.waitKey(1) & 0xFF
      if k == 13:
         break
   cv2.destroyWindow(window_name)
   return graph_bin

   

def extract_vertices(image_display, image_work, template, tW, tH):
   """Repeatedly asks the user for the amount of undetected vertices in the
   graph until all are marked. Then the false ones can be removed by user. If
   there are some vertices cannot be detected automatically, the user is
   allowed to mark them manually.
   Parameters
   ----------
   image_display : numpy matrix of integers
      The image that is intended to be displayed to the user.
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   template : numpy matrix of integers
      A piece of the original image containing an example of the vertices.
   tW and tH : int
      The dimension values of the template.
   Returns
   -------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices.
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   """
   global nodes, nodes_center, image_original, removed_stack, image_stack,\
            ref_pos, rel_pos, font_size, font_thickness
   image_display2 = image_display.copy()
   nodes = [] # stores the upper-left coordinates of the vertices
   nodes_center = [] # stores the center coordinates of the vertices
   rel_pos = (int(tW / 3), int(tH * 2 / 3))
   user_input = 1
   while user_input > 0:
      user_input = input("How many vertices are you looking for?(0 means " + 
         "done)")
      if len(user_input) > 0:
         if is_valid_type(user_input, int, "Cannot recognize the input, " +
                           "please provide a number."):
            user_input = int(user_input)
            if user_input >= 0:
               locate_vertices(user_input, image_work, template, tW, tH, nodes)
               highlight_vertices(image_display2, nodes, tW, tH)
               image_display3 = image_display2.copy()
               label_vertices(image_display3, nodes, rel_pos, FONTSIZE_INIT * FONTSIZE_BASE, THICKNESS_INIT * THICKNESS_BASE)
               cv2.startWindowThread()
               cv2.namedWindow(VERTICES)
               cv2.imshow(VERTICES, image_display3)
               cv2.waitKey(1)
               print("Current vertices:")
               print_list(nodes)
            elif user_input < 0:
               print("Please provide a non-negative integer.")
               user_input = PLACE_HOLDER_INT
         else:
            user_input = 1
      else:
         user_input = 1
      
   cv2.destroyWindow(GRAPH)
   cv2.destroyWindow(VERTICES)

   cv2.startWindowThread()
   cv2.namedWindow(VERTICES_W_LBL)
   cv2.imshow(VERTICES_W_LBL, image_display3)
   cv2.waitKey(1)
   
   # allows the user to adjust the labels and remove vertices
   nodes_center = get_center_pos(nodes, tW, tH)
   image_original = image_display.copy()
   removed_stack = []
   image_stack = []
   ref_pos = nodes
   font_size = FONTSIZE_BASE * FONTSIZE_INIT
   font_thickness = THICKNESS_BASE * THICKNESS_INIT
   print("Slide for a desired label outcome, click on a vertex to remove," +\
         " and press Return to proceed.")
   image_display_c = image_display.copy()
   highlight_vertices(image_display_c, nodes, tW, tH)
   label_vertices(image_display_c, ref_pos, rel_pos, font_size, font_thickness)
   cv2.imshow(VERTICES_W_LBL, image_display_c)
   cv2.waitKey(1)
   cv2.createTrackbar(TRACKBAR_FONTSIZE, VERTICES_W_LBL, FONTSIZE_INIT,\
                        FONTSIZE_MAX, lambda x: x)
   cv2.createTrackbar(TRACKBAR_THICKNESS, VERTICES_W_LBL, THICKNESS_INIT,\
                        THICKNESS_MAX, lambda x: x)
   cv2.setMouseCallback(VERTICES_W_LBL, remove_vertices, image_display_c)
   while (1):
      font_size = cv2.getTrackbarPos(TRACKBAR_FONTSIZE, VERTICES_W_LBL) * FONTSIZE_BASE
      font_thickness = cv2.getTrackbarPos(TRACKBAR_THICKNESS, VERTICES_W_LBL) * THICKNESS_BASE
      image_display_c = image_display.copy()
      highlight_vertices(image_display_c, nodes, tW, tH)
      label_vertices(image_display_c, ref_pos, rel_pos, font_size, font_thickness)
      cv2.imshow(VERTICES_W_LBL, image_display_c)
      k = cv2.waitKey(1)
      if k & 0xFF == 13:
         break
      else:
         if k == 2490368: # up
            rel_pos = (rel_pos[0], rel_pos[1] - 1)
         elif k == 2621440: # down
            rel_pos = (rel_pos[0], rel_pos[1] + 1)
         elif k == 2424832: # left
            rel_pos = (rel_pos[0] - 1, rel_pos[1])
         elif k == 2555904: # right
            rel_pos = (rel_pos[0] + 1, rel_pos[1])
   
   print("Current vertices:")
   print_list(nodes)
   cont = True
   initiate_UI(image_display_c, VERTICES_W_LBL, select,\
      "Please select all the remaining vertices on " + VERTICES_W_LBL)
   cv2.destroyWindow(VERTICES_W_LBL)
   nodes_center = get_center_pos(nodes, tW, tH)
   print("Current vertices:")
   print_list(nodes_center)
   label_vertices(image_display_c, nodes, rel_pos, font_size, font_thickness, tW, tH)
   cv2.startWindowThread()
   cv2.namedWindow(VERTICES_W_LBL)
   cv2.imshow(VERTICES_W_LBL, image_display_c)
   cv2.waitKey(1)
   return nodes, nodes_center, rel_pos, font_size, font_thickness


def sort_vertices(nodes, image_display, window_name, rel_pos, font_size, font_thickness):
   """Interacts with the user to sort the vertices.
   Parameters
   ----------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices.
   image_display : numpy matrix of integers
      The image that is intended to be displayed to the user.
   window_name : string
      The name of the window that is used to display the image.
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   Returns
   -------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices while the
      false ones(decided by the user) has been removed.
   """
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
                     if is_valid_type(index, int, "Please provide a valid " +
                                       "integer."):
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
                              print("The given input is not a valid " +
                                    "arithmetic sequence!")
                           else:
                              valid = True
                  except:
                     print("Please provide a sequence of valid integers.")
            else:
               print("Cannot sort the vertices, check the method indicating " +
                     "value.")
               sys.exit(1)
            for i in range(len(index_list)):
               result[index_list[i] - BASE] = nodes[i]
            nodes = result
            highlight_vertices(image_display, nodes, tW, tH)
            label_vertices(image_display, nodes, rel_pos, font_size,
                           font_thickness)
            cv2.startWindowThread()
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, image_display)
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

def centralize(image_bin):
   """Attempts to clean the noises to some degree. Ths function adapts a voting
   method: every pixel with a value 1 will vote for the sorrounding pixels
   which also have a value 1. After a few rounds the center skeleton will get
   a much higher vote than the rest, hence the noises can be reduced.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      The binary image that is being studied.
   Returns
   -------
   """
   global denoised
   matrix = np.zeros(image_bin.shape)
   for i in range(4):
      matrix2 = matrix.copy()
      for y in range(image_bin.shape[0]):
         for x in range(image_bin.shape[1]):
            if image_bin[y][x] == 1:
               if i == 0:
                  matrix2[y][x] = 1
               else:
                  nv = get_neighborhood_values(matrix, (x, y))
                  for j in range(1, len(nv)):
                     matrix2[y][x] += nv[j]
      matrix = matrix2
   max_value = 0
   for y in range(matrix.shape[0]):
      for x in range(matrix.shape[1]):
         max_value = max(max_value, matrix[y][x])
   for y in range(matrix.shape[0]):
      for x in range(matrix.shape[1]):
         matrix[y][x] = matrix[y][x] / max_value
   image_bin2 = np.zeros(image_bin.shape, np.uint8)
   for y in range(matrix.shape[0]):
      for x in range(matrix.shape[1]):
         image_bin2[y][x] = matrix[y][x] * 255
   cv2.imshow(NOISE_REDUCTION, image_bin2)
   cv2.waitKey(1)
   # add a trackbar here
   print("Slide for the best threshold value, and hit Esc to finish.")


def noise_reduction(image_gray, filename, nodes_center, radius):
   """This function interacts with the user to help them perform mathematical 
   morphology operations on the grayscale image, attempting to reduce the noise
   of the original image.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      The grayscale version of the original image.
   break_point : int
      A value indicating the threshold between the background and the content.
   filename : String
      The file name withough suffix, used in saving.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      Half of the length of the diagonal of the template.
   Returns
   -------
   result : numpy matrix of integers
      The result image.
   """
   image_bin_inv = get_binary_image(image_gray, THRESHOLD)
   show_binary_image(image_bin_inv, NOISE_REDUCTION)
   image_stack = []
   message_stack = []
   image_stack.append(image_bin_inv)
   message_stack.append('')
   kernel_shape_str = KERNEL_SHAPE
   kernel_shape = get_kernel_shape(kernel_shape_str)
   kernel_size = KERNEL_SIZE
   kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
   response = ''
   last_step = ''
   result = image_bin_inv
   print("Current kernel shape and size is: " + kernel_shape_str +
         str(kernel_size))
   print("Pelease indicate which operation to perform:")
   print("(a)djust kernel(this cannot be undone)")
   print("(d)ilation")
   print("(e)rosion")
   print("(p)roceed")
   print("(s)ave(this cannot be undone)")
   print("(t)hin")
   print("(u)ndo" + message_stack[-1])
   while len(response) == 0:
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
                  if new_size > min(image_gray.shape[0], image_gray.shape[1])\
                     or new_size <= 0:
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
         elif response[0] in ['d', 'e']:
            image_temp = image_stack[-1].copy()
            if response[0] == 'd':
               print("Performing dilation....")
               image_temp = denoise(image_temp, cv2.dilate, kernel, 1)
               last_step = " (last step was (d)ilation)"
            elif response[0] == 'e':
               print("Performing erosion....")
               image_temp = denoise(image_temp, cv2.erode, kernel, 1)
               last_step = " (last step was (e)rosion)"
            image_stack.append(image_temp)
            message_stack.append(last_step)
            show_binary_image(image_temp, NOISE_REDUCTION)
            print("Operation complete.")
            response = ''
         elif response[0] == 'u':
            if len(image_stack) > 1:
               image_stack.pop()
               message_stack.pop()
               image_temp = image_stack[-1]
               show_binary_image(image_temp, NOISE_REDUCTION)
            else:
               print("There is no last step.")
            response = ''
         elif response[0] == 'p':
            break
         elif response[0] == 's':
            print("Saving image data....")
            np.save(DATA_PATH + filename + '.npy', image_stack[-1])
            f = open(DATA_PATH + filename + '.dat', 'w')
            f.write(str(nodes_center) + "\n")
            f.write(str(radius))
            f.close()
            print("Saving complete.")
            response = ''
         elif response[0] == 't':
            print("Performing image thinning, this may take some time....")
            image_temp = thin(image_stack[-1].copy())
            print("Image thinning complete.")
            image_stack.append(image_temp)
            message_stack.append(" (last step was (t)hin")
            cv2.destroyWindow(NOISE_REDUCTION)
            show_binary_image(image_temp, NOISE_REDUCTION)
            response = ''
         else:
            print("Invalid input, please try again!")
            response = ''
         print("Current kernel shape and size is: " + kernel_shape_str +
               str(kernel_size))
         print("Pelease indicate which operation to perform:")
         print("(a)djust kernel(this cannot be undone)")
         print("(d)ilation")
         print("(e)rosion")
         print("(p)roceed")
         print("(s)ave(this cannot be undone)")
         print("(t)hin")
         print("(u)ndo" + message_stack[-1])
   cv2.destroyWindow(NOISE_REDUCTION)
   if len(image_stack) > 0:
      result = image_stack.pop()
   return result

def hide_vertices(image, nodes_center, radius, color = 0):
   """Puts a circle filled with the background color on each vertex.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is being studied.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   Returns
   -------
   None
   """
   for n in nodes_center:
      cv2.circle(image, n, int(radius), color, cv2.FILLED)
      
######################### Edge Extraction Subsection ##########################
"""This is the most techincal part of this project. There are multiple versions
of method to use, however only one should be called in a compiled program.
Currently method3 is being used, as it is the best one so far. Thre others are
previous versions.
"""

def method1(image_work, nodes_center, radius):
   """This function attempts to extract all the edges from the image with
   vertices being hidden.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      A value indicating the size of the block that is used to hide the
      vertices.
   Returns
   -------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   """
   E = []
   endpoints = get_endpoints(image_work, nodes_center, radius)
   deg_seq = []
   for i in range(len(endpoints)):
      deg_seq.append(len(endpoints[i]))
   trails = []

   for i in range(len(endpoints)):
      for j in range(len(endpoints[i])):
         edge, trail = get_edge(image_work, endpoints[i][j], [], [],\
                                 nodes_center, i, radius)
         if edge != PLACE_HOLDER_EDGE and not edge in E and\
            not [edge[1], edge[0]] in E:
            E.append(edge)
            trails.append(trail)
   image_temp = np.zeros(image_work.shape, np.uint8)
   
   
   '''
   edge, trail = get_edge(image_work, endpoints[0][0], [], [], nodes_center,\
                           0, endpoints, radius)
   print(edge)
   print(len(trail))
   
   
   image_temp = np.zeros(image_work.shape, np.uint8)
   for p in trail:
      image_temp[p[1], p[0]] = 1
   show_binary_image(image_temp, "trails", True)
   '''
   
   return E

def method2(image_work, nodes_center, radius):
   """An alternative way to obtain the edges of a graph.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      Half of the length of the diagonal of the template.
   Returns
   -------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   deg_seq : List[List[int]]
      The nonstandard degree sequence of the graph.
   """
   E = []
   endpoints = get_endpoints(image_work, nodes_center, radius)
   image_work2 = get_binary_image(image_work.copy(), 0, 255)
   for i in range(len(nodes_center)):
      cv2.putText(image_work2, str(i + BASE), nodes_center[i], cv2.FONT_HERSHEY_SIMPLEX,\
            FONTSIZE_BASE, 255, THICKNESS_BASE, cv2.LINE_AA, False)
   cv2.imshow("image_work2", image_work2)
   cv2.waitKey(1)
   deg_seq = []
   intersections, intersections_skirt = \
      find_intersections(image_work, endpoints[0][0])
   linked_to, outgoing, v2i = \
      construct_network(image_work, endpoints, intersections,\
                        intersections_skirt, nodes_center)
   merged, bridges = merge_intersections(image_work, intersections, linked_to,\
                                          outgoing)
   bridges_array = [-1] * len(intersections)
   '''
   image_copy = image_work.copy()
   for i in intersections:
      cv2.putText(image_copy, str(intersections.index(i)), (i[0], i[1]),\
                  cv2.FONT_HERSHEY_SIMPLEX, FONTSIZE_BASE, 255, THICKNESS_BASE,\
                  cv2.LINE_AA, False)
   cv2.imshow("image_copy", image_copy)
   cv2.waitKey()
   '''
   for pair in bridges:
      bridges_array[pair[0]] = pair[1]
      bridges_array[pair[1]] = pair[0]
   viv = restore_graph(v2i, linked_to, outgoing, bridges_array)
   intersections_skirt_all = []
   for iskirt in intersections_skirt:
      intersections_skirt_all += iskirt
   endpoints_all = []
   v2v = []
   for i in range(len(viv)):
      v2v.append([])
   for i in range(len(endpoints)):
      for ep in endpoints[i]:
         find_vertices(image_work, endpoints, intersections_skirt_all, v2v, [],\
                        ep, i)
   result = []
   for i in range(len(endpoints)):
      result.append(viv[i] + v2v[i])
   for v1 in range(len(result)):
      for v2 in result[v1]:
         if not [v1, v2] in E and not [v2, v1] in E:
            E.append([v1, v2])
   return E

def method3(image_work, nodes_center, radius):
   """An alternative way to obtain the edges of a graph.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      Half of the length of the diagonal of the template.
   Returns
   -------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   """
   global nodes, nodes_real, nodes_unreal, endpoints
   E = []
   endpoints = get_endpoints(image_work, nodes_center, radius)
   nodes_real, nodes_unreal = construct_network3(image_work, nodes_center, endpoints)
   nodes = merge_nodes(nodes_real, nodes_unreal, radius * (0.5 + 0.1 * R_FACTOR_INIT), image_work)
   
   print("Slide for a desired value so that all the nodes are correctly " +\
         "merged, and hit Return when finish.")
   image_bw = get_binary_image(image_work.copy(), 0, 255)
   image_bw2 = image_bw.copy()
   for n in nodes:
      cv2.circle(image_bw2, (int(n.location[0]), int(n.location[1])), 5, 255)
   cv2.imshow(NODES, image_bw2)
   cv2.waitKey(1)
   cv2.createTrackbar(TRACKBAR_RFACTOR, NODES, R_FACTOR_INIT, R_FACTOR_MAX,\
                        lambda x: set_rfactor(image_work, TRACKBAR_RFACTOR, NODES, radius))
   while (1):
      factor = cv2.getTrackbarPos(TRACKBAR_RFACTOR, NODES)
      image_bw2 = image_bw.copy()
      for n in nodes:
         cv2.circle(image_bw2, (int(n.location[0]), int(n.location[1])), 5, 255)
      cv2.imshow(NODES, image_bw2)
      k = cv2.waitKey(1)
      if k & 0xFF == 13:
         break
   
   nodes_real_final = []
   nodes_unreal_final = []
   for n in nodes:
      if n.is_real:
         nodes_real_final.append(n)
      else:
         nodes_unreal_final.append(n)
   
   config_links(nodes)
   # present(get_binary_image(image_work.copy(), 0, 255), nodes, True)
   restore_graph3(nodes_real_final, E)
   return E

def extract_edges(image_work, nodes_center, radius, method = method3):
   """An alternative way to obtain the edges of a graph.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      Half of the length of the diagonal of the template.
   Returns
   -------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   """
   E = method(image_work, nodes_center, radius)
   return E

############################  END OF SUBSECTION  ##############################

def correct_edges(image_work, E, nodes_center):
   """Allows the user to manually correct the edges.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is being studied.
   E : List[[int, int]]
      The edge list.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   Returns
   -------
   None
   """
   global image_original, cont, removed_stack, image_stack, start_linking
   image_original = image_work.copy()
   removed_stack = []
   image_stack = []
   start_linking = False
   for edge in E:
      des1 = edge[0]
      des2 = edge[1]
      cv2.line(image_work, nodes_center[des1],\
               nodes_center[des2], (0, 0, 255), 2)
   response = ""
   while response == "":
      response = input("Do you want to correct the edge(s) manually?(y/n) ")
      if len(response) > 0:
         if (response[0] == 'y' or response[0] == 'Y'):
            initiate_UI(image_work, OUTPUT, remove, "Remove false edges in " +
                        "the output window, and hit Return when finished.")
            image_work = image_original.copy()
            for edge in E:
               des1 = edge[0]
               des2 = edge[1]
               cv2.line(image_work, nodes_center[des1],\
                        nodes_center[des2], (0, 0, 255), 2)
            initiate_UI(image_work, OUTPUT, add, "Add undetected edges in " +
                        "the output window, and hit Return when finished.")
         elif response[0] != 'n' and response[0] != 'N':
            response = ""
            print("Invalis input.")
   deg_seq = [0] * len(nodes_center)
   for e in E:
      deg_seq[e[0]] += 1
      deg_seq[e[1]] += 1
   E = shift_indices(E)
   return E, deg_seq

def output(E, deg_seq):
   """Shows all the edges.
   Parameters
   ----------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   Returns
   -------
   None
   """
   print("Printing outputs....")
   print("\"vertices\": " + str([i for i in range(BASE, len(deg_seq) + BASE)]))
   print("\"edges\": " + str(E))
   print("\"degrees\": " + str(deg_seq))
   print("Displaying edges....")

#=============================================================================#
#                             Modes of Execution                              #

def start_regular_mode():
   """Starts the program with regular mode.
   """
   global ix, iy, drawing, template_num, tH, tW, image_original, E, nodes_center
   # Set up the graph image.
   graph, graph_gray, break_point, filename = get_image()
   image_original = graph.copy()
   filename_array = filename.split('.')
   filename = filename_array[0]
   
   # Some global variables.
   ix, iy = -1, -1
   drawing = False
   template = None
   template_num = None
   
   # Obtain a desired binary version of the graph so that the noises can be
   # reduced at the beginning.
   graph_bin = get_graph_bin(graph_gray, TRACKBAR_THRESHOLD, GRAPH)
   
   # Crop an example of the vertices.
   graph_gray2 = get_binary_image(graph_bin, 0, 255)
   graph2 = cv2.cvtColor(graph_gray2, cv2.COLOR_GRAY2BGR)
   while template_num is None:
      cv2.destroyWindow(GRAPH)
      initiate_UI(graph2, GRAPH, crop,\
         "Please crop an example of the vertices in " + GRAPH + " window.")

   # Process the template.
   template, (tH, tW), radius = process_template(graph2, template_num)
   
   # Find all the vertices. In particular variable nodes stores a list of
   # nodes' upper-right corner.
   nodes, nodes_center, rel_pos, font_size, font_thickness =\
      extract_vertices(graph.copy(), graph_gray.copy(), template, tW, tH)
   
   # If neccesary, sort the vertices such that the order matches the given one.
   nodes = sort_vertices(nodes, graph.copy(), "Vertices with Labels", rel_pos,
                           font_size, font_thickness)
   

   # Process the image, then extract all the edges.
   graph_work = noise_reduction(graph_gray2, filename, nodes_center, radius)
   hide_vertices(graph_work, nodes_center, radius)
   
   '''
   graph_display = get_binary_image(graph_work.copy(), 0, 255)
   cv2.imshow("graph_display", graph_display)
   cv2.waitKey()
   '''
   E = extract_edges(graph_work, nodes_center, radius)
   print("E = " + str(E))
   graph_display = graph.copy();
   for edge in E:
      des1 = edge[0]
      des2 = edge[1]
      cv2.line(graph_display, nodes_center[des1],\
               nodes_center[des2], (0, 0, 255), 2)
   cv2.imshow(OUTPUT, graph_display)
   cv2.waitKey(1)
   E, deg_seq = correct_edges(graph.copy(), toLists(E), nodes_center)
   output(E, deg_seq)
   get_invariants(E, len(deg_seq))
   
def start_analysis_mode():
   """In this mode, the input will be taken from a previous data source.
   """
   global E, nodes_center
   graph = cv2.imread("graph_input/engine.jpg")
   graph_work, nodes_center, radius = load("engine")
   
   
   hide_vertices(graph_work, nodes_center, radius)
   for i in range(len(nodes_center)):
      n = nodes_center[i]
      cv2.putText(graph_work, str(i), (int(n[0] + 2),\
                  int(n[1]) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255,\
                  1, cv2.LINE_AA, False)
   show_binary_image(graph_work, "graph_work")
   E = method3(graph_work, nodes_center, radius)
   print("E = " + str(E))
   print("len(E) = " + str(len(E)))
   '''
   E = extract_edges2(graph_work, nodes_center, radius)
   E, deg_seq = correct_edges(graph.copy(), E, nodes_center)
   print(E)
   print(deg_seq)
   
   
   f = open('temp.txt', 'w')
   for y in range(270, 290):
      line = str(y % 10) + ' '
      for x in range(218, 235):
         line += str(graph_work[y, x]) + ' '
      line += '\n'
      f.write(line)
   f.close()
   '''
#                               End of Section                                #
###############################################################################
#=============================================================================#
#                             Under Development                               #

#                               End of Section                                #
###############################################################################
###############################################################################
#                              Executing Codes                                #
if __name__ == "__main__":
   mode = ""
   while mode == "":
      mode = input("Start regular mode? (y/n) ")
      if len(mode) > 0:
         if mode[0] == 'y' or mode[0] == 'Y':
            start_regular_mode()
            halt = input("HALT (press enter to end)")
         elif mode[0] == 'n' or mode[0] == 'N':
            start_analysis_mode()
            halt = input("HALT (press enter to end)")
         else:
            print("Cannot recognize input.")
            mode = ""
            

'''
talk to OCR people 
'''