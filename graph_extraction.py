'''
main.py
Sha Lai
9/16/2016

This file contains most of the core functions being used in the project.

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
the true vertices. Once this step is finalized, a list of vertices can be
obtained. Moreover for the rest of the program whenever we need to deal with
a full list of elements that are all related to the vertices(the center
coordinate of each vertex for example), the elements in the new list must be in
the same order as in the nodes and the nodes_center. In other words we have
assigned each vertex an ID number implicitly.

Next we want to find the edges. There are two main steps involved. Firstly we
want to get rid of the noise on the image as much as possible. A commonly used
approach is mathematical morphology. In this file the denoise function can help
to do the job. When the image is in a desired stage after the noise reduction,
the function thin needs to be called such that the skeleton of the content can
be extracted. And then the user may want to hide the vertices and make sure
that the result is also in a desired view(no extra overlaps between edges for
example). Note that after this step a binary version of the image should be
obtained and we will use it throughout the rest of the program. We can now
start attempting to extract the edges.

From now on assume that in the binary image the content is represented by 1s
and the background is marked by 0s. The first thint to do is to find all the
starting points of the edges. With the vertices hidden, it is easy to scan all
the content pixels and find the ones with the smallest distances to some
located vertex. The output of this step should be a list of sets of pixels
where the ith set in the list should contain the coordinates of the pixels that
are at the end of the edges connecting to the ith vertex. Next we need to
extract the edges. There are two ways to do this.

Method 1: Dungeon Exploration
I give it this name because the idea is somewha similar to RPG dungeon
exploration. For each starting point, the algorithm to determine which vertex
is the corresponding edge linked to can be described as follows:

1. Check if the current position has already reached a vertex that is not the
   starting one. If so then we are done and return, otherwise go to step 2.
2. Obtain a neighborhood of the current potition, examine the neighborhood and
   extract a list of pixels that are marked 1 and are not in the known list.
   Add them into the known list. If the length of the list is 0 then go to step
   3. Otherwise go to step 4.
3. Abort since this is the end of a line and there is no vertex to link to.
4. If the length of the list trail is 0 then for each candidate pixel
   go back to step 1. Otherwise, if the length is 1 and if the current position
   is not in the conjunction of two edges then append the current position to
   the trail list, if not in the conjunction then take the only candidate back
   to the first step. If there are more than one candidate, then determine if
   the current position is a cross point. If so then check if it is also in a
   conjunction part. If not append the current position to the trail list. If
   this is not a cross then the current position also needs to be added to the
   trail list. Proceed to step 5.
5. Using the trail list, construct a list of vectors between each pair of
   adjacent points in the trail, then compute a weighted sum of these vectors.
   For each candidate compute a vector that points to them from the current
   position. Compare each candidate vector and the sum vector, choose the one
   with the smallest angle with the sum vector and take that corresponding
   candidate pixel to step 1. If the returned result is not a valid vertex then
   choose the 2nd closest vector and its corresponding destination position.
   Repeat until a valid result is obtained or the options are run out.

The way in which a cross is detected is simple: for each candidate pixel
compute a vector that points from the current position to it and for each pair
of these candidate vectors check if their angle is greater than or equal to 90
degree. The idea behind the approach to decide which direction to go next at a
cross point is to make the decision based on a weighted sum of the previous
experience and the previous experience does not include any conjunction part.
Currently what is problematic is the weighting function applied on the vectors.
For now I have not yet found a really "good" funtion to use as much as I do
believe that my approach is reasonable.

Method 2: DnC + BFS
This method is much more complicated than the other one, but it does yield a
more accurate result. It takes the idea from divide and conquer, and 
breadth-first-search. Here is s series of steps of this method:

1. Find all the intersections.
2. Record what other intersections or vertices is each intersection linking to,
   and record the outgoing vectors using the same weighted sum approach as
   discussed before. During implementation, this step is broken into two:
   extract the network of intersection - intersection and vertex - intersection
   first, and then get the network of vertex - vertex.
3. Find all the possible ways to pair up intersections that have a link in
   between and have the same among of outgoing vectors such that, at the end,
   there is no isolated intersection.
4. Find the pairing option leading to the minimum error sum. Here the error is
   defined to be the sum of the cos values between two paired vectors. For
   example, let 1, 2, 3, 4, 5, 6 denote the intersections and let [(1, 2), (3,
   4), (5, 6)] be a possible pairing option, if intersection 1 has 2 outgoing
   vectors(other than the link to intersection 2) v1 and v2 and hence
   intersection 2 also has 2 outgoing vectors w1 and w2, the error of this
   pair is min(cos(v1, w1) + cos(v2, w2), cos(v1, w2) + cos(v2, w1)). In short,
   we are looking for some way to merge intersections such that, at the end,
   each merged intersection will have a nice shape without the link in between.
   The "merging" is done by establishing a special "bridge" between two
   intersections.
5. Starting from each vertex, retrieve the graph by iteratively/recurrsively
   checking the next destination. If the next destination is an intersection,
   use the recorded vector corresponding to the edge between the current point
   and the intersection to, at the other end of the "bridge", find an outgoing
   vector as we did before. If the next destination is a vertex then apparently
   we are done.
   
This method will work due to a key observation. After an image is thinned and
the vertices are hidden, there are three structures formed by edge pixels in
the image:

   00100            00100              01010
   00100            00100              00100
   00100            11111              00100
   00100            00100              00100
   00100            00100              01010
   
The first case is an uninterrupted curve, the second a perfect cross, and the
third one is a typical intersection area. The first two cases are easy to
handle while the third one is not. The first step above is somewhat like divide
and conquer: if we find all the intersection points and treat them like
vertices (for now), then in the image there is only the first type of structure
remaining since the previous intersection points no longer belong to part of
edges. After finishing pairing up the intersections, the links between each
intersections in a pair will no longer be considered while determining the
directions at any future step, so each of these pairs is essentially the second
type of structure.

Overall, the first method is not as good as the second one in terms of accuracy
simply because at each iteration, the information is extremely limited; while
the second one obtains the information of all the intersections first, and then
analyze them in order to yield the best result (the error is minimized). It is
not hard to tell that the time complexity of the second method is significantly
greater than that of the first one.
'''

from common import *
###############################################################################
#                             Helper Functions                                #

def highlight_vertices(image, nodes, tW, tH):
   """Highlights each vertex with a frame.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is will be drawn on.
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the vertices.
   tW and tH : int
      Dimension values of the template.
   show_indices : boolean
      Indices will be displayed if True.
   using_console : boolean
      True if the program is run on the console. This argument is needed since
      OpenCV stores the RGB color in a different order.
   Return
   ------
   None
      Everything will be drawn on the given image.
   """
   for i in range(len(nodes)):
      cv2.rectangle(image, nodes[i], (nodes[i][0] + tW,  nodes[i][1] + tH),
                     RECT_COLOR, RECT_THICKNESS)

def label_vertices(image, ref_pos, rel_pos, font_size = FONT_SIZE,
                  font_thinkness = FONT_THICKNESS):
   """Labels all the vertices with indices.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that will be used to labeled.
   ref_pos : List[(int, int)]
      List of original coordinates for each vertex(the top-right corner).
   rel_pos : (int, int)
      Coordinates relaive to the original ones.
   font_size : float
      Indicating the font size.
   font_thickness : int
      Indicating how thick the text is.
   Returns
   -------
   None
   """
   for i in range(len(ref_pos)):
      x = abs(ref_pos[i][0] + rel_pos[0])
      y = abs(ref_pos[i][1] + rel_pos[1])
      if x >= image.shape[1]:
         x -= 2 * rel_pos[0]
      if y >= image.shape[0]:
         y -= 2 * rel_pos[1]
      position = (x, y)
      cv2.putText(image, str(i + BASE), position, cv2.FONT_HERSHEY_SIMPLEX,\
         font_size, FONT_COLOR, font_thinkness, cv2.LINE_AA, False)

def draw_edges(image, edges_center, using_console = True):
   """Puts a label near the center pixel of each edge.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is will be drawn on.
   edges_center : List[[int, int]]
      Stores the coordinates of center pixel of each edge.
   using_console : boolean
      True if the program is run on the console. This argument is needed since
      OpenCV stores the RGB color in a different order.
   Returns
   -------
   None
      Everything will be drawn on the given image.
   """
   if not using_console:
      font_color = FONT_COLOR_G
   else:
      font_color = FONT_COLOR
   for i in range(len(edges_center)):
      cv2.putText(image, str(i + BASE), (edges_center[i][0],\
                  edges_center[i][1]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, \
                  font_color, FONT_THICKNESS, cv2.LINE_AA)
      

def get_threshold(image_gray, show_detail = False):
   """Determines the threshold of color values which will be used to distinguish
   the background and the actual content. If the METHOD is 'STATIC' then a set
   value will be used immediately while if the METHOD is 'DYNAMIC' then the
   mode of all the pixel values will be used.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      The grayscale image that is being studied.
   show_detail : boolean
      More printing statements if True. This is for debugging purpose.
   Returns
   -------
   threshold : int
      The threshold of color values.
   """
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
      sys.exit('Cannot recognize the METHOD, please check the common file.')


def get_distance(p1, p2):
   """Computes the Euclidean distance between two points in 2D.
   Parameters
   ----------
   p1 and p2 : [int, int]
      Represents coordinates.
   """
   return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def remove_nodes(user_input, nodes):
   """Removes false vertices based on the user input.
   Parameters
   ----------
   user_input: string
      A string of edge indices subject to be removed.
   nodes : List[(int, int)]
      Each coordinate marks the upper-right corner of a piece of the original
      image.
   Returns
   -------
   nodes : same as above
      The false ones have been removed.
   """
   index_remove = user_input.split()
   for i in index_remove:
      nodes[int(i) - BASE] = PLACE_HOLDER_COOR
   nodes = [element for element in nodes if element != PLACE_HOLDER_COOR]
   return nodes


def get_center_pos(nodes, tW, tH):
   """Calculates the center coordinates of each vertex, based on the template
   taken by the user in some early step.
   Parameters
   ----------
   nodes : List[(int, int)]
      Each coordinate marks the upper-right corner of a piece of the original
      image.
   tW and tH : int
      The dimension values of the template that was used to find the vertices.
   Returns
   -------
   nodes_center :
   """
   nodes_center = []
   for node in nodes:
      nodes_center.append((int(node[0] + tW / 2), int(node[1] + tH / 2)))
   return nodes_center


def remove_edges(user_input, E, edges_center):
   """Removes false edges based on the user input.
   Parameters
   ----------
   user_input: string
      A string of edge indices subject to be removed.
   E : List[(int, int)]
      Each tuple contains two indices of the vertices in the vertex list.
   edges_center : List[(int, int)]
      Each element marks the estimated position of the center pixel of an edge.
   Returns
   -------
   E and edges_center : same as above
      The false ones have been removed.
   """
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

###########################  IMAGE THINNING  ##################################
"""The following functions implements a modified version of zhang-seun's image
thinning algorithm without too many explanations. The user can search for the
algorithm for more details. In order to perform image thinning, only the
function thin needs to be called by the client program.
"""
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

def thin(image_bin):
   """Thins a given binary image with zhang-seun's algorithm, assuming that in
   the image 1 represents the actual content while 0 represents the background.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      The binary image that is being studied.
   Returns
   -------
   mat : numpy matrix of integers
      A thinned version of the original binary image.
   """
   mat = image_bin.copy()
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
   
   # This is an extra part. Here we further reduce unnecessary pixels while
   # maintaining the connectivity by removing T-shape structures. The main
   # purpose is to increase the accuracy of intersection recognition.
   for y in range(mat.shape[0]):
      for x in range(mat.shape[1]):
         if mat[y, x] == 1:
            n = get_neighborhood(mat, [x, y])
            nv = get_neighborhood_values(mat, [x, y])
            count = 0
            for i in [1, 3, 5, 7]:
               if nv[i] == 1:
                  count += 1
            if count > 2:
               mat[y, x] = 0
   return mat

############################  END OF SUBSECTION  ##############################




def denoise(image_bin, method, kernel, itr = 0):
   """Performs mathematical morphology operation on the provided binary image
   in order to reduce the noise.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      A binary image that is being studied.
   method : function
      This function must either be cv2.dilate or cv2.erode in order to make
      sure the program works correctly.
   kernel : numpy matrix of integers
      The dimension values are suggested to be odd numbers.
   """
   result = method(image_bin.copy(), kernel, iterations = itr)
   return result


def get_neighborhood(image, location):
   """Extracts the coordinates of the points in the neighborhood.
   Parameters
   ----------
   image : numpy matrix of intergers
      The image that is being studied.
   location : [int, int]
      The coordinate in the form [x, y] that marks the desired position.
   Returns
   -------
   p : List[[int, int]]
      Each element is a coordinate in the form [x, y]. The indices are
      associated with the relative positions in the neighborhood in this form:
      8 1 2
      7 0 3
      6 5 4
      Additionally if the location happens to be on an edge or a corner of the
      image then [PLACE_HOLDER_INT, PLACE_HOLDER_INT] will be used for
      out-of-reach points.
   """
   temp = [PLACE_HOLDER_INT, PLACE_HOLDER_INT]
   p = [temp[:]] * 9
   p[0] = location[:]
   if location[1] - 1 >= 0:
      p[1] = [location[0], location[1] - 1]
      if location[0] + 1 < image.shape[1]:
         p[2] = [location[0] + 1, location[1] - 1]
   if location[0] + 1 < image.shape[1]:
      p[3] = [location[0] + 1, location[1]]
      if location[1] + 1 < image.shape[0]:
         p[4] = [location[0] + 1, location[1] + 1]
   if location[1] + 1 < image.shape[0]:
      p[5] = [location[0], location[1] + 1]
      if location[0] - 1 >= 0:
         p[6] = [location[0] - 1, location[1] + 1]
   if location[0] - 1 >= 0:
      p[7] = [location[0] - 1, location[1]]
      if location[1] - 1 >= 0:
         p[8] = [location[0] - 1, location[1] - 1]
   return p


def get_neighborhood_values(image, location):
   """Extracts the pixel values in the neighborhood.
   Parameters
   ----------
   image : numpy matrix of intergers
      The image that is being studied.
   location : [int, int]
      The coordinate in the form [x, y] that marks the desired position.
   Returns
   -------
   p : List[int]
      A list of pixel values in the neighborhood. The indices are associated
      with the relative positions in the neighborhood in this form:
      8 1 2
      7 0 3
      6 5 4
      Additionally is the current position happens to be on an edge or a corner
      of the image then PLACE_HOLDER_INT will be used to represent the pixel
      value.
   """
   p = [PLACE_HOLDER_INT] * 9
   p[0] = image[location[1], location[0]]
   if location[1] - 1 >= 0:
      p[1] = image[location[1] - 1, location[0]]
      if location[0] + 1 < image.shape[1]:
         p[2] = image[location[1] - 1, location[0] + 1]
   if location[0] + 1 < image.shape[1]:
      p[3] = image[location[1], location[0] + 1]
      if location[1] + 1 < image.shape[0]:
         p[4] = image[location[1] + 1, location[0] + 1]
   if location[1] + 1 < image.shape[0]:
      p[5] = image[location[1] + 1, location[0]]
      if location[0] - 1 >= 0:
         p[6] = image[location[1] + 1, location[0] - 1]
   if location[0] - 1 >= 0:
      p[7] = image[location[1], location[0] - 1]
      if location[1] - 1 >= 0:
         p[8] = image[location[1] - 1, location[0] - 1]
   return p


def print_neighborhood_values(nv):
   """Prints the pixel values in the neighborhood.
   Parameters
   ----------
   nv : List[int]
      Represents the neighborhood pixel values. A neighborhood has the form
      8 1 2
      7 0 3
      6 5 4
      where each number is associated with the index in the list.
   """
   print(str(nv[8]) + " " + str(nv[1]) + " " + str(nv[2]))
   print(str(nv[7]) + " " + str(nv[0]) + " " + str(nv[3]))
   print(str(nv[6]) + " " + str(nv[5]) + " " + str(nv[4]))


def get_vector(p_from, p_to):
   """Computes the vector from one point to another.
   Parameters
   ----------
   p_from and p_to : [int, int]
      Coordintates in the form [x, y].
   Returns
   -------
   vector : [int, int]
   """
   return [p_to[0] - p_from[0], p_to[1] - p_from[1]]


def get_weight(x):
   """Computes the weighting coefficient for a given variable.
   Parameters
   ----------
   x : int
      This serves as the independent variable in the function f(x).
   Returns
   -------
   f(x) : float
      A weighting coefficient.
   """
   # using a normal distribution
   mu = 1
   sigma = 2
   f = (exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)) / (sqrt(2 * pi) * sigma)))
   return f


def get_vector_sum_in(vector_list):
   """Computes the weighted sum of a list of incoming vectors. Currently the
   weight function is set to be the normal distribution functino.
   Parameters
   ----------
   vector_list : List[[int, int]]
      Each element is a pair of coordinates in the form [x, y].
   Returns
   -------
   result : [float, float]
      The weighted sum of the given list of vectors.
   """
   result = [0, 0]
   for i in range(len(vector_list)):
      result[0] += vector_list[-1 - i][0] * get_weight(i)
      result[1] += vector_list[-1 - i][1] * get_weight(i)
   n = np.linalg.norm(result, 2)
   result[0] /= n
   result[1] /= n
   return result
   
def get_vector_sum_out(vector_list):
   """Computes the weighted sum of a list of outgoing vectors. Currently the
   weight function is set to be the normal distribution functino.
   Parameters
   ----------
   vector_list : List[[int, int]]
      Each element is a pair of coordinates in the form [x, y].
   Returns
   -------
   result : [float, float]
      The weighted sum of the given list of vectors.
   """
   result = [0, 0]
   for i in range(len(vector_list)):
      result[0] += vector_list[i][0] * get_weight(i)
      result[1] += vector_list[i][1] * get_weight(i)
   n = np.linalg.norm(result, 2)
   result[0] /= n
   result[1] /= n
   return result

def is_intersection(current_pos, next):
   """Determines if the current position is a cross between two curves.
   Parameters
   ----------
   current_pos : List[int]
      Stores the current coordinates in the form [x, y].
   next : List[[int, int]]
      Stores the candidate coordinates each of which is in the form [x, y].
   Returns
   -------
   True if the current position is a cross point, False otherwise.
   """
   if len(next) < 2:
      return False
   # an intersection can only exist if there are more than two elements in next
   vectors = []
   for i in range(len(next)):
      vectors.append(get_vector(current_pos, next[i]))
   for i in range(len(vectors) - 1):
      for j in range(i + 1, len(vectors)):
         cos_theta = np.inner(vectors[i], vectors[j]) /\
            (np.linalg.norm(vectors[i], 2) * np.linalg.norm(vectors[j], 2))
         if cos_theta <= 0:
            return True
   return False
   
def get_score(vec1, vec2, target = -1):
   """Computes the score of the two given vectors. Here the score is defined to
   be the square difference between their cosine value and the target value.
   Parameters
   ----------
   vec1, vec2 : List[List[int]]
      The given vectors.
   target : int
      The target value used to compare. The default is -1.
   Returns
   -------
   The square difference between their cosine value and the target value.
   """
   cos = np.inner(vec1, vec2) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))
   print("vec1 = " + str(vec1))
   print("vec2 = " + str(vec2))
   return pow(cos - target, 2)
#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

def process_template(template):
   """Extracts the edges in the template.
   Parameters
   ----------
   template : a numpy matrix of integers
   Returns
   -------
   template : a numpy matrix of integers
      Contains the edges(gap between pixel values).
   (tH, tW) : (int, int)
      Dimension values of the template.
   radius : float
      Half of the length of the diagonal of the template.
   """
   template = cv2.Canny(template, 50, 200)
   (tH, tW) = template.shape[:2]
   
   # this will serve as a threshold of the distance from some end point of
   # an edge to the center of a vertex
   radius = sqrt(pow((1 + SCALE_PERCENTAGE) * tH, 2) + pow((1 + 
      SCALE_PERCENTAGE) * tW, 2)) / 2
   return template, (tH, tW), radius


def locate_vertices(amount, image, template, tW, tH, nodes):
   """This function performs multi-scale template matching in order to locate
      all the vertices, contributed by Adrian Rosebrock from PyImageResearch.
      Parameters
      ----------
      amount : int
         The estimated amount of vertices that remains undetected.
      image : numpy matrix of integers
         The image that are being studied.
      template : numpy matrix of integers
         The cropped piece of the original image containing an example of the
         vertices.
      tW and tH : int
         The dimension values of the template.
      nodes : List[(int, int)]
         Stores the upper-right coordinate of the vertices  that have been
         detected so far.
      Returns
      -------
      nodes : List[(int, int)]
         This list is returned by reference.
   """
   for i in range(amount):
      found = None
      # rescale the image, for each scale, find the piece that has the best
      # matching score
      for scale in np.linspace(0.2, 1.0, 20)[::-1]:
         resized = imutils.resize(image, width = \
            int(image.shape[1] * scale))
         r = image.shape[1] / float(resized.shape[1])
        
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

      # image is not intended to be displayed
      cv2.rectangle(image, (startX, startY), (endX, endY), \
         (255, 255, 225), cv2.FILLED)
      nodes.append((int(maxLoc[0] * r), int(maxLoc[1] * r)))

   
def get_endpoints(image_bin, nodes_center, radius):
   """Finds the endpoints of the edge pixels and associate them with the vertices
   by indices.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   nodes_center : List[(int, int)]
      Each tuple is the coordinate of the center position of a vertex.
   radius : float
      The radius of the circle block covering the vertices.
   Returns
   -------
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   """
   endpoints = []
   for i in range(len(nodes_center)):
      endpoints.append([])
   for x in range(image_bin.shape[1]):
      for y in range(image_bin.shape[0]):
         candidates = []
         if image_bin[y, x] == 1:
            for i in range(len(nodes_center)):
               d = get_distance((x, y), nodes_center[i])
               if d < radius + 1:
                  new_end = True
                  endpoints_icopy = endpoints[i][:]
                  for ep in endpoints_icopy:
                     n = get_neighborhood(image_bin, ep)
                     if [x, y] in n:
                        new_end = False
                        if d < get_distance(ep, nodes_center[i]):
                           endpoints[i].remove(ep)
                           endpoints[i].append([x, y])
                  if new_end:      
                     endpoints[i].append([x, y])
   return endpoints

#=============================================================================#
#                                  Method 1                                   #

def get_edge(image_bin, current_pos, trail, known, nodes_center,
               starting_index, radius, is_joint = False, image_debug = None):
   """Returns an edge and it's trail. An edge is in the form (a, b) where a
   marks the index of the starting vertex and b the index of the ending
   one. A trail is a list of pixels that connect the two vertices, not
   including the ones in conjunctions between two or more edges.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      This is a binary image where 1 represents the content while 0 marks
      the background.
   current_pos : List[int]
      A two-element list representing the (x, y) coordinate of hte current
      position.
   trail : List[[int, int]]
      Each element list in trail represents a coordinate of the pixel
      connecting the current pixel and the starting one.
   known : set of lists of integers
      It stores all the coordinates of the pixels that have been examinied
      so far.
   nodes_center : List[(int, int)]
      Each tuple is the coordinate of the center position of a vertex.
   starting_index : int
      The index of the starting vertex in the vertex list.
   radius : float
      The radius of the circle block covering the vertices.
   is_joint : boolean
      True if the current_pos is in a conjunction between two or more
      vertices.
   image_debug : numpy matrix of integers
      This is a binary image where 1 represents the content while 0 marks
      the background, for debugging purpose.
   Returns
   -------
   edge : a tuple of integers
      If the function manages to find an edge, then a tuple of two integers
      each of which marks the index of a vertex in the vertex list will be
      returned. Otherwise a default tuple implying that the attempt fails
      will be returned.
   trail : List[[int, int]]
      Each element list in trail represents a coordinate of the pixel
      connecting the current pixel and the starting one.
   """
   for i in range(len(nodes_center)): # check if it has reached a vertex
      if get_distance(nodes_center[i], current_pos) < radius + 1 and\
         i != starting_index:
         #print("Vertex reached.")
         return [starting_index + BASE, i + BASE], trail
   n = get_neighborhood(image_bin, current_pos)
   nv = get_neighborhood_values(image_bin, current_pos)
   
   candidate_indices = []
   for i in range(1, len(n)):
      if nv[i] == 1 and not n[i] in known:
         candidate_indices.append(i)
         known.append(n[i])
   
   if DEBUG_MODE: # just more printing statements
      print("\ncurrent_pos = " + str(current_pos))
      print("len(trail) = " + str(len(trail)))
      print_neighborhood_values(nv)
      print("len(candidate_indices) = " + str(len(candidate_indices)))
   if image_debug is None:
      image_debug = image_bin.copy()
      
   if len(candidate_indices) == 0: # if this is the end of the line/curve
      #print("End of the line.")
      #print_neighborhood_values(nv)
      #cv2.circle(image_bin, current_pos, 5, 1, cv2.FILLED)
      return PLACE_HOLDER_EDGE, trail
   else: # if there is at least one possibly unknown pixel ahead
      if len(trail) == 0: # this is a starting point
         #print("This is just a start.")
         known.append(current_pos)
         for i in candidate_indices:
            edge, new_trail = get_edge(image_bin, n[i], [current_pos], known,\
                                       nodes_center, starting_index, radius,\
                                       is_joint, image_debug)
            if edge != PLACE_HOLDER_EDGE: # returns the found edge immediately
               return edge, new_trail
         return PLACE_HOLDER_COOR, trail
      else: # we are in the middle of some edge
         if len(candidate_indices) == 1: # no choice to make, go for the next
            if is_joint == False:
               trail.append(current_pos)
            return get_edge(image_bin, n[candidate_indices[0]], trail, known,\
                              nodes_center, starting_index, radius,\
                              is_joint, image_debug)
         else: # need to make a choice
            next = [] # stores the coordinate of the candidate pixels
            for i in candidate_indices:
               next.append(n[i])
            if is_intersection(current_pos, next): # if this is a cross point
               # cv2.circle(image_debug, (current_pos[0], current_pos[1]), 4, 1, 1)
               # image_show = get_binary_image(image_debug, 0, 255)
               # cv2.imshow("image_debug", image_show)
               # cv2.waitKey()
               # print("cross:")
               # print_neighborhood_values(nv)
               # print('')
               # print(current_pos)
               
               '''
               If is_joint is True, then this is a point where we are leaving
               the conjunction part, so is_joint should be turned to False;
               otherwise we are entering one.
               '''
               if is_joint:
                  is_joint = False
               else:
                  is_joint = True
                  trail.append(current_pos)
            else: # not a cross
               trail.append(current_pos)
               

            # The following codes intend to determine which direction to go.
            vector_list = []
            for i in range(len(trail) - 1): # sums the vectors along the trail
               vector_list.append(get_vector(trail[i], trail[i + 1]))
            vector_list.append(get_vector(trail[-1], current_pos))
            vector_sum = get_vector_sum_in(vector_list)
            
            '''
            Compute the cosine value of the angle between the sum vector and
            each candidate one, starting from the direction with the highest
            cosine value, that is, with the smallest angle, select the one that
            yields a valid edge.
            '''
            cos2next = {}
            for i in candidate_indices:
               v_temp = get_vector(current_pos, n[i])
               cos_theta = np.inner(vector_sum, v_temp) /\
                           (np.linalg.norm(vector_sum, 2) *\
                            np.linalg.norm(v_temp, 2))
               cos2next[cos_theta] = n[i]
            keys_list = list(cos2next.keys())
            keys_list.sort()
            result = PLACE_HOLDER_EDGE
            new_trail = trail[:]
            temp = 0
            while result == PLACE_HOLDER_EDGE and temp < len(keys_list):
               result, new_trail = get_edge(image_bin,\
                                             cos2next[keys_list[-1 - temp]],\
                                             trail, known, nodes_center,\
                                             starting_index, radius, is_joint,\
                                             image_debug)
               temp += 1
            return result, new_trail
         
         
#                              End of Subsection                              #
#=============================================================================#
#                                  Method 2                                   #

def find_vertices(image_bin, endpoints, intersections_skirt_all, v2v, known,\
                  current, starting):
   """Finds the other vertices which the one being studied directly links to
   without intersecting any other edge.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   intersections_skirt_all: List[(int, int)]
      Stores all the pixels adjacent to the intersection points.
   v2v : {int : List[int]}
      Each key is the index of a vertex, each value is a list indices of other
      vertices. This is the output dictionary.
   known : List[List[int, int]]
      Stores explored points.
   current : List[int, int]
      Stores the coordinates of the current point.
   starting : int
      The index of the vertex which the process is started with.
   Returns
   -------
   By reference: v2v
   """
   temp = -1
   for i in range(len(endpoints)):
      if current in endpoints[i] and i != starting and not i in v2v[starting]:
         temp = i
         v2v[starting].append(i)
   if temp == -1 and not current in intersections_skirt_all:
      known.append(current)
      n = get_neighborhood(image_bin, current)
      nv = get_neighborhood_values(image_bin, current)
      for i in range(1, len(n)):
         if nv[i] == 1 and not n[i] in known:
            find_vertices(image_bin, endpoints, intersections_skirt_all, v2v,\
                           known, n[i], starting)

def find_intersections(image_bin, end_point):
   """Finds all the intersection points in the image. An intersection is
   defined as a point whose 8-point neighborhood has at least 3 meaningful
   points.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   end_point : List[int, int]
      This is a starting point of an edge.
   Returns
   -------
   intersections : List[List[int, int]]
      A list of intersection points. Note that the index of each intersection
      in this list is unique across the program.
   intersections_skirt : List[List[int, int]]
      A list of points in the 8-point neighborhood of each intersection. This
      list maybe useful in other functions.
   """
   intersections = []
   image_debug = image_bin.copy()
   known = [end_point]
   n = get_neighborhood(image_bin, end_point)
   nv = get_neighborhood_values(image_bin, end_point)
   prev = [end_point]
   next2prev = []
   next_level = []
   for i in range(1, len(n)):
      if nv[i] == 1:
         next_level.append(n[i])
         next2prev.append(0)
         
   # Iteratively examines all the content pixels and records all the
   # intersections.
   while len(next_level) > 0:
      # print("prev = " + str(prev))
      # print("next2prev = " + str(next2prev))
      # print("next_level = " + str(next_level))
      # print('')
      next_temp = []
      next2prev_temp = []
      for i in range(len(next_level)):
         pos = next_level[i]
         known.append(pos)
         n = get_neighborhood(image_bin, pos)
         nv = get_neighborhood_values(image_bin, pos)
         unknown = []
         for j in range(1, len(n)):
            if nv[j] == 1:
               if not n[j] in known and not n[j] in next_temp:
                  next_temp.append(n[j])
                  next2prev_temp.append(i)
               if get_distance(n[j], prev[next2prev[i]]) > 1:
                  unknown.append(n[j])
         if is_intersection(pos, unknown) and not pos in intersections:
            '''
            print(pos)
            print(prev[next2prev[i]])
            print_neighborhood_values(nv)
            print('')
            '''
            intersections.append(pos)
            # cv2.circle(image_debug, (pos[0], pos[1]), 4, 1, 1)
            # image_show = get_binary_image(image_debug, 0, 255)
            # cv2.imshow("image_debug", image_show)
            # cv2.waitKey(1)
      next2prev = next2prev_temp
      prev = next_level
      next_level = next_temp
   '''
   image_temp = np.zeros([image_bin.shape[0], image_bin.shape[1]], np.uint8)
   for k in known:
      image_temp[k[1], k[0]] = 1
   image_temp = get_binary_image(image_temp, 0, 255)
   cv2.imshow("image_temp", image_temp)
   cv2.waitKey(1)
   '''
   intersections_skirt = []
   for i in range(len(intersections)):
      intersections_skirt.append(get_neighborhood(image_bin, intersections[i]))
   return intersections, intersections_skirt

def construct_network(image_bin, endpoints, intersections,\
                           intersections_skirt, nodes_center):
   """Builds the network of the intersections and records the links between
   vertices and some intersections.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   intersections : List[List[int, int]]
      A list of intersection points.
   intersections_skirt : List[List[int, int]]
      A list of points in the 8-point neighborhood of each intersection.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices. This parameter
      is useful only in showing the intermediate outputs for debugging.
   Returns
   -------
   linked_to : List[List[int, int]]
      Stores a list of destinations each intersection links to. A positive
      is the index of some other intersection while a negative value is the
      shifted index of a vertex (-1 means 0, -2 means 1, -3 means 2, etc.).
   outgoing : List[List[List[double, double]]]
      Stores a list of outgoing weighted vectors for each intersection.
   v2i : {int : List[int]}
      Each key is the shifted index of a vertex while each value is a list of
      indices of the intersections the corresponding vertex links to.
   """
   linked_to = []
   outgoing = []
   v2i = {}
   for i in range(len(endpoints)):
      v2i[-i - 1] = []
   '''
   for i in range(len(intersections)):
      print("intersections: " + str(intersections[i]))
      print_neighborhood_values(get_neighborhood_values(image_bin, intersections[i]))
      print('')
   '''
   for i in range(len(intersections)):
      n = intersections_skirt[i]
      nv =  get_neighborhood_values(image_bin, intersections[i])
      linked_to.append([])
      outgoing.append([])
      for j in range(1, len(n)):
         if nv[j] == 1:
            find_path(image_bin, n[j], [intersections[i], n[j]],\
                     [intersections[i]], endpoints, intersections,\
                     intersections_skirt, linked_to, outgoing, i, v2i,\
                     nodes_center)
      print(linked_to)
      print(outgoing)
      print('')
   print("v2i = " + str(v2i))
   draw_vectors(image_bin, intersections, outgoing);
   return linked_to, outgoing, v2i
      
def draw_vectors(image_bin, starting_points, vectors):
   """Draws vectors on a given binary image.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   starting_points : List[(int, int)]
      Stores the starting point of each vector.
   vectors : List[List[(int, int)]]
      Stores the vectors at each starting point.
   Returns
   -------
   None
   """
   image_temp = image_bin.copy();
   for i in range(len(starting_points)):
      for j in range(len(vectors[i])):
         cv2.arrowedLine(image_temp, (starting_points[i][0], starting_points[i][1]),\
                           (int(np.ceil(starting_points[i][0] + 10 * vectors[i][j][0])), int(np.ceil(starting_points[i][1] + 10 * vectors[i][j][1]))), 255)
   cv2.imshow("vectors", image_temp)
   cv2.waitKey(1)

def find_path(image_bin, current_pos, trail, known, endpoints, intersections,\
               intersections_skirt, linked_to, outgoing, starting_intersection,\
               v2i, nodes_center):
   """Explores an outgoing path starting from an intersection, records when a
   destination is reached.
   Parameters:
   -----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   current_pos : List[int, int]
      The current position.
   trail : List[List[int, int]]
      A list of points that have been explored in the current branch.
   known : List[List[int, int]]
      A list of points that have been explored so far.
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   intersections : List[List[int, int]]
      A list of intersection points. 
   linked_to : List[List[int, int]]
      Stores a list of destinations each intersection links to. A positive
      is the index of some other intersection while a negative value is the
      shifted index of a vertex (-1 means 0, -2 means 1, -3 means 2, etc.).
   outgoing : List[List[List[double, double]]]
      Stores a list of outgoing weighted vectors for each intersection.
   starting_intersection : int
      The index of the intersection which the iteration starts from.
   """
   known.append(current_pos)
   node_index = -1
   for m in range(len(endpoints)):
      if current_pos in endpoints[m]:
         node_index = m
   intersection_index = -1
   for m in range(len(intersections_skirt)):
      if m != starting_intersection and current_pos in intersections_skirt[m]:
         intersection_index = m
   if node_index != -1:
      if not (-1 * (node_index + 1)) in linked_to[starting_intersection]:
         linked_to[starting_intersection].append(-1 * (node_index + 1))
         vector_list = []
         for i in range(1, len(trail)):
            vector_list.append(get_vector(trail[i - 1], trail[i]))
         outgoing[starting_intersection].append(get_vector_sum_out(vector_list))
         v2i[-1 * (node_index + 1)].append(starting_intersection)
         '''
         image_temp = np.zeros([image_bin.shape[0], image_bin.shape[1]], np.uint8)
         for k in known:
            image_temp[k[1], k[0]] = 1
         image_temp = get_binary_image(image_temp, 0, 255)
         cv2.circle(image_temp, (intersections[starting_intersection][0],\
                     intersections[starting_intersection][1]), 5, 255, 1)
         #cv2.putText(image_temp, str(node_index), nodes_center[node_index], cv2.FONT_HERSHEY_SIMPLEX,\
         #   FONT_SIZE, 255, FONT_THICKNESS, cv2.LINE_AA, False)
         for c in nodes_center:
            cv2.circle(image_temp, (c[0], c[1]), 5, 255, 1)
            cv2.putText(image_temp, str(node_index), nodes_center[node_index], cv2.FONT_HERSHEY_SIMPLEX,\
               FONT_SIZE, 255, FONT_THICKNESS, cv2.LINE_AA, False)
         cv2.imshow("image_temp", image_temp)
         cv2.waitKey()
         '''
   elif intersection_index != -1:
      if not intersection_index in linked_to[starting_intersection]:
         linked_to[starting_intersection].append(intersection_index)
         vector_list = []
         for i in range(1, len(trail)):
            vector_list.append(get_vector(trail[i - 1], trail[i]))
         outgoing[starting_intersection].append(get_vector_sum_out(vector_list))
         '''
         image_temp = np.zeros([image_bin.shape[0], image_bin.shape[1]], np.uint8)
         for k in known:
            image_temp[k[1], k[0]] = 1
         image_temp = get_binary_image(image_temp, 0, 255)
         cv2.circle(image_temp, (intersections[starting_intersection][0],\
                     intersections[starting_intersection][1]), 5, 255, 1)
         cv2.putText(image_temp, str(intersection_index),\
                     (intersections[intersection_index][0],\
                     intersections[intersection_index][1]), cv2.FONT_HERSHEY_SIMPLEX,\
            FONT_SIZE, 255, FONT_THICKNESS, cv2.LINE_AA, False)
         for c in nodes_center:
            cv2.circle(image_temp, (c[0], c[1]), 5, 255, 1)
            cv2.putText(image_temp, str(node_index), nodes_center[node_index], cv2.FONT_HERSHEY_SIMPLEX,\
               FONT_SIZE, 255, FONT_THICKNESS, cv2.LINE_AA, False)
         cv2.imshow("image_temp", image_temp)
         cv2.waitKey()
         '''
   else:
      n = get_neighborhood(image_bin, current_pos)
      nv = get_neighborhood_values(image_bin, current_pos)

      for i in range(1, len(n)):
         if nv[i] == 1 and not n[i] in known:
            find_path(image_bin, n[i], trail + [n[i]], known, endpoints,\
                        intersections, intersections_skirt, linked_to,\
                        outgoing, starting_intersection, v2i, nodes_center)
            '''
            temp = -1
            for j in range(len(endpoints)):
               if n[i] in endpoints[j]:
                  temp = j
            if temp != -1 or n[i] in intersections:
               break
            '''
            
def merge_intersections(image_bin, intersections, linked_to, outgoing):
   """Merges the intersections so that, after the process, there only remain
   X-shape intersections and all the Y-shape ones are gone.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   intersections : List[List[int, int]]
      A list of intersection points.
   linked_to : List[List[int, int]]
      Stores a list of destinations each intersection links to.
   outgoing : List[List[List[double, double]]]
      Stores a list of outgoing weighted vectors for each intersection.
   Returns
   -------
   merged : List[List[int, int]]
      Stores a list of destinations each intersection links to. This is a
      trimmed version of linked_to.
   bridges : List[(int, int)]
      Stores a list of tuples representing a link between two intersectoins.
   """
   unmerged = [i for i in range(len(intersections))]
   print("linked_to = " + str(linked_to))
   print("outgoing = " + str(outgoing))
   candidates = []
   bridges = []
   merged = deepcopy(linked_to)
   get_merging_options(unmerged, [], candidates, linked_to)
   print("candidates = " + str(candidates))
   print('')
   smallest = sys.maxsize
   for i in range(len(candidates)):
      linked_to_copy = deepcopy(linked_to)
      outgoing_copy = deepcopy(outgoing)
      temp = 0
      for pair in candidates[i]:
         
         temp += merge_2_intersections(linked_to_copy, outgoing_copy, pair)
         '''
         print("pair = " + str(pair))
         print("linked_to_copy = " + str(linked_to_copy))
         print("outgoing_copy = " + str(outgoing_copy))
         print("temp = " + str(temp))
         print('')
         '''
      if smallest > temp:
         merged = deepcopy(linked_to_copy)
         bridges = candidates[i]
         smallest = temp
   '''
   print("smallest = " + str(smallest))
   print("merged = " + str(merged))
   print("bridges = " + str(bridges))
   '''
   return merged, bridges
   
def get_merging_options(unmerged, current_list, result, linked_to):
   """Find all the valid merging options. A valid way to merge should not
   result in set of isolated intersections at the end.
   Parameters
   ----------
   unmerged : List[int]
      A list of indices of intersections which have not been paired with any
      other intersection yet.
   current_list : List[(int, int)]
      A list of index pairs indicating that the two intersections are paired.
   result : List[List[(int, int)]]
      A list of lists of index pairs. Each list is a valid output of a valid
      merging.
   linked_to : List[List[int, int]]
      Stores a list of destinations each intersection links to.
   """
   print("unmerged = " + str(unmerged))
   if len(unmerged) == 0:
      result.append(deepcopy(current_list))
   else:
      current = unmerged[0]
      for i in linked_to[current]:
         if i > 0 and i in unmerged and len(linked_to[current]) == len(linked_to[i]):
            current_list.append((current, i))
            temp = unmerged[1 :]
            temp.remove(i)
            get_merging_options(temp, current_list, result, linked_to)
            current_list.pop()

def merge_2_intersections(linked_to_copy, outgoing_copy, indices):
   """Merges two intersections A and B by removing index(B) in
   linked_to_copy[A] and index(A) in linked_to_copy[B] and do the similar work
   on outgoing_copy. This method assumes that the two intersections have the
   same amount of outgoing vectors.
   Parameters
   ----------
   linked_to_copy : List[List[int, int]]
      Stores a list of destinations each intersection links to.
   outgoing_copy : List[List[List[double, double]]]
      Stores a list of outgoing weighted vectors for each intersection.
   indices : (int, int)
      Stores the indices of the intersections pending to merge.
   Returns
   -------
   smallest : double
      The minimum value of error of this merge.
   """
   del outgoing_copy[indices[0]][linked_to_copy[indices[0]].index(indices[1])]
   del outgoing_copy[indices[1]][linked_to_copy[indices[1]].index(indices[0])]
   linked_to_copy[indices[0]].remove(indices[1])
   linked_to_copy[indices[1]].remove(indices[0])
   smallest = sys.maxsize
   n = len(outgoing_copy[indices[0]])
   '''
   for i in range(n):
      sum = 0
      for j in range(n):
         sum += np.inner(outgoing_copy[indices[0]][j], \
                           outgoing_copy[indices[1]][(i + j) % n]) /\
             (np.linalg.norm(outgoing_copy[indices[0]][j], 2) *\
             np.linalg.norm(outgoing_copy[indices[1]][(i + j) % n], 2))
      smallest = min(smallest, sum)
   '''
   minimum = get_min_score(outgoing_copy[indices[0]], outgoing_copy[indices[1]], [])
   return minimum

def get_min_score(vec_list_static, vec_list_dynamic, current):
   """Computes the minimum score out of all the possible pairing options
   between the two given vector lists. Note that only one vector list needs to
   be permuted in order to obtain all the possible bijections.
   Parameters
   ----------
   vec_list_static : List[List[int]]
      This vector list does not change at each iteration,
   vec_list_dynamic : List[List[int]]
      This vector list changed at each iteration.
   current : List[List[int]]
      This list stores the current permutation of a vector list.
   Returns
   -------
   result : double
      The minium score (related to the best bijection) of the current match.
   """
   if len(vec_list_dynamic) == 0:
      temp = 0
      for i in range(len(current)):
         temp += get_score(vec_list_static[i], current[i], -1)
      return temp
   else:
      result = sys.maxsize;
      for i in range(len(vec_list_dynamic)):
         current.append(vec_list_dynamic[i])
         result = min(result, get_min_score(vec_list_static, vec_list_dynamic[0 : i] + vec_list_dynamic[i + 1 :], current))
         current.pop()
      return result

def restore_graph(v2i, linked_to, outgoing, bridges_array):
   """Retrieves the graph data.
   Parameters
   ----------
   v2i : {int : List[int]}
      v2i : {int : List[int]}
      Each key is the shifted index of a vertex while each value is a list of
      indices of the intersections the corresponding vertex links to.
   linked_to : List[List[int, int]]
      Stores a list of destinations each intersection links to.
   outgoing : List[List[List[double, double]]]
      Stores a list of outgoing weighted vectors for each intersection.
   bridges_array : List[int]
      Stores the bridges in a list/array form.
   Returns
   -------
   viv : {int : List[int]}
      Each key is the 0-base index of a vertex and each value is a list of
      0-base index of vertices being linked to.
   """
   viv = {}
   for i in range(len(v2i)):
      viv[i] = []
      prev = [-i - 1] * len(v2i[-i - 1])
      next = v2i[-i - 1]
      while len(next) != 0:
         '''
         print("prev = " + str(prev))
         print("next = " + str(next))
         print('')
         '''
         current = next
         prev_temp = []
         next = []
         for j in range(len(current)):
            des = current[j]
            if des < 0 and des != -i - 1:
               viv[i].append(-des - 1)
            elif des >= 0:
               vec1_index = linked_to[des].index(prev[j])
               vec1 = outgoing[des][vec1_index]
               next_des = len(linked_to)
               minimum = 1
               for k in range(len(outgoing[bridges_array[des]])):
                  vec2 = outgoing[bridges_array[des]][k]
                  cos_theta = np.inner(vec1, vec2) / (np.linalg.norm(vec1, 2) *\
                              np.linalg.norm(vec2, 2))
                  if cos_theta < minimum:
                     minimum = cos_theta
                     next_des = linked_to[bridges_array[des]][k]
               prev_temp.append(bridges_array[des])
               current[j] = bridges_array[des]
               next.append(next_des)
         prev = prev_temp
   return viv
      

#                               End of Section                                #
###############################################################################