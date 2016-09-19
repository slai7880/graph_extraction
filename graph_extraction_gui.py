'''
graph_extraction_ui.py
Sha Lai
8/30/2016

This program does the same job as the original program does but interacts with
the user via a GUI instead.
'''

import sys
import numpy as np
import imutils
import cv2
from common import *
from graph_extraction import *
from math import sqrt, inf, fabs
from os import listdir
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# This class helps redirecting stdout to the text area.
class EmittingStream(QObject):
   _stdout = None
   textWritten = pyqtSignal(str)
   
   def flush(self):
      pass
      
   def fileno(self):
      return -1
      
   def write(self, text):
      #if (not self.signalsBlocked):
         #self.textWritten.emit(str(text))
      self.textWritten.emit(str(text))

class Window(QWidget):
   def __init__(self):
      super(Window, self).__init__()
      sys.stdout = EmittingStream(textWritten = self.outputText)
      self.input = ''
      self.state = 0
      self.initUI()
   
   # Initiates the GUI with merely an output and an input console.
   def initUI(self):
      # set up the basic window
      self.window = QWidget()
      self.window.setWindowTitle("Graph Extraction GUI")
      
      # create image area
      self.label = QLabel()
      
      # create a text area for outputs
      self.outputConsole = QTextEdit(self)
      self.outputConsole.setReadOnly(True)
      pal = self.outputConsole.palette()
      pal.setColor(QPalette.Base, QColor(255, 255, 255))
      self.outputConsole.setPalette(pal)
      self.outputConsole.setStyleSheet("QTextEdit {color:black}")
      font = self.outputConsole.font()
      font.setPointSize(OUTPUT_FONT)
      self.outputConsole.setFont(font)
      self.outputConsole.setFixedSize(400, OUTPUT_CONSOLE_HEIGHT)
      
      # display the initial message
      print("Welcome!")
      self.dir_path = GRAPH_PATH
      self.input_dir = listdir(self.dir_path)
      print("Files in the input directory:")
      print_list(self.input_dir)
      print("Please provide the file by index of the graph: ")
      
      
      # create a text area for input
      self.inputConsole = QLineEdit(self)
      self.inputConsole.returnPressed.connect(self.onChanged)
      
      
      # set up the layout
      grid = QGridLayout()
      grid.addWidget(self.label, 1, 1)
      grid.setAlignment(self.label, Qt.AlignHCenter)
      
      vBox = QVBoxLayout()
      vBox.addLayout(grid)
      vBox.addWidget(self.outputConsole)
      vBox.addWidget(self.inputConsole)
      
      self.window.setLayout(vBox)
      
      # display the window
      self.window.show()
   
   # Takes an openCV image as a parameter, shrinks the image if necessary, and
   # then displays the image to the target area in the window.
   def loadImage(self, cvImage):
      h, w, channel = cvImage.shape
      bytesPerLine = 3 * w
      self.pixmap = QPixmap(QImage(cvImage.data, w, h, bytesPerLine,\
                              QImage.Format_RGB888))
      if w >= GRAPH_SIZE_MAX[0]:
         self.pixmap = self.pixmap.scaledToWidth(GRAPH_SIZE_MAX[0],\
                                                   Qt.KeepAspectRatio)
      elif h >= GRAPH_SIZE_MAX[1]:
         self.pixmap = self.pixmap.scaledToHeight(GRAPH_SIZE_MAX[1],\
                                                   Qt.KeepAspectRatio)
      self.label.setPixmap(self.pixmap)
      # may need to resize the output console
      self.outputConsole.setFixedSize(self.pixmap.size().width(),\
                                       OUTPUT_CONSOLE_HEIGHT)
      
   # Takes an openCV image in gray scale as a parameter, shrinks the image if
   # necessary, and then displays the image to the target area in the window.
   def loadImageGray(self, cvImage_gray):
      h, w = cvImage.shape
      bytesPerLine = 3 * w
      self.pixmap = QPixmap(QImage(cvImage.data, w, h, bytesPerLine,\
                              QImage.Format_RGB888))
      if w >= GRAPH_SIZE_MAX[0]:
         self.pixmap = self.pixmap.scaledToWidth(GRAPH_SIZE_MAX[0],\
                                                   Qt.KeepAspectRatio)
      elif h >= GRAPH_SIZE_MAX[1]:
         self.pixmap = self.pixmap.scaledToHeight(GRAPH_SIZE_MAX[1],\
                                                   Qt.KeepAspectRatio)
      self.label.setPixmap(self.pixmap)
      # may need to resize the output console
      self.outputConsole.setFixedSize(self.pixmap.size().width(),\
                                       OUTPUT_CONSOLE_HEIGHT)
   
   # Writes the passed in text to the output console.
   def outputText(self, text):
      cursor = self.outputConsole.textCursor()
      cursor.movePosition(QTextCursor.End)
      cursor.insertText(text)
      self.outputConsole.setTextCursor(cursor)
   
   # Examines the input, and performs the corresponding work. This function
   # operates based on state. Each state is likely to consist of three
   # sub-states: takes user input, perform corresponding work, and then provide
   # the starting information of the next state.
   def onChanged(self):
      self.input = self.inputConsole.text()
      self.inputConsole.clear()
      print(self.input)
      
      # get the graph
      if self.state == 0:
         self.graph, self.graph_gray = self.get_image(self.input, "graph")
         if not self.graph is None and not self.graph_gray is None:
            self.state += 1
            self.break_point = get_threshold(self.graph_gray)
            self.loadImage(self.graph)
            
            # ask the user for the template
            self.dir_path = TEMPLATE_PATH
            self.input_dir = listdir(self.dir_path)
            print("Files in the input directory:")
            print_list(self.input_dir)
            print("Please provide the file by index of the template: ")
      
      # get the template
      elif self.state == 1:
         self.template, self.template_gray = self.get_image(self.input, "template")
         if not self.template is None and not self.template_gray is None:
            self.state = 2
            self.template, (self.tH, self.tW), self.radius =\
                  process_template(self.template)
            
            # start to find the vertices
            self.nodes = []
            self.nodes_center = []
            self.graph_display = self.graph.copy()
            self.graph_work = self.graph_gray.copy()
            print("How many vertices are you looking for?(0 means done) ")
            
      # whlie looking for the vertices, keep asking the user for more
      elif self.state == 2:  
         if self.input == '0':
            self.state = 2.5
            
            # start to ask the user to sort the vertices
            print("Please indicate non-vertex elements in the list in a " +
                  "sequence of indices or \"done\" to proceed to next step:")
         else:
            if not is_valid_type(self.input, int, "Please provide an integer!"):
               pass
            elif int(self.input) < 0:
               print("Please provide a positive integer!")
            else:
               
               locate_vertices(int(self.input), self.graph_work, self.template,\
                              self.tW, self.tH, self.nodes)
               print("Current vertices:")
               print_list(self.nodes)
               self.update_display(True)
               print("How many vertices are you looking for?(0 means done) ")
      
      # remove the false vertices
      elif self.state == 2.5:
         if self.input == DONE:
            self.state = 3
            print("Do you want to sort the vertices?(y/n)")
         else:
            indices = self.input.split()
            valid = True
            for i in indices:
               if not is_valid_type(i, int, "Invalid input detected!"):
                  valid = False
               elif int(i) < BASE or int(i) >= BASE + len(self.nodes):
                  print("Error: index out of bound!\n")
                  valid = False
            if valid == True:
               self.nodes = remove_indices(self.input, self.nodes)
               self.update_display(True)
               print("Current vertices:")
               print_list(self.nodes)
               print("Please indicate non-vertex elements in the list in a " +
                     "sequence of indices or \"done\" to proceed to next step:")
      
      # ask the user if they want to sort the vertices
      elif self.state == 3:
         if self.input[0] == 'y' or self.input[0] == 'Y':
            self.state = 3.5
            
            # ask the user for the method
            print("Please indicate the method by index you want to help" +
                  " sorting:")
            print("1. One-by-one,")
            print("2. Once-for-all.")
         elif self.input[0] == 'n' or self.input[0] == 'N':
            self.end_sorting()
      
      # the user wants to sort
      elif self.state == 3.5:
         self.index_list = []
         if self.input == '1': # one-by-one
            self.state = 3.51
            self.current_index = 0
            print("What's the correct index value of the vertex " +
                  str(self.current_index + BASE) + ". " +
                  str(self.nodes[self.current_index]) + "?")
         elif self.input == '2': # once-for-all
            self.state = 3.52
            print("Please provide a sequence of correct indices for each " +
                  "vertex or \"done\" to proceed to next step:")
         else:
            print("Invalid input, please try again!")
         
      # the user wants to sort and chooses method 1
      elif self.state == 3.51:
         index = self.input
         if is_valid_type(index, int, "Please provide a valid integer."):
            index = int(index)
            if index < BASE or index >= BASE + len(self.nodes):
               print("Error: index out of bound!\n")
            elif index in self.index_list:
               print("Duplicate index detected, please provide another one.")
            else:
               self.index_list.append(index)
               self.current_index += 1
            if len(self.index_list) == len(self.nodes): # when done sorting
               self.end_sorting()
            else:
               print("What's the correct index value of the vertex " +
                     str(self.current_index + BASE) + ". " +
                     str(self.nodes[self.current_index]) + "?")
      # the user wants to sort and chooses method 2
      elif self.state == 3.52:
         indices = self.input
         try:
            indices = indices.split()
            if not len(indices) == len(self.nodes):
               print("Not enough integers or too many of them, please try again.")
            else:
               for i in range(len(indices)):
                  if int(indices[i]) in self.index_list:
                     print("Duplicate index detected, please provide another one.")
                     break
                  else:
                     self.index_list.append(int(indices[i]))
               if len(self.index_list) == len(self.nodes):
                  if not max(self.index_list) + 1 - BASE == len(self.index_list):
                     print("The given input is not a valid arithmetic sequence!")
                  else:
                     self.end_sorting()
         except:
            print("Please provide a sequence of valid integers.")
      
      # the user can remove the false edges
      elif self.state == 4:
         if self.input == DONE:
            self.state = 5
            self.update_display(True, True)
            print("Number of edges detected: " + str(len(self.E)))
            print("Edges:")
            print(self.E)
         else:
            indices = self.input.split()
            valid = True
            for i in indices:
               if not is_valid_type(i, int, "Invalid input detected!"):
                  valid = False
               elif int(i) < BASE or int(i) >= BASE + len(self.eddes):
                  print("Error: index out of bound!\n")
                  valid = False
            if valid == True:
               self.edges = remove_indices(self.input, self.edges)
               self.update_display(True, True)
               print("Edges:")
               print(self.E)
               print("Please indicate non-edge elements in the list in a " +
                     "sequence of indices or \"done\" to proceed to next step:")
               
   # Takes two strings as parameters, where response is the user input and
   # keyword is either the directory path of graph image files or that of
   # template image files, opens the seleted file as an opencv image and
   # returns it along with its gray-scale version.
   def get_image(self, response, keyword):
      image = None
      image_gray = None
      if is_valid_type(response, int, "Please provide an integer!"):
         index = int(response)
         if index >= 0 + BASE and index < len(self.input_dir) + BASE:
            try:
               image = cv2.imread(self.dir_path + self.input_dir[index - BASE])
               image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               valid = True
               print("Selected " + keyword + " file: " + 
                  str(self.input_dir[index - BASE]))
            except:
               print("Error: the " + keyword + " file is invalid or \
                  cannot be processed.")
               response = ''
         else:
            print("Error: index out of bound!\n")
      return image, image_gray
   
   # This function should be called either when the user chooses not to sort
   # the vertices, or when they finish doing so. It reorganizes the vertices
   # if need to, and constructs a list of center position of each vertex under
   # the new order.
   def end_sorting(self):
      if self.state > 3.5:
         result = [0] * len(self.nodes)
         for i in range(len(self.index_list)):
            result[self.index_list[i] - BASE] = self.nodes[i]
         self.nodes = result
         graph_display = self.graph.copy()
         draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
         self.loadImage(graph_display)
      self.nodes_center = get_center_pos(self.nodes, self.tW, self.tH)
      self.extract_contours()

   # Takes a boolean value indicating is the image needs to be thinned, stores
   # all the contours in the image.
   def extract_contours(self):
      self.contours = extract_contours(self.graph_gray, self.nodes_center,\
                                       self.radius, self.break_point)
      print("Number of contours detected: " + str(len(self.contours)))
      self.E, self.edges_center =\
            get_endpoints(self.contours, self.nodes_center, self.radius,\
                        int(self.input))
      print(len(self.E))
      
      # show the image
      self.update_display(False, True)
      
      # start to ask the user to remove false edges
      print("Please indicate non-edge elements in the list in a " +
            "sequence of indices or \"done\" to proceed to next step:")
      self.state = 4
   
   # Takes two boolean values, update_vertices and update_edges, among which
   # the later has a default value False, updates the displayed image using the
   # most recent list of vertices or that of edges.
   def update_display(self, update_vertices, update_edges = False):
      graph_display = self.graph.copy()
      if update_vertices:
         draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
      if update_edges:
         draw_edges(graph_display, self.edges_center, False)
      self.loadImage(graph_display)

   # Restores the stdout back to default.
   def __del__(self):
      try:
         sys.stdout = sys.__stdout__
      except:
         pass

###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":
   
   state = 0
   app = QApplication(sys.argv)
   window = Window()
   
   #window.loadImage(graph) 
   sys.exit(app.exec_()) 
   
   