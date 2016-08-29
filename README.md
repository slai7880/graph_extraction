This project aims at helping the user to extract a mathematical graph from a digital image using python. A graph, usually denoted as *G = (V, E)*, is widely used across various fields. There are multiple ways to store a graph in a computer(for example, a list). Meanwhile there are many applications that are capable of presenting a graph in a picture in a fairly clear way. However I have not yet found any published software that can do the reverse work, that is, recognize a graph from an image and store the data inside. Due to the variety of graph images, I myself cannot come up with a robust idea to allow the computer to automatically extract a graph from any type of images. Nevertheless I design this python project to interact with the user and help them do this work.

The programs are written in Python 3.5, along with additional packages including OpenCV, NumPy, SciPy, and PyQT which is used to create a better GUI. The file graph_extraction.py contains all the main functions implementing the algorithms I use, hence it also contains a detailed description at the beginning. The file main.py and graph_extraction_ui.py serve as client functions. The former interacts with the user in the console while the later creates a GUI window to do the same work. In addition the common.py file contains all the constants for all the other files.

The description of the ideas behind is put in graph_extraction.py. Here I want to provide an introduction about how to use my programs.

There are two executable files in the directory. One is main.py and the other is graph_extraction_ui.py. The later is merely a variation with a GUI of the former and it is still under development. So I will focus on describing how to use the first one.

First of all I do not recommend the user to execute main.py in any IDE. The reason is that my I am using OpenCV to display images, but it appears that the IDEs do not like the windows popped by this third-party package, causing some weird issue when displaying. So for a better display, use the console to execute the codes.

To begin, let's break the entire procedure into a few steps.

1. Initialization

    Before executing this program, the user must put the graph image in the directory called graph_input. And then they must crop an example of the vertices from the original image and put it in the folder vertex_example. Although the user is allowed to modify the file common where a number of constants are defined, do so at your own risk!

    After execution, the program first reads all the files in the input directory
for graph images. It lists all the files out and asks the user to indicate the
desired one by index. Same for selecting the vertex example.

2. Locating Vertices

    Next the program will ask the user to provide the number of vertices they want from the image. This number may not be accurate, and the request will be repeated until the user enter 0, implying that there is no vertex left not found. Due to the accuracy restriction of template matching which is the main technique being used here, there is no guarantee that all the detected vertices are true ones. However we still ask the user to keep staying at this step till all the vertices are marked. Depending on the input images, there may be some vertices
in the image remain unmarked no matter how many times the user repeats the finding procedure. In this case, I apologize that my program cannot handle the given image.

    After the user enter 0, the program will label all the vertices on the image, and asks the user to give a sequence of indices of false vertices. The input must be a sequence of valid integers separated by space, however a single integer is allowed as well.

    Next the user will be asked if they want to correct the order of the vertices. The user may answer yes if they want the indices to match the original labels on the image. There are two ways to correct. The first method, one-by-one, allows the user to correct the labels one by one as the name implies; the other method, once-for-all, allows the user to provide a sequence of correct labels, for example, let v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub>, ..., v<sub>n</sub> be a list of detect vertices with some labels assigned during the finding step, the user may enter a sequence of integers like 2, 1, 4, 6, ... to provide the correct indices for each vertex. Apparently the second method is better if the graph size is rather small.

3. Detecting Edges

    The program should now ask the user if they want to thin the image. This technique aims to improve the accuracy for some images with a price of wait time but for now I haven't found it necessary on the example images in the input directory. Next the program will start to process the image. When it completes, it will again display all the detected edges with labels and ask the user to provide a sequence of indices indicating the false edges just like the find-vertices step.

At the end a list of edges will be printed to the console.

Minor Notes:

1. When interacting with the user, there is a chance that the user may enter some invalid input, some are dangerous enough to break the entire program. To handle this type of issue, I write some codes to check the inputs and in most of the cases the program will keep asking the user till an acceptable one is given.

2. There is no undo or redo here, as much as it sounds unreasonable in an image processing program.