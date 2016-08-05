This project aims at helping the user to extract a mathematical graph from a digital image using python. A graph, usually denoted as G = (V, E), is widely used across various fields. There are multiple ways to store a graph in a computer, for example, a list. Meanwhile there are many applications that are capable of presenting a graph in a picture in a fairly clear way. However I have not yet found any published software that can do the reverse work, that is, recognize a graph from an image and store the data inside. Due to the variety of graph images, I myself cannot come up with a robust idea to allow the computer to automatically extract a graph from any type of images. Nevertheless I degisn this python project to interact with the user and help them do this work.

The main tool which I use to manipulate images is OpenCV and the initial program is written with python 3.

The algorithm, developed under the help from Adrian Rosebrock at PyImageRedearch,  is rather simple, and the steps are listed below:
1. Locate the vertices.

2. Find the edges.

The main approaches at each step are illustrated below.

1. Locating the Vertices

	The main idea is multi-scale template matching(http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/). The user must crop the image for an example of the vertices before executing the program. When the program accepts the image as well as the template, it will attempt to create a window on the image and slide through the entire image, trying to find a piece that is the most similar to the given template. Due to the limit of this algorithm, it cannot be guaranteed that all the vertices are correctly identified. Therefore the program will keep asking the user to provide the number of remaining unlocated vertices on the image untill all are found, and then allow the user to remove all the false vertices.

	It is possible that in the original image all the vertices are labeled with a sequence of numbers. Recognizing text is another deep topic in computer vision, so I did not implement any feature that can do this job for the user. Instead, for now I can only ask the user to correct the order of the vertices.

2. Detecting Edges

	Although it is suggested to use canny edge detector followed by contour extraction function to obtain the edges, the vertices might also be recognized as part of the edges if the two approaches are directly applied. Once the location of each vertex is obtained , we can first create a binary copy of the original image, and put a box filled with the background color at each vertex so that the vertices are hiddden and all the edges are separated from the vertices. The findContours function will then detect all the separated line segment left on the image, which forms a set of edges.

	Note that the fincContours function only returns a list of "contours", that is, a list of pixels that surround each line segment. Meanwhile, fortunately the function tends to put the pixel near the corner as the first element in each returned list, we then can approximate the end points of each line segment and find out which vertices the endpoints are likely to connect. In a more detailed way of saying, each list of pixels of contours will be examined and the program will attempt to get the end point of the corresponding edge, at last the program will examine all of these endpoints and determine if one is close enough to any vertex.


The above procedure suggested by Adrian does work, however the last part is problematic. The experiments show that the way we find which vertices is an edge connecting is not accurate enough. The problem appear to occur at marking the endpoints part. Since the line segments are not "lines" in mathematics, they have width formed by pixels, which turns out to be affecting the accuracy of determining the pixels that mark the two endpoints of a segment. Hence I introduce an algorthm that can improve the existing one: image thinning. By thinning the image using zhang-suen algorithm after blocking the vertices, all the remaining line segments are skeletons of the original ones, which reduces the "width" to merely one pixel. Due to the limit of my knowledge I cannot provode a further discussion about the underlying reason but it turns out that this approach dramastically improves the accuracy of the program.