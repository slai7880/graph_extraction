# The path of input images. By default it is an empty string, so the images are
# in the same folder as the program.
GRAPH_PATH = ''
TEMPLATE_PATH = ''

# The text that is displayed at the beginning of the output.
GREETING = "Welcome! Please provide the input filename:"

# The input indicating that the step is complete, only used to test uesr input.
DONE = 'done'

# The replacing element for a false node in the list.
PLACE_HOLDER = (-1, -1)

# The relative position of the label of each node when displaying them. Alternatively
# if REL_POS = 'MIDDLE' is active, then the label will be placed around the middle of the frames.
# REL_POS = (25, 25)
REL_POS = 'MIDDLE'

# 1. Sort vertices one by one.
# 2. Sort them all at once.
SORT = 1

# The starting index of the vertices.
INDEX_START = 0

# When extracting the edges, nodes must be blocked, this number affects the size of each block,
# so it may also affect the accuracy of the process.
SCALE_PERCENTAGE = 0.6

# This constant is the threshold factor of the ditance between an end point of an edge and the
# center of a node, a value greater than 1 is recommended.
TOLERANCE_FACTOR = 1.5

# Determines how many edges will be displayed in a group when deleting the false ones.
EDGES_PER_GROUP = 10