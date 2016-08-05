'''
thinning.py
Sha Lai
8/4/2016

This sub-program implements zhang-seun's image thinning algorithm.

'''
import numpy
import sys

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
            p = (int(mat[i, j]), int(mat[i - 1, j]), int(mat[i - 1, j + 1]), int(mat[i, j + 1]), \
                  int(mat[i + 1, j + 1]), int(mat[i + 1, j]), int(mat[i + 1, j - 1]), \
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

# Takes a binay image as input, assuming that the contents are represented as 1s
# while the rest are 0s, thins the contents.
def thinning(image):
   mat = image.copy()
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
   return mat