import numpy as np
import random

class Geometry:
    count = 0
    def __init__(self, name = "Shape", points = None):
        self.name = name
        # name is string that is a name of gemoetry
        self.points = points
        # points is a list of tuple points = [(x0, y0), (x1, y1), ...]
        Geometry.count += 1

    def calculate_area(self):
        return 0.0

    def get_name(self):
        return self.name

    @classmethod
    def count_number_of_geometry(cls):
        # TODO: Your task is to implement the class method
        # to get the number of instance that have already created
        pass

class Triangle(Geometry):
    def __init__(self, a, b, c):
        # a, b, c are tuples that represent for 3 vertices of a triangle
        # TODO: Your task is to implement the constructor
        #super(Triangle, self).__init__(?, ?)
        pass

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        pass

class Rectangle(Geometry):
    def __init__(self, a, b):
        # a, b are tuples that represent for top and bottom vertices of a rectangle
        # TODO: Your task is to implement the constructor
        # super(Rectangle, self).__init__(?, ?)
        pass

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        pass

class Square(Rectangle):
    def __init__(self, a, length):
        # a is a tuple that represent a top vertex of a square
        # length is the side length of a square
        # TODO: Your task is to implement the constructor
        # super(Square, self).__init__(?, ?)
        pass

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        pass

class Circle(Geometry):
    def __init__(self, o, r):
        # o is a tuple that represent a centre of a circle
        # r is the radius of a circle
        # TODO: Your task is to implement the constructor
        # super(Circle, self).__init__(?, ?)
        pass

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        pass

class Polygon(Geometry):
    def __init__(self, points):
        # points is a list of tuples that represent vertices of a polygon
        # TODO: Your task is to implement the constructor
        # super(Polygon, self).__init__(?, ?)
        pass

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        pass

def test_geomery():
    ## Test cases for Problem 1

    triangle = Triangle((0, 1), (1, 0), (0, 0))
    print("Area of %s: %0.4f" % (triangle.name, triangle.calculate_area()))

    rectangle = Rectangle((0, 0), (2, 2))
    print("Area of %s: %0.4f" % (rectangle.name, rectangle.calculate_area()))

    square = Square((0, 0), 2)
    print("Area of %s: %0.4f" % (square.name, square.calculate_area()))

    circle = Circle((0, 0), 3)
    print("Area of %s: %0.4f" % (circle.name, circle.calculate_area()))

    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    print("Area of %s: %0.4f" % (polygon.name, polygon.calculate_area()))


def matrix_multiplication(A, B):
    # TODO: Your task is to required to implement
    # a matrix multiplication between A and B
    pass

def test_matrix_mul():
    ## Test cases for matrix multplication ##

    for test in range(10):
        m, n, k = random.randint(3, 10), random.randint(3, 10), random.randint(3, 10)
        A = np.random.randn(m, n)
        B = np.random.randn(n, k)
        assert np.mean(np.abs(A.dot(B) - matrix_multiplication(A, B))) <= 1e-7, "Your implmentation is wrong!"
        print("[Test Case %d]. Your implementation is correct!" % test)

def recursive_pow(A, n):
    # TODO: Your task is required implementing
    # a recursive function

    pass

def iterative_pow(A, n):
	# TODO: Your task is required implementing
    # a iterative function
    pass

def test_pow():
    ## Test cases for the pow function ##

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Recursive: A^{} = {}".format(n, recursive_pow(A, n)))

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Iterative: A^{} = {}".format(n, recursive_pow(A, n)))

def get_A():
    # TODO: Find a matrix A
    # You have to return in the format of numpy array
    pass

def fibo(n):
    # TODO: Calcualte the n'th Fibonacci number
    A = get_A()
    pass

def f(n, k):
    # TODO: Calcualte the n'th number of the recursive sequence
    pass

def test_fibonacci():
    ## Test Cases for Fibonacci and Recursive Sequence ##

    a, b = 1, 1
    for i in range(2, 10):
        c = a + b
        assert (fibo(i) == c), "You implementation is incorrect"
        print("[Test Case %d]. Your implementation is correct!. fibo(%d) = %d" % (i - 2, i, fibo(i)))
        a = b
        b = c

    for n in range(5, 11):
        for k in range(2, 5):
            print("f(%d, %d) = %d" % (n, k, f(n, k)))
def DFS(A):
    # A is a mxn matrix
    pass

def BFS(A):
    # A is a mxn matrix
    pass


def findMinimum(A):
    # A is a mxn matrix
    pass

def test_bfs_dfs_find_minimum():
    ## Test Cases for BFS, DFS, Find Minimum ##
    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])

    BFS(A)

    DFS(A)

    A = np.array([[1, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 2], [1, 1, 0, 2, 1], [1, 1, 0, 2, 1]])

    findMinimum(A)

## Testing Your Code

test_geomery()
test_matrix_mul()
test_pow()
test_fibonacci()
test_bfs_dfs_find_minimum()

