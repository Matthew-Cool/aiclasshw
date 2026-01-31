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
        return Geometry.count

    def distanceBetweenTuples(self, a, b):
        x1, y1 = self.points[a]
        x2, y2 = self.points[b]
        distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return np.sqrt(distance)

class Triangle(Geometry):
    def __init__(self, a, b, c):
        # a, b, c are tuples that represent for 3 vertices of a triangle
        # TODO: Your task is to implement the constructor
        #super(Triangle, self).__init__(?, ?)
        super(Triangle, self).__init__("Triangle", [a, b, c])

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        side1 = self.distanceBetweenTuples(0, 1)
        side2 = self.distanceBetweenTuples(0, 2)
        side3 = self.distanceBetweenTuples(1, 2)

        semiPerimeter = (side1+side2+side3)/2

        return np.sqrt(semiPerimeter * (semiPerimeter - side1)*(semiPerimeter - side2)*(semiPerimeter - side3))
        

class Rectangle(Geometry):
    def __init__(self, a, b):
        # a, b are tuples that represent for top and bottom vertices of a rectangle
        # TODO: Your task is to implement the constructor
        # super(Rectangle, self).__init__(?, ?)
        x1, y1 = a
        x2, y2 = b
        d = (x2, y1)
        c = (x1, y2)
        super(Rectangle, self).__init__("Rectangle", [a, d, b, c])

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        x = self.distanceBetweenTuples(0, 1)
        y = self.distanceBetweenTuples(0, 3)

        return x*y

class Square(Rectangle): #MIGHT NEED TO DOUBLE CHECK IMPLEMENTATION
    def __init__(self, a, length):
        # a is a tuple that represent a top vertex of a square
        # length is the side length of a square
        # TODO: Your task is to implement the constructor
        # super(Square, self).__init__(?, ?)
        b = (a[0] + length, a[1] - length)
        super(Square, self).__init__(a, b)

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        side = self.distanceBetweenTuples(0, 1)
        return side * side

class Circle(Geometry):
    def __init__(self, o, r):
        # o is a tuple that represent a centre of a circle
        # r is the radius of a circle
        # TODO: Your task is to implement the constructor
        # super(Circle, self).__init__(?, ?)
        super(Circle, self).__init__("Circle", o)
        self.r = r

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        return (self.r ** 2)*(np.pi)

class Polygon(Geometry):
    def __init__(self, points):
        # points is a list of tuples that represent vertices of a polygon
        # TODO: Your task is to implement the constructor
        # super(Polygon, self).__init__(?, ?)
        super(Polygon, self).__init__("Polygon", points)

    def calculate_area(self):
        #TODO: Your task is required to implement a area function
        numOfPoints = len(self.points)
        area = 0
        for i in range(2, numOfPoints):
            side1, side2, side3 = self.distanceBetweenTuples(0, i-1), self.distanceBetweenTuples(0, i), self.distanceBetweenTuples(i, i-1)
            semiPerimeter = (side1 + side2 + side3)/2
            area += np.sqrt(semiPerimeter * (semiPerimeter - side1)*(semiPerimeter - side2)*(semiPerimeter - side3))
        return area


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
    n, k = A.shape
    j, m = B.shape
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for p in range(m):
            for l in range(k):
                C[i][p] += A[i,l] * B[l,p]
    return np.array(C)

def test_matrix_mul():
    ## Test cases for matrix multplication ##

    for test in range(10):
        m, n, k = random.randint(3, 10), random.randint(3, 10), random.randint(3, 10)
        A = np.random.randn(m, n)
        B = np.random.randn(n, k)
        assert np.mean(np.abs(A.dot(B) - matrix_multiplication(A, B))) <= 1e-7, "Your implmentation is wrong!"
        print("[Test Case %d]. Your implementation is correct!" % test)

def recursive_pow(A, n): # FIXME: Should work, but not how the slides do it so idk if thats an issue lol
    # TODO: Your task is required implementing
    # a recursive function
    if n == 0:
        return np.identity(A.shape[0])
    elif n == 1:
        return A
    else:
        return matrix_multiplication(A, recursive_pow(A, n-1))

def iterative_pow(A, n):
	# TODO: Your task is required implementing
    # a iterative function
    if n == 0:
        return np.identity(A.shape[0])
    B = A
    for i in range(n-1):
        B = matrix_multiplication(A, B)
    return B

def test_pow():
    ## Test cases for the pow function ##

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Recursive: A^{} = {}".format(n, recursive_pow(A, n)))

    for test in range(10):
        n = random.randint(2, 5)
        A = np.random.randn(n, n)
        print("Iterative: A^{} = {}".format(n, iterative_pow(A, n))) #fixed, this should be iterative, not recursive

def get_A():
    # TODO: Find a matrix A
    # You have to return in the format of numpy array
    return np.array([[1.0,1.0], [1.0, 0.0]])

def fibo(n):
    # TODO: Calcualte the n'th Fibonacci number
    A = get_A()
    fib1 = np.array([[1.0],[1.0]])
    fibN = recursive_pow(A, n-1) @ fib1
    return int(fibN[0][0])

def f(n, k):
    # TODO: Calcualte the n'th number of the recursive sequence
    fn = 0
    if (n < k):
        return 1
    else:
        for i in range(1, k+1):
            fn += f(n-i, k)
        return fn

    
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

def recursiveDFS(A, x, y, visited, path):
    m, n = A.shape
    visited[x][y] = 1

    #are we at the end??? If so, lets print this and be done
    if(x == m-1 and y == n-1):
        print("(0,0)", end="")
        for (u,v) in path:
            print(" -> (%d, %d)" % (u,v), end="")
        print()
        return True
    
    #lets check next door 
    for xx, yy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        u = x + xx
        v = y + yy
        if (0 <= u) and (u < m) and (0 <= v) and (v < n) and (A[u][v] != 0) and visited[u][v] == 0:
            path.append((u,v))

            if(recursiveDFS(A,u,v,visited,path) == True):
                return True
            
            path.pop()

    return False


def DFS(A):
    # A is a mxn matrix
    result = recursiveDFS(A, 0, 0, np.array([[0] * A.shape[1] for _ in range(A.shape[0])]), [])
    if not result: #if no path is found at all... I think it works lol
        print("-1")

def BFS(A):
    # A is a mxn matrix
    m, n = A.shape
    
    # Check if start or end is blocked
    if A[0][0] == 0 or A[m-1][n-1] == 0:
        print("-1")
        return
    
    visited = np.zeros((m, n), dtype=int)
    px, py = np.full((m, n), -1, dtype=int), np.full((m, n), -1, dtype=int)
    queue = []
    queue.append((0, 0))
    visited[0][0] = 1

    while queue:
        x, y = queue.pop(0)
        
        #check if reached destination
        if x == m - 1 and y == n - 1:
            break
        
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            u = x + dx
            v = y + dy
            if 0 <= u < m and 0 <= v < n and A[u][v] != 0 and visited[u][v] == 0:
                queue.append((u, v))
                visited[u][v] = 1
                px[u][v] = x
                py[u][v] = y

    #
    if visited[m-1][n-1] == 0:
        print("-1")
    else:
        path = []
        x, y = m - 1, n - 1
        while not (x == 0 and y == 0):
            u, v = px[x][y], py[x][y]
            path.append((u, v))
            x, y = u, v
        
        for u, v in path[::-1]:
            print("(%d, %d) -> " % (u, v), end="")
        print("(%d, %d)" % (m - 1, n - 1))





def findMinimum(A):
    # A is a mxn matrix
    pass

def test_bfs_dfs_find_minimum():
    ## Test Cases for BFS, DFS, Find Minimum ##
    A = np.array([[1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 0], 
                  [1, 1, 1, 1, 1], 
                  [1, 1, 0, 1, 0], 
                  [1, 1, 0, 1, 1]])

    BFS(A)

    DFS(A)

    A = np.array([[1, 1, 1, 0, 1], 
                  [0, 0, 1, 0, 0], 
                  [1, 1, 1, 1, 2], 
                  [1, 1, 0, 2, 1], 
                  [1, 1, 0, 2, 1]])

    findMinimum(A)

## Testing Your Code

test_geomery()
test_matrix_mul()
test_pow()
test_fibonacci()
test_bfs_dfs_find_minimum()

