# PiRaP lab2
import numpy as np

"""
Task 1
Define a function getDistanceMatrix that takes an NxD  array of N points in a D-dimensional space and returns
an NxN matrix with Euclidean distances between points. Loops are allowed only for iterating over points. 
Note, that the matrix is symmetric.
"""


def getDistanceMatrix(arr):
    # Transposition
    arrT = np.transpose(arr)

    n = arrT.shape[0]  # How many points
    d = arrT.shape[1]  # How many dimensions

    dist = np.empty((n, n))  # Distance implementation of n,n (Symetric)

    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(arrT[i, :] - arrT[j, :])
    return dist


"""
Task 2

Define a function eliminateDistances(f, Q) which alters each element of Q with probability f by changing it to the infinity (numpy.inf constant). 
Do not modify the original – return a new matrix. Loops are not allowed! 
Hint: generate a matrix of random numbers from a uniform distribution and use logical indexing.
"""


def eliminateDistance(f, Q):
    # uniform distribution of size Q
    r = np.random.uniform(size=Q.shape)

    new_Q = np.where(r < f, np.inf, Q)

    return new_Q


"""
Task 3

Define a function calculateFloydWarshall which executes Floyd-Warshall algorithm for finding shortest paths in a graph
using distance matrix (see the pseudocode). Do not modify the original – return a new matrix.

for k from 1 to N
    for i from 1 to N
        for j from 1 to N
            if Q(i,j) > Q(i,k) + Q(k,j) then Q(i,j) = Q(i,k) + Q(k,j)
"""


def calculateFloydWarshall(dist):
    n = dist.shape[0]
    A = dist.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                A[i][j] = min(A[i][j], A[i][k] + A[k][j])

    return A


"""
Task 5
Define a function generateSystem() which generates a system of linear equations AX = B: 

The function takes a column vector X and returns a tuple containing matrix A of coefficients and column vector B of free terms. 
Elements of A are drawn from normal distribution N(0, 100) with an exception of diagonal elements which can be any values 
fulfilling the diagonal dominance requirement: |aii| >= ∑|aij|. Loops are not allowed.
"""


def generateSystem(X):
    # Rows of X matrix
    shape = X.shape[0]
    A = np.random.normal(0, 100, (shape, shape))
    # Sums of matix columns
    diag_sum = np.abs(A).sum(axis=0)
    # Filling diagonal sum in A matrix
    np.fill_diagonal(A, diag_sum)

    # Calculating dot product AX
    B = np.dot(A, X)

    return A, B


"""
Task 6
"""


def solveSystem(A, b):
    x = np.zeros(len(A[0]))

    # diagonal
    D = np.diag(A)

    # Zero diagonal
    R = A - np.diagflat(D)

    new_x = (b - np.dot(R, x)) / D

    while not (np.all(np.isclose(new_x, x, rtol=1e-02, atol=1e-04))):
        x = new_x
        new_x = (b - np.dot(R, x)) / D

    return x


if __name__ == '__main__':
    # Exemplary matrix for task1, rows are dimesnions and columns points
    arr1 = np.array([
        [2, 2, 3, 4],
        [4, 5, 6, 4],
        [1, 2, 3, 4],
        [3, 4, 2, 5],
        [3, 4, 7, 5]
    ])

    temp1 = getDistanceMatrix(arr1)
    print(f'Taks 1:\n', temp1)

    # Exemplary matrix for task2
    arr2 = np.array([[1, 2], [3, 4]])
    # probability treshold
    prob = 0.3

    elim = eliminateDistance(prob, arr2)
    print(f'Task 2\n', elim)

    # Exemplary matrix for task3
    dist = np.array([
        [0, 3, 8, np.inf, -4],
        [np.inf, 0, np.inf, 1, 7],
        [np.inf, 4, 0, np.inf, np.inf],
        [2, np.inf, -5, 0, np.inf],
        [np.inf, np.inf, np.inf, 6, 0]
    ])
    graph = np.array([
        [0, 3, np.inf, 5],
        [2, 0, np.inf, 4],
        [np.inf, 1, 0, np.inf],
        [np.inf, np.inf, 2, 0]
    ])

    shortest = calculateFloydWarshall(graph)

    print(f'Task3:\n', shortest)

    """
    Task 4

    Generate randomly a set of 100 points in a 3D Euclidean space. Calculate distance matrix Q between all points with getDistanceMatrix. 
    Eliminate 50% of elements from Q with eliminateDistances function and apply calculateFloydWarshall. 
    Compare sum of all elements in the original and output matrix. 
    Repeat the experiment for 90% and 99% elimination. Comment the results.
    """

    # Matrix 3x100
    points = np.random.rand(3, 100)
    temp4 = getDistanceMatrix(points)

    prob2 = 0.5
    elim2 = eliminateDistance(prob2, temp4)

    shortest4 = calculateFloydWarshall(elim2)

    print(f'Task 4 f:{prob2}')
    print(f'Sum of rows of distances', sum(temp4.sum(axis=1)))
    print(f'Sum of rows using FloydWarshall', sum(shortest4.sum(axis=1)))

    # Exemplary vactor X for task 5
    X = np.array([1, 2, 3])

    A, B = generateSystem(X)
    print(f'Task 5:\n A={A}\n B={B}')

    A = np.array([[2, 1],
                  [5, 7]])

    b = np.array([11, 13])

    temp6 = solveSystem(A, b)

    print(f'Task 6:\n A:\n{A}\n b:\n{b}\n x:{temp6}')


