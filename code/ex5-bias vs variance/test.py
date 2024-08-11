class Mat(list):
    def __matmul__(self, B):
        A = self
        return Mat([[sum(A[i][k]*B[k][j] for k in range(len(B)))
                    for j in range(len(B[0])) ] for i in range(len(A))])
A = Mat([[1,3],[7,5]])
B = Mat([[6,8],[4,2]])
print(A @ B)