from typing import Tuple, Literal

class InteractionMatrix:

    def __init__(self, rows, columns):
        self._rows = rows 
        self._columns = columns
        self._matrix = [[1 for i in range(len(columns))] for j in range(len(rows))]

    def getByName(self, rowName, columnName):
        row = self._rows.index(rowName)
        column = self._columns.index(columnName)
        return self._matrix[row][column]
    
    def getByIndices(self, row, column):
        return self._matrix[row][column]
    
    def __str__(self):
        # width of each column
        col_widths = [max(len(str(self._matrix[i][j])) for i in range(len(self._rows))) for j in range(len(self._columns))]
        col_widths = [max(col_widths[j], len(self._columns[j])) for j in range(len(self._columns))]
        row_width = max(len(row) for row in self._rows)

        # header
        string = " " * (row_width + 1)
        for j in range(len(self._columns)):
            string += f"{self._columns[j]:<{col_widths[j]}} "
        string += "\n"

        # each row 
        for i in range(len(self._rows)):
            string += f"{self._rows[i]:<{row_width}} "
            for j in range(len(self._columns)):
                string += f"{self._matrix[i][j]:<{col_widths[j]}} "
            string += "\n"
        return string
    
    def replaceEntriesMatching(self, c, p):
        for i in range(len(self._rows)):
            for j in range(len(self._columns)):
                if c in self._rows[i] or c in self._columns[j]:
                    self._matrix[i][j] = p

    def replaceEntriesInColumn(self, c, p):
        for j in range(len(self._columns)):
            if c in self._columns[j]:
                for i in range(len(self._rows)):
                    self._matrix[i][j] = p

    def replaceEntriesInRow(self, c, p):
        for i in range(len(self._rows)):
            if c in self._rows[i]:
                for j in range(len(self._columns)):
                    self._matrix[i][j] = p

    def containsEntry(self, entry):
        for i in range(len(self._rows)):
            if entry in self._matrix[i]:
                return True
        return False

    @staticmethod
    def generateEdgewiseInteractionMatrixForTwoModels(m1, m2):
        # matrix is one-directional, meaning [m1][m2] != [m2][m1]
        # so each row and column must be composed of all possible combinations, e.g
        # s1s2 s1s2 etc... 
        # recall that infection occurs at an SI edge NOT an IS edge
        rows = []
        columns = []
        for s1 in m1.compartments():
            for s2 in m2.compartments():
                rows.append(f"{(s1)}+{(s2)}")
                columns.append(f"{(s1)}+{(s2)}")
        return InteractionMatrix(rows, columns)
    
    @staticmethod
    def generateNodewiseInteractionMatrixForTwoModels(m1, m2):
        # nodewise is much simpler -- m1 has rows, m2 has columns
        # as actual compartments is unordered

        rows = [(c) for c in m1.compartments()]
        columns = [(c) for c in m2.compartments()]
        return InteractionMatrix(rows, columns)
    
    @staticmethod
    def shaveCName(c):
        return c.split(".")[-1]
    
    @staticmethod 
    def generateCrossImmuneInteractionMatrixForTwoModels(m1, m2, previouslyInfectedCompartments):
        mat = InteractionMatrix.generateEdgewiseInteractionMatrixForTwoModels(m1, m2)
        for c in previouslyInfectedCompartments:
            mat.replaceEntriesInRow(c, 0)
        return mat
