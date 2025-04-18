# Interaction matriced between two models, either edgewise or nodewise
#
# Copyright (C) 2025 Brogan Irwin
#
# This file is part of epydemic, epidemic network simulations in Python.
#
# epydemic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# epydemic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with epydemic. If not, see <http://www.gnu.org/licenses/gpl.html>.


# There is a circular import between InteractionMatrix and CompartmentedModel, and between
# at the typing level (but not at the execution
# level), when providing types for the static methods. To
# deal with this we only import CompartmentedModel in order to type-check
# InteractionMatrix, and not for execution.  (See
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epydemic import CompartmentedModel



class InteractionMatrix:
    """An interaction matrix contains the probabilities of a particular event occurring
    based on the properties of the source and target compartments. Interaction matrices
    may be node-wise or edge-wise, depending on the type of interaction. Underneath the 
    matrix is simply represented as a 2D array of floats with various implemented 
    convenience methods.
    
    Interaction matrices can be used as interactions, although a simpler interface is 
    available in the `Interaction` class.
    """

    def __init__(self, rows: list[str], columns: list[str], fieldValue: float = 1.0) -> None:
        self._rows = rows
        self._columns = columns
        self._matrix = [
            [fieldValue for i in range(len(columns))] for j in range(len(rows))
        ]

    def getByName(self, rowName: str, columnName: str) -> float:
        """Get the value of the matrix at the given row and column names.
        
        :param rowName: the name of the row
        :param columnName: the name of the column
        :return: the value of the matrix at the given row and column names
        :raises ValueError: if the row or column name is not found
        :raises IndexError: if the row or column index is out of range
        """
        row = self._rows.index(rowName)
        column = self._columns.index(columnName)
        return self._matrix[row][column]

    def getByIndices(self, row: int, column: int) -> float:
        """Get the value of the matrix at the given row and column indices.

        :param row: the index of the row
        :param column: the index of the column
        :return: the value of the matrix at the given row and column indices
        :raises ValueError: if the row or column index is not found
        :raises IndexError: if the row or column index is out of range
        """
        return self._matrix[row][column]

    def __str__(self) -> str:
        # width of each column
        col_widths = [
            max(len(str(self._matrix[i][j])) for i in range(len(self._rows)))
            for j in range(len(self._columns))
        ] # width of each column is length of longest entry at i,j 
        col_widths = [
            max(col_widths[j], len(self._columns[j])) for j in range(len(self._columns))
        ] # width is increased to length of column header if > length of entry
        row_width = max(len(row) for row in self._rows) # then row width is max row (header) length

        # header
        string = " " * (row_width + 1)
        for j in range(len(self._columns)):
            string += f"{self._columns[j]:<{col_widths[j]}} " # ensure fits 
        string += "\n"

        # each row
        for i in range(len(self._rows)):
            string += f"{self._rows[i]:<{row_width}} "
            for j in range(len(self._columns)):
                string += f"{self._matrix[i][j]:<{col_widths[j]}} "
            string += "\n"
        return string

    def replaceEntriesMatching(self, c: str, p: float) -> None:
        """Replace all entries in the matrix that match the given compartment name (substring match) with the given value.
        The match is satisfied on either column or row.
        
        :param c: the compartment name to match
        :param p: the value to replace the matching entries with
        """
        for i in range(len(self._rows)):
            for j in range(len(self._columns)):
                if c in self._rows[i] or c in self._columns[j]:
                    self._matrix[i][j] = p

    def replaceEntriesInColumn(self, c: str, p: float) -> None: 
        """Replace all entries in the matrix that match the given compartment name (substring match) with the given value.
        The match is satisfied on column _only_.

        :param c: the compartment name to match
        :param p: the value to replace the matching entries with
        """
        for j in range(len(self._columns)):
            if c in self._columns[j]:
                for i in range(len(self._rows)):
                    self._matrix[i][j] = p

    def replaceEntriesInRow(self, c: str, p: float) -> None:
        """Replace all entries in the matrix that match the given compartment name (substring match) with the given value.
        The match is satisfied on row _only_.

        :param c: the compartment name to match
        :param p: the value to replace the matching entries with
        """
        for i in range(len(self._rows)):
            if c in self._rows[i]:
                for j in range(len(self._columns)):
                    self._matrix[i][j] = p

    def replaceEntryByRowColumn(self, row: str, column: str, p: float) -> None:
        """Replace the entry in the matrix at the given row and column with the given value.
        Unlike :meth:`replaceEntryByRowColumnMatch`, this method does not perform a substring match,
        and will throw an error if the row or column is not found.

        :param row: the name of the row
        :param column: the name of the column
        :param p: the value to replace the entry with
        :raises ValueError: if the row or column name is not found
        :raises IndexError: if the row or column index is out of range
        """
        self._matrix[self._rows.index(row)][self._columns.index(column)] = p

    def replaceEntryByRowColumnMatch(self, r: str, c: str, p: float) -> None:
        """Replace the entry in the matrix at the given row and column with the given value.
        This does perform a substring match on the row and column names but will only replace the 
        value if a match is found for _both_.

        :param r: the name of the row
        :param c: the name of the column
        :param p: the value to replace the entry with
        :raises ValueError: if the row or column name is not found
        :raises IndexError: if the row or column index is out of range
        """

        for i in range(len(self._rows)):
            for j in range(len(self._columns)):
                if r in self._rows[i] and c in self._columns[j]: # both must match
                    self._matrix[i][j] = p

    def containsEntry(self, entry: float) -> bool:
        """Check if the matrix contains the given entry. Returns on the first match found.
        
        :param entry: the entry to check for
        :return: True if the entry is found, False otherwise
        """
        for i in range(len(self._rows)):
            if entry in self._matrix[i]:
                return True
        return False

    def replaceEntriesInRowOnSecondMatch(self, c: str, p: float) -> None:
        """Replace all entries in the matrix that match the given compartment name where
        the compartment name must be the second component of the row/column name (still a substring match).
        In normal use the second component of the row/column name is a compartment of the target model, so this 
        method can be useful when modifying the matrix to contain an entry based on the compartment of the target model.

        :param c: the compartment name to match
        :param p: the value to replace the matching entries with
        """
        for i in range(len(self._rows)):
            if c in self._rows[i].split("+")[1]:
                for j in range(len(self._columns)):
                    self._matrix[i][j] = p

    @staticmethod
    def generateEdgewiseInteractionMatrixForTwoModels(m1: "CompartmentedModel", m2: "CompartmentedModel", fieldValue: float = 1.0) -> "InteractionMatrix":
        """Generate an edgewise interaction matrix for two models. This is a convenience
        method that uses the `generateEdgewiseInteractionMatrixForCompartments` method with 
        the compartments of the model.
        
        :param m1: the first model
        :param m2: the second model
        :param fieldValue: the value to fill the matrix with
        :return: the edgewise interaction matrix 
        """
        return InteractionMatrix.generateEdgewiseInteractionMatrixForCompartments(
            m1.compartments(), m2.compartments(), fieldValue
        )

    @staticmethod
    def generateNodewiseInteractionMatrixForTwoModels(m1: "CompartmentedModel", m2: "CompartmentedModel", fieldValue: float = 1.0) -> "InteractionMatrix":
        """Generate a nodewise interaction matrix for two models. This is a convenience
        method that obtains the compartments of each model. The compartments of m1 are 
        assigned as rows, and the compartments of m2 are assigned as columns. An entry
        is added for each combination of compartments from m1 and m2. 

        It is important to note that these matrices are directed and so cannot be 
        transposed to produce the same meaning.
        
        :param m1: the first model
        :param m2: the second model
        :param fieldValue: the value to fill the matrix with
        :return: the nodewise interaction matrix"""

        rows = [(c) for c in m1.compartments()]
        columns = [(c) for c in m2.compartments()]
        return InteractionMatrix(rows, columns, fieldValue)

    @staticmethod
    def shaveCName(c: str) -> str:
        """Shave the compartment name of the prefix to shorten its representation. An example would be
        `shaveCName("epydemic.sir.S") = "S"`. This is useful for displaying the matrix in a more readable format,
        and should not be used otherwise.

        :param c: the compartment name to shave
        :return: the shaved compartment name
        """
        return c.split(".")[-1]

    @staticmethod
    def generateCrossImmuneInteractionMatrixForTwoModels(
        m1: "CompartmentedModel", m2: "CompartmentedModel", previouslyInfectedCompartments: list[str]
    ) -> "InteractionMatrix":
        """Generate a cross-immune interaction matrix for two models. This creates an
        edgewise interaction matrix between the two models with each entry being 1 as default,
        except for rows that match any previously infected compartments which are set to 0.
        This does, of course, require that all compartments where the node is regarded as previously
        infected be specified which reduces flexibility and increases complexity where there are many
        post-infection compartments.
        
        :param m1: the first model
        :param m2: the second model
        :param previouslyInfectedCompartments: the compartments that are previously infected
        :return: the cross-immune interaction matrix
        """
        mat = InteractionMatrix.generateEdgewiseInteractionMatrixForTwoModels(m1, m2)
        for c in previouslyInfectedCompartments:
            mat.replaceEntriesInRow(c, 0)
        return mat

    @staticmethod
    def generateInfectionPreconditionInteractionMatrixForTwoModels(
        m1: "CompartmentedModel", m2: "CompartmentedModel", susceptibleD1Compartments: list[str], infectedD2Compartments: list[str]
    ) -> "InteractionMatrix":
        """Generate an infection precondition interaction matrix for two models. This creates an
        edgewise interaction matrix between the two models with each entry being 0 as default,
        except for rows that match any susceptible compartments in model 1 and columns that match
        any infected compartments in model 2 which are set to 1. This has the effect of allowing
        infection only when the node has a compartment for d2 which i in `infectedD2Compartments`.

        :param m1: the first model
        :param m2: the second model
        :param susceptibleD1Compartments: the compartments that are susceptible in model 1
        :param infectedD2Compartments: the compartments that are infected in model 2
        :return: the infection precondition interaction matrix
        """
        mat = InteractionMatrix.generateEdgewiseInteractionMatrixForTwoModels(m1, m2)
        for c1 in susceptibleD1Compartments:
            for c2 in infectedD2Compartments:
                mat.replaceEntryByRowColumnMatch(c1, c2, 0)
        return mat

    @staticmethod
    def generateInfectionEnablerInteractionMatrixForTwoModels(m1: "CompartmentedModel", m2: "CompartmentedModel", uninfected: list[str]) -> "InteractionMatrix":
        """Generate an infection enabler interaction matrix for two models. This creates an
        edgewise interaction matrix between the two models with each entry being 1 as default,
        except for rows that match any uninfected compartments which are set to 0. This has the effect
        of allowing infection only when the node has a compartment for d2 which is not in `uninfected, 
        i.e when the node is considered infected.

        :param m1: the first model
        :param m2: the second model
        :param uninfected: the compartments that are uninfected in model 2
        :return: the infection enabler interaction matrix
        """

        mat = InteractionMatrix.generateEdgewiseInteractionMatrixForTwoModels(m1, m2)
        for c in uninfected:
            mat.replaceEntriesInRowOnSecondMatch(c, 0)
        return mat

    @staticmethod
    def generateEdgewiseInteractionMatrixForCompartments(c1s: list[str], c2s: list[str], fieldValue: float = 1.0) -> "InteractionMatrix":
        """Generate an edgewise interaction matrix for the given compartments. This creates an
        edgewise interaction matrix between the two models with each entry being 1 as default.
        The rows and columns are of the form c1+c2, with the rows being d1 and the columns being d2.
        
        :param c1s: the compartments of the first model
        :param c2s: the compartments of the second model
        :param fieldValue: the value to fill the matrix with
        :return: the edgewise interaction matrix
        """
        rows = []
        columns = []
        for s1 in c1s:
            for s2 in c2s:
                rows.append(f"{(s1)}+{(s2)}")
                columns.append(f"{(s1)}+{(s2)}")
        return InteractionMatrix(rows, columns, fieldValue)

    @staticmethod
    def generateNodewiseInfectionPreconditionInteractionMatrixForTwoModels(
        m1: "CompartmentedModel", m2: "CompartmentedModel", susceptibleD1Compartments: list[str], infectedD2Compartments: list[str]
    ) -> "InteractionMatrix":
        """Generate a _nodewise_ infection precondition interaction matrix for two models. Such a matrix
        can be used against events which do not originate from an edge but originate from a node itself. A clear
        use includes spontaneous emergence of a secondary process -- a nodewise interaction matrix could be used to prevent
        a node being infected if it would not be permitted to through ordinary infection, i.e preventing seeding on an 
        otherwise immune node.
          
        :param m1: the first model
        :param m2: the second model
        :param susceptibleD1Compartments: the compartments that are susceptible in model 1
        :param infectedD2Compartments: the compartments that are infected in model 2
        :return: the infection precondition interaction matrix
        """
        mat = InteractionMatrix.generateNodewiseInteractionMatrixForTwoModels(m1, m2, 1)
        for s1 in susceptibleD1Compartments:
            mat.replaceEntriesInRow(s1, 0)
        return mat

    @staticmethod
    def generateNodewiseInfectionEnablerInteractionMatrixForTwoModels(
        m1: "CompartmentedModel", m2: "CompartmentedModel", uninfected: list[str]
    ) -> "InteractionMatrix":
        """Generate a _nodewise_ infection enabler interaction matrix for two models. Such a matrix
        can be used against events which do not originate from an edge but originate from a node itself.
        
        :param m1: the first model
        :param m2: the second model
        :param uninfected: the compartments that are uninfected in model 2
        :return: the infection enabler interaction matrix
        """

        mat = InteractionMatrix.generateNodewiseInteractionMatrixForTwoModels(m1, m2, 1)
        for c in uninfected:
            mat.replaceEntriesInColumn(c, 0)
        return mat
