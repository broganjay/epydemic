# Singleton class containing a compartment history (stored as a tuple of strings)
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


class CompartmentHistory:
    """A simple wrapper class containing a series of compartments in-order, so that 
    they may be interpreted as a compartment history. This class implements the singleton
    pattern, with an internal static dictionary of existing instances, to ensure that there 
    is sufficient reuse -- in standard models, one would expect that most nodes have identical 
    compartment histories.

    Compartment histories are immutable, with the underlying structure being a tuple of length n. 
    Histories may also be empty. 
    
    """

    _templates: dict[tuple[str, ...], "CompartmentHistory"] = dict() # stores existing instances of CompartmentHistory

    def __new__(cls, compartments: tuple[str, ...] = tuple()) -> "CompartmentHistory":
        """Create a new CompartmentHistory object. If the compartments are already in the
        _templates dictionary, return the existing object instead of creating a new one.

        :param compartments: a tuple of strings representing the compartments in the history
        :return: a CompartmentHistory object

        """
        if compartments in cls._templates:  # if obj already exists
            return cls._templates[compartments]  # return the existing
        else:
            template: CompartmentHistory = super().__new__(cls)  # or create a new one
            cls._templates[compartments] = template
            return template

    def __init__(self, compartments: tuple[str, ...] = tuple()) -> None:
        if not hasattr(self, "_initialised"):  # prevent re-setting attrs
            self._compartments: tuple[str, ...] = compartments
            self._length: int = len(compartments)
            self._initialised: bool = True
        # else: nothing
        # (__init__ is called every time regardless of whether newly created or just passed reference to existing)

    def getCompartments(self) -> tuple[str, ...]:
        """Get the compartments in the history.
        :return: a tuple of strings representing the compartments in the history
        """
        return self._compartments

    def getNumberOfCompartments(self) -> int:
        """Get the number of compartments in the history.
        :return: the number of compartments in the history
        """
        return self._length
    
    def getLatestCompartment(self) -> str:
        """Return the latest compartment in the history. This
        would be the current compartment of the node, provided the
        history is modified on each compartment change.
        
        :return: the latest compartment in the history
        """
        return self._compartments[-1] if self._length > 0 else "" # else an empty compartment...

    def __len__(self) -> int:
        return self._length

    def __contains__(self, compartment: str) -> bool:
        return compartment in self._compartments

    def __getitem__(self, compartment: int) -> str:
        return self._compartments[compartment]

    def __str__(self) -> str:
        return f"CompartmentHistory{self._compartments}"

    @staticmethod
    def updateHistory(history: "CompartmentHistory", nc: str) -> "CompartmentHistory":
        """To a caller, this method appears to update the current history with a new compartment.
        What is actually returned is a new object with the new compartment added, respecting the 
        immutability and subsequent reusability of the objects. Update history is likely called
        with identical arguments many times so this reuse makes significant savings."""
        return CompartmentHistory(
            history.getCompartments() + (nc,)
        )  # add new compartment to history
