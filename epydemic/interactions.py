# Class containing all information necessary for representing an interaction between two models
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

class Interaction:
    """An `Interaction` is essentially a record which contains information about an interaction 
    including the source, target, their compartments and the nature of the interaction. It is a lightweight wrapper class
    which makes comprehension of otherwise complex interactions simpler than an alternative (such as, say, a tuple).
    Interactions may define both a `modifier` and an `otherwise` value. The `modifier` is returned if the interaction
    condition is evaluated to true, and the `otherwise` value is returned if the condition is evaluated to false. The 
    `otherwise` value is optional. This class could easily be extended further to chain interactions or add more specific 
    conditions such as those which are sensitive to time or other factors. 
    """

    STD: str = "epydemic.interactions.std"
    CUSTOM: str = "epydemic.interactions.custom"


    CROSSIMMUNITY: str = "epydemic.interactions.crossimmunity"
    CAUSES_IMMUNITY: str = "epydemic.interactions.causesImmunity"
    REQUIRED_PREINFECTION: str = "epydemic.interactions.requiredPreinfection"
    INFECTION_PRECONDITION: str = "epydemic.interactions.infectionPrecondition"

    def __init__(self, source: str, eventName: str, modifier: float, targetCompartments: list[str], isIn: bool, historical: bool, target: str, otherwise: float) -> None:
        self._source = source
        self._eventName = eventName
        self._modifier = modifier
        self._targetCompartments = targetCompartments
        self._isIn = isIn
        self._historical = historical
        self._target = target
        self._otherwise = otherwise

    def isHistorical(self) -> bool:
        """Returns whether the interaction is historical or not.
        A historical interaction is one which considers the entire history of the target node
        or just considers the current node (last history entry).
        
        :return: True if the interaction is historical, False otherwise
        """
        return self._historical

    def requiresIn(self) -> bool:
        """Return whether the interaction requires the specified compartments to be in the history,
        or if it requires them _not_ to be in the history.

        :return: True if the interaction requires the compartments to be in the history, False otherwise
        """
        return self._isIn

    def getModifier(self) -> float:
        """Return the modifier value used if the interaction is satisfied.
        
        :return: the modifier value
        """
        return self._modifier

    def getOtherwise(self) -> float:
        """Return the otherwise value used if the interaction is not satisfied.

        :return: the otherwise value
        """
        return self._otherwise

    def getTargetCompartments(self) -> list[str]:
        """Return the compartments which are the target of the interaction.

        :return: the target compartments
        """
        return self._targetCompartments

    def getSource(self) -> str:
        """Return the source of the interaction.

        :return: the source of the interaction
        """
        return self._source

    def getEventName(self) -> str:
        """Return the event name associated with the interaction.

        :return: the event name associated with the interaction
        """
        return self._eventName

    def getTarget(self) -> str:
        """Return the target of the interaction.
        
        :return: the target of the interaction
        """
        return self._target

    def __str__(self) -> str:
        return f"interaction for disease {self._source}: event {self._eventName} has modified probability of occurring {self._modifier} if compartment(s) {self._targetCompartments} are{"" if self._isIn else " not"} in {self._target} {"disease history" if self._historical else "current compartment"}, otherwise {self._otherwise}"


    @staticmethod
    def createCrossImmunityBetween(
        source: str,
        target: str,
        eventName: str,
        sourceCompartments: list[str],
        targetCompartments: list[str],
    ) -> list["Interaction"]:
        """Create a cross-immunity interaction between two diseases. As interactions are directed, 
        this method returns two separate interactions, one for each direction. Each individual interaction
        encodes that the source is unable to infect the node if the node has previously been infected by the target.
        
        :param source: the source disease
        :param target: the target disease
        :param eventName: the event name associated with the interaction
        :param sourceCompartments: the compartments of the source disease
        :param targetCompartments: the compartments of the target disease
        :return: a list of interactions, one for each direction
        """
        interactions = []
        sourceInteraction = Interaction(
            source, eventName, 0.0, targetCompartments, True, True, target, 1.0
        )
        targetInteraction = Interaction(
            target, eventName, 0.0, sourceCompartments, True, True, source, 1.0
        )
        interactions.append(sourceInteraction)
        interactions.append(targetInteraction)
        return interactions

    @staticmethod
    def createRequiredPreinfectionFor(source: str, target: str, eventName: str, targetCompartments: list[str]) -> list["Interaction"]:
        """Create a required preinfection interaction between two diseases. This interaction
        is one way, meaning that this method returns an interaction encoding that the source disease 
        may only infect nodes if the nodes have been previously infected by target.
        
        :param source: the source disease
        :param target: the target disease
        :param eventName: the event name associated with the interaction
        :param targetCompartments: the compartments of the target disease
        :return: the infection precondition interaction (in a list)
        """
        # where source needs node to have been infected with target
        interaction = Interaction(
            source, eventName, 1.0, targetCompartments, True, True, target, 0.0
        )
        return [interaction]
    
    @staticmethod
    def createInfectionImmunityFor(source: str, target: str, eventName: str, targetCompartments: list[str]) -> list["Interaction"]:
        """Create an infection immunity interaction between two diseases, where 
        previous infection with target prevents infection with source (one way).
        
        :param source: the source disease
        :param target: the target disease
        :param eventName: the event name associated with the interaction
        :param targetCompartments: the compartments of the target disease
        :return: the infection immunity interaction (in a list)
        """
        # where previous infection with target prevents infection with source
        interaction = Interaction(
            source, eventName, 0.0, targetCompartments, True, True, target, 1.0
        )
        return [interaction]
