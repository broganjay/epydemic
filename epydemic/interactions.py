class Interaction:

    CROSSIMMUNITY: str = "epydemic.interactions.crossimmunity"
    REQUIRED_PREINFECTION: str = "epydemic.interactions.requiredPreinfection"

    def __init__(
        self,
        source,
        eventName,
        modifier,
        targetCompartments,
        isIn,
        historical,
        target,
        otherwise,
    ):
        self._source = source
        self._eventName = eventName
        self._modifier = modifier
        self._targetCompartments = targetCompartments
        self._isIn = isIn
        self._historical = historical
        self._target = target
        self._otherwise = otherwise

    def isHistorical(self):
        return self._historical

    def requiresIn(self):
        return self._isIn

    def getModifier(self):
        return self._modifier

    def getOtherwise(self):
        return self._otherwise

    def getTargetCompartments(self):
        return self._targetCompartments

    def __str__(self):
        return f"interaction for disease {self._source}: event {self._eventName} has modified probability of occurring {self._modifier} if compartment(s) {self._targetCompartments} are{"" if self._isIn else " not"} in {self._target} {"disease history" if self._historical else "current compartment"}, otherwise {self._otherwise}"

    def getSource(self):
        return self._source

    def getEventName(self):
        return self._eventName

    def getTarget(self):
        return self._target

    @staticmethod
    def createCrossImmunityBetween(
        source, target, eventName, sourceCompartments, targetCompartments
    ):
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
    def createRequiredPreinfectionFor(source, target, eventName, targetCompartments):
        # where source needs node to have been infected with target
        interaction = Interaction(
            source, eventName, 1.0, targetCompartments, True, True, target, 0.0
        )
        return [interaction]
