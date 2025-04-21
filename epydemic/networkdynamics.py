# Networks dynamics simulation base class
#
# Copyright (C) 2017--2023 Simon Dobson
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

import json
import sys
from heapq import heappush, heappop
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, cast

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final
from networkx import Graph
from epydemic import (
    NetworkExperiment,
    Locus,
    Process,
    NetworkGenerator,
    EventFunction,
    EventDistribution,
    Element,
    InteractionMatrix,
    Condition,
    rng,
    Interaction,
)

# Event types (not exported outside this file)
PostedEventFunction = Callable[[], None]
PostedEvent = Tuple[float, int, Optional[Process], Optional[PostedEventFunction], Element, str]


class Dynamics(NetworkExperiment):
    """An abstract simulation framework for running a process over a network.

    This is the abstract base class for implementing different kinds
    of network process dynamics as computational experiments suitable
    for running under ``epyc``. Sub-classes provide synchronous and
    stochastic (Gillespie) simulation dynamics.

    The dynamics runs a network process provided as a :class:`Process`
    object. It is provided with a network generator that is called to
    generate a new experimental network instance for each run. The
    generator can be any iterator but will typically be an instance of
    :class:`NetworkGenerator`.

    :param p: network process to run
    :param g: (optional) prototype network or network generator"""

    # Additional metadata elements
    TIME: Final[str] = (
        "epydemic.monitor.time"  #: Metadata element holding the logical simulation end-time.
    )
    EVENTS: Final[str] = (
        "epydemic.monitor.events"  #: Metadata element holding the number of events that happened.
    )

    def __init__(self, p: Process, g: Optional[Union[Graph, NetworkGenerator]] = None):
        super().__init__(g)

        # initialise other fields
        self._eventId: int = 0  # counter for posted events
        self._process: Process = p  # network process to run
        self._process.setDynamics(
            self
        )  # back-link from process to dynamics (for events)
        self._simulationTime: float = 0.0  # on-going simulation time
        self._loci: Dict[str, Locus] = dict()  # dict from names to loci
        self._processLoci: Dict[Process, Dict[str, Locus]] = (
            dict()
        )  # dict from processes to loci for events
        self._perElementEvents: Dict[Process, EventDistribution] = (
            dict()
        )  # dict from processes to events that occur per-element
        self._perLocusEvents: Dict[Process, EventDistribution] = (
            dict()
        )  # dict from processes to events that occur per-locus
        self._postedEvents: List[PostedEvent] = []  # pri-queue of fixed-time events
        self._postedEventFinder: Dict[int, PostedEvent] = (
            {}
        )  # mapping from event id to event structure
        # self._interactions: Dict[str, List[Tuple[str, InteractionMatrix, str]]] = dict()   # dict from names to interaction specs
        self._interactions: List[Interaction] = []  # list of interactions
        self._conditionals: List[
            Optional[Tuple[float, Process, Element, EventFunction, Condition, Optional[str]]]
        ] = []  # list of conditional events

    # ---------- Configuration ----------

    def process(self) -> Process:
        """Return the network process being run.

        :returns: the process"""
        return self._process

    def currentSimulationTime(self) -> float:
        """Return the current simulation time.

        :returns: the current time"""
        return self._simulationTime

    def setCurrentSimulationTime(self, t: float):
        """Set the current simulation time. This should only be used by
        sub-classes when running the simulation: doing so in any other
        context risks damaging the simulation.

        :param t: the new simualtion time

        """
        self._simulationTime = t

    # ---------- Results ----------

    def experimentalResults(self) -> Dict[str, Any]:
        """Report the process' experimental results. This simply calls through
        to the :meth:`Process.results` method of the process being
        simulated.

        :returns: the results of the process

        """
        return self._process.results()

    # ---------- Set-up and tear-down ----------

    def setUp(self, params: Dict[str, Any]):
        """Set up the experiment for a run. This performs the inherited
        actions, then builds the network process that the dynamics is
        to run.

        :params params: the experimental parameters

        """
        super().setUp(params)

        # set up the event stream
        self._loci = dict()
        self._processLoci = dict()
        self._postedEvents = []
        self._postedEventFinder = dict()
        self._eventId = 0
        self._simulationTime = 0.0
        self._interactions = []

        # custom_interactions = params.get('custom_interactions', dict())
        # if isinstance(custom_interactions, str):
        #     custom_interactions = json.loads(custom_interactions)
        # self._interactions = custom_interactions

        # build and set up the process
        self._process.reset()
        self._process.build(params)
        self._process.setUp(params)

        # can only elucidate interactions once processes built and set up as need compartments list
        # elucidate interactions
        std_interactions = params.get("std_interactions", dict())
        if isinstance(std_interactions, str):
            std_interactions = json.loads(std_interactions)
            for k, v in std_interactions.items():
                if isinstance(v, str):
                    std_interactions[k] = json.loads(v)
        for source, specs in std_interactions.items():
            for spec in specs:
                for target, props in spec.items():
                    preset = props[0]
                    name = props[1]
                    interactions = self.generateInteractions(
                        source, target, name, preset
                    )
                    self._interactions.extend(interactions)

    def tearDown(self):
        """At the end of each experiment, throw away any posted by un-executed
        events.

        """
        self._process.tearDown()
        super().tearDown()

        # discard any remaining posted events
        self._postedEventFinder = {}
        self._postedEvents = []
        self._interactions = {}
        self._conditionals = []

    # ---------- Stochastic events (drawn from a distribution) ----------

    def addLocus(self, p: Process, n: str, l: Optional[Locus] = None) -> Locus:
        """Add a named locus associated with the given process.

        :param p: the process
        :param n: the locus name
        :param l: the locus (defaults to a simple set-based locus)
        :returns: the locus"""
        if n in self._loci.keys():
            raise Exception("Locus {n} already exists in the simulation".format(n=n))

        # store locus by name
        if l is None:
            l = Locus(n)
        self._loci[n] = l

        # update process record
        if p not in self._processLoci:
            # new process, add loci and event lists
            self._processLoci[p] = dict()
            self._perElementEvents[p] = []
            self._perLocusEvents[p] = []
        self._processLoci[p][n] = l

        # link locus to the process that defined it
        l.setProcess(p)

        # return the locus
        return l

    def locus(self, n: str) -> Locus:
        """Retrieve a locus by name.

        :param n: the locus name
        :returns: the locus"""
        return self._loci[n]

    def loci(self) -> Dict[str, Locus]:
        """Return all the loci in the simulation.

        :returns: a dict from names to loci"""
        return self._loci

    def lociForProcess(self, p: Process) -> Dict[str, Locus]:
        """Return all the loci defined for a specific process.

        :param p: the process
        :returns: a dict from names to loci"""
        if p in self._processLoci:
            # process has loci, return them
            return self._processLoci[p]
        else:
            # process doesn't have loci, return an empty dict
            return dict()

    def setLociForProcess(self, p: Process, loci: Dict[str, Locus]):
        """Set the loci for a process.

        :param p: the process
        :param loci: the loci"""
        self._processLoci[p] = loci

    def perElementEventDistribution(self, t: float) -> EventDistribution:
        """Return the distribution of of all processes' per-element events at the given time.

        Note that this method returns the *probability* of events, not their
        expected *rates*, for which use :meth:`perElementEventRateDistribution`.

        :param t: the simulation time
        :returns: a list of (locus, probability, event function) triples"""
        dist = []
        for p in self._process.allProcesses():
            dist.extend(p.perElementEventDistribution(t))
        return dist

    def perElementEventRateDistribution(self, t: float) -> EventDistribution:
        """Return the rates of per-element events for all processes at the given time.

        Note that this is method returns event *rates*, not their *probabilities*
        as returned by :meth:`perElementEventDistribution`. The rate is simply
        an event's probability multiplied by the size of the locus on which it occurs,
        giving the expected number of events occurring from that locus in unit time.

        :param t: the simulation time
        :returns: a list of (locus, rate, event function) triples"""
        dist = []
        for p in self._process.allProcesses():
            dist.extend(p.perElementEventRateDistribution(t))
        return dist

    def fixedRateEventDistribution(self, t: float) -> EventDistribution:
        """Return the distribution of fixed-rate events for all processes at the given time.

        :param t: the simulation time
        :returns: a list of (locus, probability, event function) triples"""
        dist = []
        for p in self._process.allProcesses():
            dist.extend(p.fixedRateEventDistribution(t))
        return dist

    def eventRateDistribution(self, t: float) -> EventDistribution:
        """Return the event distribution, a sequence of (l, r, f, n) tuples
        where l is the locus where the event occurs, r is the rate at
        which an event occurs, f is the event function called to
        make it happen, and n is the event name (which may be None).

        Note the distinction between a *rate* and a *probability*:
        the former can be obtained from the latter simply by
        multiplying the event probability by the number of times it's
        possible in the current network, which for per-element events
        is the population of nodes or edges in a given state.

        It is perfectly fine for an event to have a zero rate. The process
        is assumed to have reached equilibrium if all events have zero rates.

        :param t: current time
        :returns: a list of (locus, rate, event function, event name) tuples"""
        return self.perElementEventRateDistribution(
            t
        ) + self.fixedRateEventDistribution(t)

    # ---------- Posted events (occurring at a fixed time) ----------

    def _nextEventId(self) -> int:
        """Generate a sequence number for a posted event. This is used to ensure that
        all event triples pushed onto the priqueue are unique in their first two elements,
        and therefore never try to do comparisons with functions (the third element).

        :returns: a sequence number"""
        id = self._eventId
        self._eventId += 1
        return id

    def postEvent(
        self,
        t: float,
        p: Process,
        e: Element,
        ef: EventFunction,
        name: Optional[str] = None,
    ) -> int:
        """Post an event that calls the :term:`event function` at time t.
        A unique id it returned that can be used to remove the event
        before it fires using :meth:`unpostEvent`.

        The optional name is used in conjunction with
        :ref:`event taps <event-taps>` when calling :meth:`eventFired`.

        :param t: the current time
        :param p: the process originating the event
        :param e: the element (node or edge) on which the event occurs
        :param ef: the event function
        :param name: (optional) meaningful name of the event
        :returns: the event id

        """
        if t < self.currentSimulationTime():
            ct = self.currentSimulationTime()
            raise ValueError(f"Posting event in the past ({t} < {ct})")

        id = self._nextEventId()
        ev = [t, id, p, (lambda: ef(t, e)), e, name]
        event = cast(PostedEvent, ev)
        self._postedEventFinder[id] = event
        heappush(self._postedEvents, event)
        return id

    def postRepeatingEvent(
        self,
        t: float,
        dt: float,
        p: Process,
        e: Element,
        ef: EventFunction,
        name: Optional[str] = None,
    ):
        """Post an event that starts at time t and re-occurs at interval dt.
        Repeating events can't be removed once posted.

        The optional name is used in conjunction with
        :ref:`event taps <event-taps>` when calling :meth:`eventFired`.

        :param t: the start time
        :param dt: the interval
        :param p: the process originating the event
        :param e: the element (node or edge) on which the event occurs
        :param ef: the element function
        :param name: (optional) meaningful name of the event

        """

        # event function to fire an event and then re-schedule it
        def repeat(tc, e):
            ef(tc, e)
            self.postEvent(tc + dt, p, e, repeat, name)

        self.postEvent(t, p, e, repeat, name)

    def postConditionalEvent(
        self,
        t: float,
        p: Process,
        e: Element,
        ef: EventFunction,
        condition: Condition,
        name: Optional[str] = None,
    ):
        """Post an event that fires only on satisfaction of a condition.
        Conditions are checked at each new simulation time to ensure any changes
        in the previous time step are taken into account before newer events are fired.
        
        :param t: the current time
        :param p: the process originating the event
        :param e: the element (node or edge) on which the event occurs
        :param ef: the event function to be fired when the condition is satisfied
        :param condition: the condition to be satisfied
        :param name: (optional) meaningful name of the event
        """
        self._conditionals.append((t, p, e, ef, condition, name))

    def unpostEvent(self, id: int, fatal: bool = True) -> Optional[float]:
        """Un-post a posted event. This is only legal before the
        event has fired, and will normally raise a KeyError if called on one
        that's not queued: set fatal to False to avoid this.

        :param id: the event id
        :param fatal: whether to raise KeyError for missing events (defaults to True)
        :returns: the simulation time at which the event would have fired, or None"""
        pe = self._postedEventFinder.get(id, None)

        # decide what to do if there's no such event
        if pe is None:
            if fatal:
                raise KeyError(id)
            else:
                return None

        # replace the event function with a dummy
        pe[3] = None

        # remove from the event finder
        del self._postedEventFinder[id]

        # return the time for which the event was scheduled
        return pe[0]

    def pendingEventTime(self, id: int) -> float:
        """Return the time for which the given event is posted for. This will
        raise a KeyError if the event isn't in the queue, thrtough having fired
        or being un-posted.

        :poram id: the event
        :returns: the event's posted simulation time"""
        (et, _, _, _, _, _) = self._postedEventFinder[id]
        return et

    def _discardUnpostedEvents(self):
        """Chew-up and discard any unposted events at the head of the posted
        event queue. After a call to this method the next event on the
        posted event queue (if there is one) is guaranteed to be "real"."""
        while len(self._postedEvents) > 0:
            (_, id, _, ef, _, _) = self._postedEvents[0]
            if ef is None:
                # the head event has been unposted, discard it
                heappop(self._postedEvents)
            else:
                # head event is "real", we're done
                break

    def nextPendingEvent(self) -> Optional[PostedEvent]:
        """Return the next posted event, or `None` if there are no events.

        :returns: the next posted event or None"""
        self._discardUnpostedEvents()
        if len(self._postedEvents) > 0:
            # return the next event
            pe = heappop(self._postedEvents)
            (_, id, _, ef, _, _) = cast(PostedEvent, pe)
            del self._postedEventFinder[id]
            return pe
        else:
            # no posted events
            return None

        # if we get here there are no events remainiong
        # return None

    def nextPendingEventTime(self) -> Optional[float]:
        """Return the simulation time for the next pending posted event, without
        affecting the event queue.

        :returns: the simulation time or None"""
        self._discardUnpostedEvents()
        if len(self._postedEvents) > 0:
            # return the next event time
            (et, _, _, _, _, _) = cast(PostedEvent, self._postedEvents[0])
            return et
        else:
            # no posted events
            return None

    def nextPendingEventBefore(self, t: float) -> Optional[PostedEvent]:
        """Return the next pending event to occur at or before time t.

        :param t: the current time
        :returns: a posted event or None"""
        et = self.nextPendingEventTime()
        if et is None:
            # no posted events remaining
            return None
        elif et <= t:
            # next event should be fired now
            return self.nextPendingEvent()
        else:
            # next event lies after the requested time
            return None

    def runPendingEvents(self, t: float) -> int:
        """Retrieve and fire any pending events at time t. This handles
        the case where firing an event posts another event that needs to be run
        before other already-posted events coming before time t: in other words,
        it ensures that the simulation order is respected.

        :param t: the current time
        :returns: the number of events fired"""
        n = 0
        while True:
            self._checkConditionals(t)
            pe = self.nextPendingEventBefore(t)
            if pe is None:
                # no more pending events, return however many we've fired already
                return n
            else:
                # fire the event
                (et, _, p, pef, e, name) = cast(PostedEvent, pe)
                self.setCurrentSimulationTime(et)  # set the correct time
                if p is None:
                    r = 1.0
                else:
                    if pef is not None:   
                        r = self.getEventSuccessProbability(
                            et, e, pef, name, p.instanceName()
                        )
                        if rng.random() <= r:       
                            pef()
                            self.eventFired(t, p, name, e)
                            n += 1

    def _checkConditionals(self, t: float):
        """Check the status of each conditional held for the current simulation time `t`.
        If the conditional is satisfied the associated event is posted at the current time, 
        and the condition removed. Else, the condition is retained for the next time step.

        :param: t: the current time
        """
        indexesToPop = [] # indexes of conditionals to be removed i.e after being satisfied
        for i in range(len(self._conditionals)):
            conditional = self._conditionals[i]
            if conditional is None:
                continue
            conditional  = cast(Tuple[float, Process, Element, EventFunction, Condition, str], conditional)
            (et, p, e, ef, condition, name) = conditional
            if self._evaluateCondition(t, condition):
                self.postEvent(t, p, e, ef, name) # post the associated event
                indexesToPop.append(i)
        for i in indexesToPop:
            self._conditionals[i] = None
        self._conditionals = [c for c in self._conditionals if c is not None] 

    def _evaluateCondition(self, t: float, condition: Condition) -> bool:
        ctype, target = condition
        if ctype == "equilibrium":
            targetProcess = self._findProcess(target)
            return targetProcess.atEquilibrium(t) and t < self._process.maximumTime()
            # ensure not firing anything after the simulation is supposed to have ended
        else:
            raise NotImplementedError(ctype, "type not yet implemented.")

    def generateInteractions(self, source: str, target: str, name: Optional[str], preset: str) -> List[Interaction]:
        """Given a source and target process, generate the interaction specified in the preset.
        Bi-directional interactions will generate at least two interaction objects, e.g for cross-immunity
        an immunity interaction is generated for both source-target and target-source. If the preset does not exist,
        an exception is raised.

        :param source: the source process
        :param target: the target process
        :param name: the name of the interaction
        :param preset: the preset type of interaction
        :returns: the generated interactions"""

        sourceModel = self._findProcess(source)
        targetModel = self._findProcess(target)
        if preset == "causes_immunity":
            infectEventName = sourceModel.getInfectEventName()
            targetCompartments = targetModel.getInfectedCompartments()
            interactions = Interaction.createInfectionImmunityFor(source, target, infectEventName, targetCompartments)
        elif preset == "must_have_for_infection":
            infectEventName = sourceModel.getInfectEventName()
            targetCompartments = targetModel.getInfectedCompartments()
            interactions = Interaction.createRequiredPreinfectionFor(
                source, target, infectEventName, targetCompartments
            )
        else:
            raise Exception("Preset not found")
        return interactions

    def _findProcess(self, name: str) -> Process:
        """Find the process with the given name. Raises an Exception 
        if not found, as this is assumed to be a fatal error.
        
        :param name: the process name
        :returns: the process"""

        for process in self._process.allProcesses():
            if process.instanceName() == name:
                return process
        raise Exception("Process not found")

    def getEventSuccessProbability(self, t: float, e: Any, ef: PostedEventFunction, name: str, origin: Optional[str]) -> float:
        """Given the event function and the element on which it occurs, return the success probability.
        This is the probability that the event function is actually run and is unrelated to any other measure
        (such as propensity, if using stochastic simulation). This probability is dependent on any rules
        held in the dynamics, such as process interactions, which may impede the event.

        :param t: the current time
        :param e: the element on which the event occurs
        :param ef: the event function
        :param name: the name of the event
        :param origin: the origin process of the event
        :returns: the success probability of the event function

        """

        if e == None:
            return 1 # then not occuring at node/edge; assume can occur
        elif isinstance(e, tuple):
            n1, n2 = e
        else:
            n1 = e
        # important to remember that n1 is the one that is modified NOT n2
        # recall SI edge -> II edge (first node changes)
        n1Data = self.network().nodes[n1]
        # ignore n2data 
        if "history" not in n1Data:
            n1Data["history"] = dict()
        n1History = n1Data["history"]
        for interaction in self.getSourceInteractions(origin, ef.__name__):
            modifier = self.evaluateInteraction(interaction, n1History)
            if modifier >= 0 and modifier <= 1:
                # then valid modifier
                # print("n1Data", n1Data)
                return modifier
            # else continue
        return 1  # fallback

    def getSuitableEmergenceNode(self, model, t):
        """For the particular model, return a node that is suitable for emergence, or None if none exists.
        A target node is deemed suitable if there are no conflicting interactions which would prevent such seeding.
        The current implementation is simple, yet naive. The method will keep attempting to find a node until the network is exhausted,
        which can be problematic for large networks. 

        Subclasses could implement their own method for returning a suitable emergence node.
        
        :param model: the model
        :param t: the current time
        :returns: the node id or None"""

        nodesList = list(self.network().nodes) # as list for copy
        while not len(nodesList) == 0:
            rng.shuffle(nodesList)
            n = int(rng.random() * len(nodesList))
            if (self.getEventSuccessProbability(t, n, model.emerge, "try_emergence", model.instanceName()) > 0):
                # if interaction not outright prohibited then the node is suitable
                return n
            nodesList.pop(n) # else remove the node -- will never be suitable
        return None

    def getSourceInteractions(self, source, efName):
        """Return all interactions attached to the source AND matching the event function name.
        
        :param source: the source process
        :param efName: the event function name
        :returns: a list of interactions"""
        return list(
            filter(
                lambda i: i.getSource() == source and i.getEventName() == efName,
                self._interactions,
            )
        )

    def evaluateInteraction(self, interaction, nodeHistory):
        """Evaluate an interaction against a given node history. If
        the interaction is satisfied, return the associated modifier.
        If not satisfied, return the `otherwise` value, if one exists.
        This method is intended to be used after sufficient validation 
        (i.e matching of source and efName), so these are not checked.
        
        :param interaction: the interaction
        :param nodeHistory: the node history
        :returns: the modifier associated with the interaction or otherwise clause, depending on the truthiness of the interaction"""

        history, timings = nodeHistory[interaction.getTarget()]
        if interaction.isHistorical():
            interactionSatisfied = all(
                c in history for c in interaction.getTargetCompartments()
            )
        else:
            interactionSatisfied = (
                history.getLatestCompartment() in interaction.getTargetCompartments()
            )

        if interactionSatisfied:
            return interaction.getModifier()
        else:
            return interaction.getOtherwise()
