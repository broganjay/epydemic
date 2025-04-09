# Compartmented models base class
#
# Copyright (C) 2017--2021 Simon Dobson
#
# This file is part of epydemic, epidemic network simulations in Python.
#
# epydemic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published byf
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

import math
from itertools import permutations
from networkx import Graph, DiGraph, all_simple_paths
from typing import Dict, Any, List, Tuple, Callable, cast
from epydemic import rng, Locus, Process, Node, Edge, Element, CompartmentHistory

# Helper types
Handlers = Tuple[Callable[[Graph, Element], None],   # add handler
                 Callable[[Graph, Element], None],   # leave handler
                 Callable[[Graph, Element], None],   # enter handler
                 Callable[[Graph, Element], None]]   # remove handler


class CompartmentedLocus(Locus):
    '''A locus based on the compartments that nodes reside in.

    :param name: the locus' name'''

    def __init__(self, name: str):
        super().__init__(name)


    def compartments(self) -> List[str]:
        '''Return the compartments this locus monitors. This should be
        overridden by sub-classes as required.

        :returns: the compartments'''
        return []
    
    def replaceCompartment(self, c: str, new_c: str):
        '''Replace a compartment with a new compartment.

        :param c: the compartment to replace
        :param new_c: the new compartment'''
        pass


class CompartmentedNodeLocus(CompartmentedLocus):
    '''A locus for dynamics occurring at a single node, where the
    locus tracks all nodes in a single compartment.

    :param name: the locus' name
    :param c: the compartment'''

    def __init__(self, name: str, c: str):
        super().__init__(name)
        self._compartment = c


    def compartments(self) -> List[str]:
        '''Return the node compartment we monitor.

        :returns: the compartment'''
        return [self._compartment]


    def addHandler(self, g: Graph, n: Element):
        '''A node is added to the network.

        :param, g: the network
        :param n: the node'''
        if not isinstance(n, tuple):
            super().addHandler(g, n)

    def leaveHandler(self, g: Graph, n: Element):
        '''A node changes compartment.

        :param, g: the network
        :param n: the node'''
        if not isinstance(n, tuple):
            super().leaveHandler(g, n)


    def enterHandler(self, g: Graph, n: Element):
        '''A node enters a compartment.

        :param, g: the network
        :param n: the node'''
        if not isinstance(n, tuple):
            super().enterHandler(g, n)


    def removeHandler(self, g: Graph, n: Element):
        '''A node is removed from the network.

        :param, g: the network
        :param n: the node'''
        if not isinstance(n, tuple):
            super().removeHandler(g, n)

    def replaceCompartment(self, c, new_c):
        '''Replace a compartment with a new compartment.

        :param c: the compartment to replace
        :param new_c: the new compartment'''
        if self._compartment == c:
            self._compartment = new_c


class CompartmentedEdgeLocus(CompartmentedLocus):
    '''A locus for dynamics occurring at an edge, where the locus tracks all
    edges whose endpoint nodes are in specified compartments. Since the network may
    also be adaptive, we need to track additions and removals of edges too.

    :param name: the locus' name
    :param l: the left compartment
    :param r: the right compartment'''

    def __init__(self, name: str, l: str, r: str):
        super().__init__(name)
        self._left = l
        self._right = r


    def compartments(self) -> List[str]:
        '''Return the compartments of the node endpoints we monitor.

        :returns: the compartments'''
        return [self._left, self._right]


    def matches(self, g: Graph, n: Node, m: Node) -> int:
        '''Test whether the given edge has the right compartment endpoints for this compartment. The
        method returns 1 if the edge has the right compartments in the orientation (n, m), -1 if it has
        the right compartments in orientation (m, n), and 0 otherwise.

        :param g: the network
        :param n: the first node
        :param m: the second node
        :returns: match status -1, 0, or 1'''
        p = cast(CompartmentedModel, self.process())
        cn = p.getCompartment(n)
        cm = p.getCompartment(m)
        if (cn == self._right) and (cm == self._left):
            return -1
        else:
            if (cn == self._left) and (cm == self._right):
                return 1
            else:
                return 0


    def addHandler(self, g: Graph, e: Edge):
        '''An edge is added to the network, check if its endpoint compartments match the
        locus and add it if so.

        :param, g: the network
        :param e: the edge'''
        if isinstance(e, tuple):
            (n, m) = e
            match = self.matches(g, n, m)
            if match == -1:
                #print('edge ({m}, {n}) added to {l}'.format(n = n, m = m, l = self._name))
                self.add((m, n))
            else:
                if match == 1:
                    #print('edge ({n}, {m}) added {l}'.format(n = n, m = m, l = self._name))
                    self.add((n, m))


    def leaveHandler(self, g: Graph, n: Node):
        '''Node leaves one of the edge's compartments, remove any incident edges
        that no longer have the correct orientation.

        :param g: the network
        :param n: the node'''
        for (nn, mm) in g.edges(n):
            match = self.matches(g, nn, mm)
            if match == -1:
                #print('edge ({m}, {n}) leaves {l}'.format(n = nn, m = mm, l = self._name))
                self.discard((mm, nn))
            else:
                if match == 1:
                    #print('edge ({n}, {m}) leaves {l}'.format(n = nn, m = mm, l = self._name))
                    self.discard((nn, mm))


    def enterHandler(self, g: Graph, n: Node):
        '''Node enters one of the edge's compartments, add any incident edges
        that now have the correct orientation.

        :param g: the network+
        :param n: the node'''
        for (nn, mm) in g.edges(n):
            match = self.matches(g, nn, mm)
            if match == -1:
                #print('edge ({m}, {n}) enters {l}'.format(n = nn, m = mm, l = self._name))
                self.add((mm, nn))
            else:
                if match == 1:
                    #print('edge ({n}, {m}) enters {l}'.format(n = nn, m = mm, l = self._name))
                    self.add((nn, mm))


    def removeHandler(self, g: Graph, e: Edge):
        '''An edge is removed from the network, check whether it was in this locus and
        remove it if so.

        :param, g: the network
        :param e: the edge'''
        if isinstance(e, tuple):
            (n, m) = e
            match = self.matches(g, n, m)
            if match == -1:
                #print('edge ({m}, {n}) removed from {l}'.format(n = n, m = m, l = self._name))
                self.discard((m, n))
            else:
                if match == 1:
                    #print('edge ({n}, {m}) removed from {l}'.format(n = n, m = m, l = self._name))
                    self.discard((n, m))

    def replaceCompartment(self, c, new_c):
        '''Replace a compartment with a new compartment.

        :param c: the compartment to replace
        :param new_c: the new compartment'''
        if self._left == c:
            self._left = new_c
        if self._right == c:
            self._right = new_c


class CompartmentedModel(Process):
    '''The base class a :term:`compartmented model of disease`, which
    represents a disease as a collection of discrete :term:`compartments` with each
    compartment specifying some facet of the disease's progression. Nodes
    transition between compartments with some probability, so the disease's
    progression forms a stochastic process in which nodes typically move
    through several compartments according to the probabilities defined.

    When run, a compartmented model generates results consisting of the size of each
    compartment at the end of the experiment. This can be changed by overriding the
    :meth:`results` method.

    :param: name (optional) instance name'''

    # Placeholders for model state variables
    COMPARTMENT: str = None               #: State variable holding a node's compartment.
    OCCUPIED: str = None                  #: State variable that's True for occupied edges.

    # Shared state variables
    T_OCCUPIED: str = "tOccupied"                 #: State variable holding the occupation time of an edge.
    T_HITTING: str = "tHitting"                   #: State variable holding the infection time of a node.
    HITTING_PROCESS_NAME: str = "hittingProcess"  #: State variable holding the name of the infecing process.

    START_COMPARTMENT: str = "startCompartment"  
    EMERGENCE_CONDITION: str = "emergenceCondition" 
    EMERGENCE_TARGET: str = "emergenceTarget"

    def __init__(self, name: str = None):
        super().__init__(name)
        self._compartments: Dict[str, float] = dict()         # compartment -> initial probability
        self._effects: Dict[str, List[Handlers]] = dict()     # compartment -> event handlers

        # state variable unique tags
        self.COMPARTMENT = self.stateVariable('compartment')
        self.OCCUPIED = self.stateVariable('occupied')



    # ---------- Setup and initialisation ----------

    def reset(self):
        '''Reset the model ready to be built.'''
        super().reset()
        self._compartments = dict()
        self._effects = dict()

    def emerge(self, t, n):
        if self._emerged:
            # then already emerged
            return
        # default emergence behaviour is to randomly seed a node on the network
        # n = int(rng.random() * len(self.network().nodes))
        en = self.getSuitableEmergenceNode(t)
        if en == None:
            # then failed to emerge
            return
        else:
            # print("emerging at en, cs, t =", t, self.network().nodes[en])
            # inRemoved = len(self._dynamics._findProcess("d1").compartment("epydemic.sir.I@d1"))
            # # if inRemoved > TODO HERE
            self.postEvent(t, en, self.infect, "spontaneous_infection")
            self._emerged = True

    def setUp(self, params: Dict[str, Any]):
        '''Set up the initial population of nodes into compartments.

        :param params: the simulation parameters'''
        super().setUp(params)

        # initialise all nodes to an empty compartment
        # (so we can assume all nodes have a compartment attribute)
        g = self.network()
        for n in g.nodes():
            g.nodes[n][self.COMPARTMENT] = None

        # mark edges as unoccupied
        for (_, _, data) in g.edges(data=True):
            data[self.OCCUPIED] = False
        
        # self.decorateAllNames()
        # place nodes in initial compartments
        try:
            [emergenceCondition] = self.getParameters(params, [self.EMERGENCE_CONDITION])
            [emergenceTarget] = self.getParameters(params, [self.EMERGENCE_TARGET])
        except KeyError:
            emergenceCondition = None
            emergenceTarget = None
        if emergenceCondition is not None:
            self._emerged = False
            [startCompartment] = self.getParameters(params, [self.START_COMPARTMENT])
            for c in self._compartments:
                if c != startCompartment:
                    self.changeCompartmentInitialOccupancy(c, 0.0)
            self.changeCompartmentInitialOccupancy(startCompartment, 1.0) 
            self.params = params
            self.postConditionalEvent(0, None, self.emerge, (emergenceCondition, emergenceTarget), "emergence")

        self.initialCompartments()


    def initialCompartmentDistribution(self) -> List[Tuple[str, float]]:
        '''Return the initial distribution of nodes to compartments. The
        result should be a valid distribution, with probabilities summing
        to one. (A ``ValueError`` exception is raised if not.) The
        distribution is used by :meth:`initialCompartments` to set the initial
        compartment of each node.

        :returns: a list of (compartment, probability) pairs'''
        dist = []
        for cp in self._compartments.items():
            dist.append(cp)

        # sanity-check the distribution
        a = 0.0
        for (_, p) in dist:
            a += p

        if not math.isclose(a, 1.0):
            raise ValueError('Bad initial compartment distribution (probabilities of model don\'t sum to one)')

        return dist


    def initialCompartments(self):
        '''Place each node in the network into its initial compartment. The default
        initialises the nodes into a random compartment according to the initial
        compartment distribution returned by :meth:`initialCompartmentDistribution`.
        This method may be overridden to, for example, structure the initial
        population non-randomly.'''

        # get the initial compartment distribution
        dist = self.initialCompartmentDistribution()

        # assign nodes to compartments
        g = self.network()
        for n in g.nodes():
            # select a compartment according to the initial distribution
            r = rng.random()
            a = 0.0
            for (c, p) in dist:
                a += p
                if r <= a:
                    # change node's compartment
                    self.changeInitialCompartment(n, c)
                    # on to the next node
                    break


    def changeInitialCompartment(self, n: Node, c: str):
        '''Change the initial compartment of a node. This is called only from
        :meth:`initialCompartments`, and by default simply calls
        :meth:`changeCompartment`. Overriding this method changes the behaviour
        of assigning initial compartments independently of the behaviour
        of later changes.

        :param n: the node
        :param c: the new compartment for the node'''
        self.changeCompartment(n, c)


    # ---------- Termination and results ----------

    def compartments(self) -> List[str]:
        '''Return the set of compartments for this model.

        :returns: the compartments'''
        return list(self._compartments.keys())


    def compartment(self, c: str) -> List[Any]:
        '''Return all the nodes currently in a particular compartment in a
        network. This works for all compartments, not just those that
        are loci for dynamics -- but is a *lot* slower, so it's better
        to create a :term:`locus` is you're going to access a
        compartment frequently.

        :param c: the compartment
        :returns: a collection of nodes

        '''
        return [n for n in self.network().nodes() if self.getCompartment(n) == c]


    def results(self) -> Dict[str, Any]:
        '''Create a dict of experimental results for the experiment, consisting of the final
        sizes of all the compartments.

        :returns: a dict of experimental results'''
        rc = super().results()

        # add size of each compartment
        for c in self.compartments():
            rc[c] = len(self.compartment(c))
        return rc


    def skeletonise(self) -> Graph:
        '''Remove unoccupied edges from the network. This leaves the network
        consisting of only "occupied" edges that were used to transmit the
        infection between nodes, also known as the :term:`contact tree`. Note
        that this process means that further dynamics over the network probably
        don't make sense, unless you're actually wanting to run on the residual
        network post-infection.

        :returns: the network with unoccupied edges removed'''

        # find all unoccupied edges
        g = self.network()
        edges = []
        for (n, m, data) in g.edges(data=True):
            if (self.OCCUPIED not in data.keys()) or (not data[self.OCCUPIED]):
                # edge is unoccupied, mark it to be removed
                # (safe because there are no parallel edges)
                edges.insert(0, (n, m))

        # remove all the unoccupied edges
        g.remove_edges_from(edges)

        return g

    def inverseSkeletonise(self) -> Graph:
        # more complicated as must also remove nodes that have been hit
        g = self.network()
        edges = []
        for (n, m, data) in g.edges(data=True):
            if (data[self.OCCUPIED]):
                # edge is occupied, mark it to be removed
                # (safe because there are no parallel edges)
                edges.insert(0, (n, m))
        
        # remove all the occupied edges
        g.remove_edges_from(edges)

        # remove all nodes that have been hit
        nodes = []
        for (n, data) in g.nodes(data=True):
            if self.T_HITTING in data.keys():
                nodes.insert(0, n) 

        g.remove_nodes_from(nodes)

        return g


    # ---------- Managing compartments ----------

    def addCompartment(self, c: str, p: float = 0.0):
        '''Add a compartment to the model. A node is assigned to the
        compartment initially with the given probability. The
        probabilities for all compartments in the model must sum to 1.

        :param c: the compartment name
        :param p: the initial occupancy probability (defaults to  0.0)'''
        self._compartments[c] = p


    def changeCompartmentInitialOccupancy(self, c: str, p: float):
        '''Change the initial occupancy probability for a compartment. This
        method is used when sub-classing an existing model: it only
        makes sense during the build process (see :meth:`build`)
        before the model is initialised in :meth:`setUp`.

        :param c: the compartment
        :param p: the new initial occupation probability'''
        if c not in self.compartments():
            raise Exception('Compartment {c} not defined in model'.format(c=c))
        self._compartments[c] = p


    def trackNodesInCompartment(self, c: str, name: str = None):
        '''Add a locus tracking nodes in a given compartment.

        :param c: the compartment to track
        :param name: (optional) the name of the locus (defaults to the compartment name)
        :returns: the locus used to track the nodes'''
        if name is None:
            name = c

        # name = self.decoratedNameInInstance(name)

        # add locus
        locus = CompartmentedNodeLocus(name, c)
        return self.addLocus(name, locus)


    def trackEdgesBetweenCompartments(self, l: str, r: str, name: str = None):
        '''Add a locus to track edges with endpoint nodes in the given compartments.

        :param l: the compartment of the left node
        :param r: the compartment of the right node
        :param name: (optional) the name of the locus (defaults to a combination of the two compartment names)
        :returns: the locus used to track the nodes'''

        if name is None:
            name = f'{l}-{r}'

        # add locus
        locus = CompartmentedEdgeLocus(name, l, r)
        return self.addLocus(name, locus)


    def addLocus(self, n: str, l: Locus = None) -> Locus:
        '''Add a locus to the model, initialising the handler functions.

        :param n: the name
        :param l: (optional) the locus
        :returns: the locus'''
        locus = super().addLocus(n, l)
        # if we've added a compartmented locus, add handler functions for
        # when its population changes
        if isinstance(locus, CompartmentedLocus):
            for c in locus.compartments():
                if c not in self._effects.keys():
                    self._effects[c] = []
                self._effects[c].append((locus.addHandler, locus.leaveHandler, locus.enterHandler, locus.removeHandler))

        # return the locus
        return locus


    def _handlerCompartments(self, e: Element) -> List[str]:
        '''Return the compartments that a given element's change might affect.

        :param e: a node or edge
        :returns: a list of compartments'''
        if isinstance(e, tuple):
            # element is an edge, check compartments at the endpoints
            (n, m) = e
            cs = [self.getCompartment(n), self.getCompartment(m)]
        else:
            # element is a node, check its own compartment
            cs = [self.getCompartment(e)]
        return cs


    def _callAddHandlers(self, e: Element):
        '''Call all handlers affected by a node or edge being added to the network.

        :param e the node or edge'''
        g = self.network()
        for c in self._handlerCompartments(e):
            if c in self._effects.keys():
                for (ah, _, _, _) in self._effects[c]:
                    ah(g, e)


    def _callLeaveHandlers(self, e: Element, c: str):
        '''Call all handlers affected by a node or edge leaving a compartment.

        :param e: the node or edge
        :param c: the compartment'''
        g = self.network()
        for c in self._handlerCompartments(e):
            if c in self._effects.keys():
                for (_, lh, _, _) in self._effects[c]:
                    lh(g, e)


    def _callEnterHandlers(self, e: Element, c: str):
        '''Call all handlers affected by a node or edge entering a compartment.

        :param e: the node or edge
        :param c: the compartment'''
        g = self.network()
        for c in self._handlerCompartments(e):
            if c in self._effects.keys():
                for (_, _, eh, _) in self._effects[c]:
                    eh(g, e)


    def _callRemoveHandlers(self, e: Element):
        '''Call all handlers affected by a node or edge being removed from the network.

        :param e: the node or edge'''
        g = self.network()
        for c in self._handlerCompartments(e):
            if c in self._effects.keys():
                for (_, _, _, rh) in self._effects[c]:
                    rh(g, e)


    # ---------- Accessing and evolving the network ----------

    def setCompartment(self, n: Node, c: str):
        '''Set the compartment of a node. This assumes that the node doesn't
        already have a compartment set, and so should be used only for
        initialising new nodes: in all other cases, use
        :meth:`changeCompartment`.

        :param n: the node
        :param c: the new compartment for the node'''
        g = self.network()

        # set the correct node attribute
        g.nodes[n][self.COMPARTMENT] = c

        # propagate the change to any other compartments
        self._callEnterHandlers(n, c)


    def getCompartment(self, n: Node) -> str:
        '''Return the compartment of a node.

        :param n: the node
        :returns: its compartment'''
        return self.network().nodes[n][self.COMPARTMENT]


    def changeCompartment(self, n: Node, c: str):
        '''Change the compartment of a node.

        :param n: the node
        :param c: the new compartment for the node'''
        g = self.network()
        oc = g.nodes[n][self.COMPARTMENT]

        # record oc in disease history

        # first check if history exists 
        if 'history' not in g.nodes[n]:
            g.nodes[n]['history'] = {}
        
        # then check if the instance has a history entry
        if self.instanceName() not in g.nodes[n]['history']:
            g.nodes[n]['history'][self.instanceName()] = []
            timings = []
            oldHistory = CompartmentHistory()
        else:
            # tuple of format (template, [t1, t2, ...])
            timings = g.nodes[n]['history'][self.instanceName()][1]
            oldHistory = g.nodes[n]['history'][self.instanceName()][0]

        # add new compartment to history -- assume old has already been added
        # (this is a fair assumption, as even initial compartments method uses this)

        timings.append(self.currentSimulationTime())

        newHistory = CompartmentHistory.updateHistory(oldHistory, c)

        g.nodes[n]['history'][self.instanceName()] = (newHistory, timings)

        # propagate effects of leaving the current compartment
        if oc is not None:
            self._callLeaveHandlers(n, oc)

        # record new compartment on node
        g.nodes[n][self.COMPARTMENT] = c

        # propagate effects of entering new compartment
        self._callEnterHandlers(n, c)



    def markOccupied(self, e: Edge, t: float, firstOnly: bool = True):
        '''Mark the given edge as having been occupied by the dynamics, i.e., to
        have been traversed in transmitting the disease, at time t. By default
        the time of first occupation is recorded: if firstOnly is False then
        subsequent occupations overwrite the ealier ones.

        :param e: the edge
        :param t: the simulation time at which it was occupied
        :param firstOnly: (optional) only record the first occupation time (defaults to True)'''
        g = self.network()
        (n, m) = e
        data = g.get_edge_data(n, m)
        if (not firstOnly) or (not data.get(self.OCCUPIED, False)):
            data[self.OCCUPIED] = True
            data[self.T_OCCUPIED] = t


    def markHit(self, n: Node, t: float, firstOnly: bool = True):
        '''Mark the node as "hit", which happens when the epidemic first
        reaches the node. By default the first hitting time is
        recorded: if firstOnly is False then subsequent infections
        overwrite the ealier ones. If the process has an instance name
        that is stored as well, to indicate which instance hit the node.

        For :class:`SIR` the hitting time is the time of infection;
        for :class:`SIS` the time of first infection; for
        :class:`SEIR` the time of exposure.

        :param n: the node
        :param t: the simulation time
        :param firstOnly: (optional) only record the first hitting time (defaults to True)

        '''
        g = self.network()
        if (not firstOnly) or self.T_HITTING not in g.nodes[n]:
            # store the hitting time
            g.nodes[n][self.T_HITTING] = t

            # store the hitting instance if it's named
            name = self.instanceName()
            if name is not None:
                g.nodes[n][self.HITTING_PROCESS_NAME] = name


    def addNode(self, n: Node, c: str = None, **kwds):
        '''Add a node to the working network, adding it to the appropriate compartment
        if one is provided].

        :param n: the new node
        :param c: (optional) compartment for the node
        :param kwds: (optional) node attributes'''
        super().addNode(n, **kwds)
        if c is not None:
            self.setCompartment(n, c)


    def removeNode(self, n: Node):
        '''Remove a node from the working network, updating any affected compartments.

        :param n: the node'''

        # remove node from any loci, and from its compartment
        self._callRemoveHandlers(n)

        # remove the node
        super().removeNode(n)


    def addEdge(self, n: Node, m: Node, **kwds):
        '''Add an edge between nodes, adding the edge to any appropriate compartments.

        :param n: the start node
        :param m: the end node
        :param kwds: (optional) edge attributes'''
        super().addEdge(n, m, **kwds)

        # add edge to any compartments it should be in
        self._callAddHandlers((n, m))


    def removeEdge(self, n: Node, m: Node):
        '''Remove an edge from the working network and from any compartments.

        :param n: the start node
        :param m: the end node'''

        # remove edge from any compartments it should be in
        self._callRemoveHandlers((n, m))

        # remove the edge from the network
        super().removeEdge(n, m)
    
    def effects(self):
        '''Return the effects of the model.

        :returns: the effects'''
        return self._effects

    def compartmentChangeEvent(self, t: float, e: Any, compartment: str):
        if isinstance(e, tuple):
            (n, _) = e
            self.markOccupied(e, t, firstOnly=True)
        else:
            n = e
        self.markHit(n, t, firstOnly=True)
        self.changeCompartment(n, compartment)

    def decorateAllNames(self):
        # decorate compartments
        newCompartments = {}
        for c, f in self._compartments.items():
            newCompartments[self.decoratedName(c)] = f
            if c in self._effects.keys():
                copy = self._effects[c]
                del self._effects[c]
                self._effects[self.decoratedName(c)] = copy
            for name, locus in self.loci().items():
                if c in locus.compartments():
                    locus.replaceCompartment(c, self.decoratedName(c))
        self._compartments = newCompartments

    def getSuitableEmergenceNode(self, t):
        return self._dynamics.getSuitableEmergenceNode(self, t)

    def getPossibleCompartmentTransitions(self):
        '''Return all possible compartment transitions, used in the construction of
        disease templates. The default implementation returns an empty list.

        :returns: a list of compartment transitions'''
        return []
    
    def generateTransitionGraph(self):

        transitionGraph = DiGraph()

        for c in self.compartments():
            transitionGraph.add_node(c)

        for (c1, c2) in self.getPossibleCompartmentTransitions():
            transitionGraph.add_edge(c1, c2)

        return transitionGraph
    
    def generatePossibleHistoriesTemplates(self):

        transitionGraph = self.generateTransitionGraph()

        allTransitions = []

        for startC, targetC in permutations(transitionGraph.nodes, 2):
            paths = list(all_simple_paths(transitionGraph, startC, targetC))
            allTransitions.extend(list(map(tuple, paths)))
        
        return list(map(lambda t: CompartmentHistory(t), allTransitions))
        

