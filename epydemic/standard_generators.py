# Standard generators
#
# Copyright (C) 2017--2020 Simon Dobson
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
from epydemic import NetworkGenerator
from networkx import (
    Graph,
    fast_gnp_random_graph,
    barabasi_albert_graph,
    configuration_model,
)
import sys
import numpy as np

if sys.version_info >= (3, 8):
    from typing import Any, Dict, Optional, Final, Union
else:
    # backport compatibility with older typing
    from typing import Any, Dict, Optional
    from typing_extensions import Final

from epydemic import rng


class FixedNetwork(NetworkGenerator):
    """A network generator that always returns a copy of the same network.

    :param g: the prototype network
    :param limit: (optional) maximum number of identical instances to generate"""

    def __init__(self, g: Graph, limit: Optional[int] = None):
        super().__init__(limit=limit)
        self._graphPrototype: Graph = g

    def topology(self) -> str:
        """Return the topoology flag for this generator.

        :returns: the topology marker ("Arbitrary")"""
        return "Arbitrary"

    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Return a copy of the prototype network.

        :param params: experimental parameters (ignored)
        :returns: a network instance"""
        return self._graphPrototype.copy()


class ERNetwork(NetworkGenerator):
    """Generate Erdos-Renyi (ER) networks from a given order (:attr:`N`) and one of an edge occupation
    probability (:attr:`PHI`) or a mean degree (:attr:`KMEAN`). These parameters are taken from the
    experimental parameters.

    An ER network has nodes with Poisson-distributed independent degrees. The construction process
    can be thought of as taking a set of :math:`N` nodes and then adding an edge between every pair
    with independent probability :math:`\\phi`. This process gives a discrete normal (Poisson) distribution
    of the node degrees with mean degree :math:`\\langle k \\rangle = N \\phi`. The node degrees will
    be uncorrelated.

    The actual construction of ER networks uses the `networkx.fast_gnp_random_graph()` function.

    :param params: (optional) experiment parameters
    :param limit: (optional) meximum  number of instances to generate"""

    # Experimental parameters
    N: Final[str] = "N"  #: Experimental parameter for the size (order) of the network.
    PHI: Final[str] = (
        "phi"  #: Experimental parameter for the occupation probability of edges.
    )
    KMEAN: Final[str] = (
        "kmean"  #: Experimental parameter for the mean degree of the network.
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        super().__init__(params, limit)

    def topology(self) -> str:
        """Return the topoology flag for this generator.

        :returns: the topology marker ("ER")"""
        return "ER"

    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Generate an ER network from an order (represented by the parameter :attr:`N`)
        and one of an edge occupation probability (:attr:`PHI`) or mean degree (:attr:`KMEAN`).

        :param params: experimental parameters
        :returns: the ER network"""

        # extract the parameters
        N = params[self.N]
        if self.PHI in params:
            phi = params[self.PHI]
        elif self.KMEAN in params:
            kmean = params[self.KMEAN]
            phi = (kmean + 0.0) / N
        else:
            raise AttributeError("Need one of occupation probability or mean degree")

        # build the network
        g = fast_gnp_random_graph(N, phi)
        return Graph(g)


class BANetwork(NetworkGenerator):
    """Generate Barabasi-Albert (BA) networks from a given order (:attr:`N`) and rate of attachment
    (:attr:`M`) taken from the experimental parameters.

    A BA network has node degrees distributed according to a powerlaw distribution. The construction process
    can be thought of as taking an initial set of :math:`M` nodes and then adding additional nodes
    one at a time, adding adges between the new node and :math:`M` other nodes chosen at random. This
    continues until the network has :math:`N` nodes. This process favours attachment to nodes that are
    in the network early, leading to "hubs" with very high degree. The node degrees will
    be uncorrelated.

    The actual construction of BA networks uses the `networkx.barabasi_albert_graph()` function.

    :param params: (optional) experiment parameters
    :param limit: (optional) meximum  number of instances to generate"""

    # Experimental parameters
    N: Final[str] = "N"  #: Experimental parameter for the size (order) of the network.
    M: Final[str] = (
        "MperNode"  #: Experimental parameter for the number of edges added per node.
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        super().__init__(params, limit)

    def topology(self) -> str:
        """Return the topoology flag for this generator.

        :returns: the topology marker ("BA")"""
        return "BA"

    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Generate a BA network from an order (represented by the parameter :attr:`N`)
        and attachment rate (:attr:`M`).

        :param params: experimental parameters
        :returns: the BA network"""

        # extract the parameters
        N = params[self.N]
        M = params[self.M]

        # build the network
        g = barabasi_albert_graph(N, M)
        return Graph(g)


class ConfigurationModel(NetworkGenerator):
    """Generate networks from a given order (:attr:`N`) and distribution of degrees (:attr:`DIST`) of length :attr:`N`.
    These parameters are taken from the experimental parameters.

    The configuration model is a basic form of random graph. Edges are created between nodes randomly such that each node has degree
    according to the given degree sequence. 

    The actual construction of configuration model networks uses the `networkx.configuration_model()` function.
    Any self-loops or multi-edges are removed from the generated network by forceful cast to a standard `networkx.Graph` type.

    :param params: (optional) experiment parameters
    :param limit: (optional) maximum number of instances to generate"""


    N: Final[str] = "N"  #: Experimental parameter for the size (order) of the network.
    DIST: Final[str] = "dist"  #: Experimental parameter for the degree distribution.

    def __init__(self, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        super().__init__(params, limit)

    def topology(self) -> str:
        """Return the topology flag for this generator.

        :returns: the topology marker ("CM")"""
        return "CM"

    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Generate a configuration model network from an order (represented by the parameter :attr:`N`)
        and a degree distribution (:attr:`DIST`).

        :param params: experimental parameters
        :returns: the configuration model network"""
        N = params[self.N]
        dist = params[self.DIST]
        if isinstance(dist, str):
            dist = json.loads(dist)
        g = configuration_model(dist)
        return Graph(g) # cast to remove self-loops and multi-edges


class ClusteredNetwork(NetworkGenerator):
    """Generate clustered networks from a given order (:attr:`N`), mean single-edge degree (:attr:`TREEMEAN`) 
    and mean triangle participation (:attr:`TRIMEAN`). These parameters are taken from the experimental parameters.

    The construction of a `ClusteredNetwork` follows the process of the generalised configuration model. Each 
    vertex in the network is assigned a number of tree stubs and triangle stubs according to the given parameters.
    Tree stubs are paired to form uncorrelated edges, and three triangle stubs are connected to create triangles.
    Thus, the resulting network has notable meso-structures of order 3.
    
    :param params: (optional) experiment parameters
    :param limit: (optional) maximum number of instances to generate"""

    N: Final[str] = "N"  #: Experimental parameter for the size (order) of the network.
    TREEMEAN: Final[str] = (
        "treeMean"  #: Experimental parameter for the mean single-edge degree
    )
    TRIMEAN: Final[str] = (
        "triMean"  #: Experimental parameter for the mean triangle participation
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        super().__init__(params, limit)

    def topology(self) -> str:
        """Return the topology flag for this generator.

        :returns: the topology marker ("GCM")"""
        return "GCM"

    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Generate a clustered network from an order (represented by the parameter :attr:`N`),
        mean single-edge degree (:attr:`TREEMEAN`) and mean triangle participation (:attr:`TRIMEAN`).
        :param params: experimental parameters
        :returns: the clustered network"""
        # extract params
        N = params[self.N]
        treeMean = params[self.TREEMEAN]
        triMean = params[self.TRIMEAN]
        g: Graph = Graph()
        g.add_nodes_from(range(N))

        kTree = rng.poisson(treeMean, N) # poisson dist of number of tree stubs per node
        kTri = rng.poisson(triMean, N) # poisson dist of number of triangle stubs per node

        totalTri = sum(kTri) # total number of triangle stubs -- must be divisible by 3
        
        # choose randomly which triangle stubs to remove for each over a clean multiple of 3
        indices = rng.choice(len(kTri), totalTri % 3, replace=False)
        kTri[indices] -= 1 # remove one stub from each

        treeStubs = []
        triStubs = []

        for i in range(N):
            # each node is added once per stub 
            treeStubs.extend([i] * kTree[i]) 
            triStubs.extend([i] * kTri[i])

        remainder = len(treeStubs) % 2 # remainder should be 0 -- must be even number of tree stubs for pairing
        if remainder != 0:
            # if not even, add another random node to the list 
            # artificially modifies network structure (introducing bias), but 
            # effects are mitigated beyond trivially small networks
            treeStubs.extend(rng.choice(range(N), 2 - remainder)) 

        remainder = len(triStubs) % 3
        if remainder != 0:
            # and again for triangles -- adding to make it a multiple of 3 
            triStubs.extend(rng.choice(range(N), 3 - remainder))

        # shuffle the stubs before connecting
        rng.shuffle(triStubs)
        rng.shuffle(treeStubs)

        for i in range(0, len(treeStubs), 2): 
            u, v = treeStubs[i], treeStubs[i + 1] 
            if u != v: # avoid self-connection!
                g.add_edge(u, v)
            
        for i in range(0, len(triStubs), 3):
            u, v, w = triStubs[i], triStubs[i + 1], triStubs[i + 2]
            if u != v and u != w and v != w: # all-diff
                g.add_edge(u, v)
                g.add_edge(v, w)
                g.add_edge(w, u)
            # else, do nothing... just skipped a triangle
            # computationally more viable but less likely to produce graph with
            # the desired mean triangles; for large N _may_ be neglible, alternative
            # could be to reshuffle all remaining tri stubs (including u, v, w) to avoid collision
            # but would require some other arb fallback (e.g max attempts before skipping triangle) to avoid
            # (unlikely, but possible) situation where last three stubs are all the same node 
        return g
