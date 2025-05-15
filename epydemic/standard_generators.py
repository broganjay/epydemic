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
from typing import List, Tuple
from epydemic import NetworkGenerator, rng
from networkx import (
    Graph,
    MultiGraph,
    fast_gnp_random_graph,
    barabasi_albert_graph,
    configuration_model,
    disjoint_union_all
)
import sys
import numpy as np

if sys.version_info >= (3, 8):
    from typing import Any, Dict, Optional, Final, Union
else:
    # backport compatibility with older typing
    from typing import Any, Dict, Optional
    from typing_extensions import Final
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
    FULL_DIST: Final[str] = "fullDist"  #: Experimental parameter for the degree distribution in its entirety.
    DIST_TYPE : Final[str] = "distType"  #: Experimental parameter for the type of degree distribution.
    KMEAN: Final[str] = "kmean"  #: Experimental parameter for the mean degree of the network, supplied when distribution requires it

    POISSON: Final[str] = "poisson"  #: Type of degree distribution, Poisson

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
        if self.FULL_DIST in params:
            # then full dist is given, so just use it every time
            fullDist = params[self.FULL_DIST]
            if isinstance(fullDist, str):
                # then it is a string, so parse it
                fullDist = json.loads(fullDist)
                # no need for N, just take length of fullDist
        elif params[self.DIST_TYPE] == self.POISSON:
            # then create Poisson deg distribution with supplied kmean
            kmean = params[self.KMEAN]
            N = params[self.N]
            fullDist = rng.poisson(kmean, N)
        else:
            raise KeyError('"fullDist" or "distType" not in params or not valid')
    
        if sum(fullDist) % 2 != 0:
            # then the sum of the degree distribution is odd, so add an extra stub somewhere
            # (some bias, but mitigated for large N)
            fullDist[rng.choice(len(fullDist))] += 1
        g = configuration_model(fullDist)
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

        # totalTri = sum(kTri) # total number of triangle stubs -- must be divisible by 3
        
        # # choose randomly which triangle stubs to remove for each over a clean multiple of 3
        # indices = rng.choice(len(kTri), totalTri % 3, replace=False)
        # kTri[indices] -= 1 # remove one stub from each

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

        while len(treeStubs) >= 2:
            retries = 0
            success = False

            while retries < 10:
                rng.shuffle(treeStubs)
                u, v = treeStubs[0], treeStubs[1]

                if u != v:
                    g.add_edge(u, v)
                    del treeStubs[0:2]
                    success = True
                    break

                retries += 1

            if not success:
                del treeStubs[0:2]

            
        while len(triStubs) >= 3:
            retries = 0
            success = False

            while retries < 10:
                rng.shuffle(triStubs)
                u, v, w = triStubs[0], triStubs[1], triStubs[2]

                if len({u, v, w}) == 3:
                    g.add_edge(u, v)
                    g.add_edge(v, w)
                    g.add_edge(w, u)
                    del triStubs[0:3]
                    success = True
                    break

                retries += 1

            if not success:
                # print("SKIPPED TRIANGLE")
                del triStubs[0:3]

        return g

class MultilayerNetwork(NetworkGenerator):
    """Generate a multilayer network from a dictionary of layer name and the underlying network generator.
    Each generator is called individually and passed the same parameters, meaning that the parameters dictionary
    must contain the necessary parameters for each generator. Parameters may be decorated with the layer name to 
    allow the use of the same generator with different parameters across layers. This generator is likely more 
    useful in custom experiments, which can specify inter-layer edges if any exist. This generator returns a 
    `networkx.Graph` object with the layers as node attributes. Fundamentally, then, these multi-layer networks
    are not much different to the other network types and are merely a convenience.

    A multilayer network is a network with multiple layers, each of which can have its own topology.
    The layers can connected by inter-layer edges.

    :param params: (optional) experiment parameters
    :param limit: (optional) maximum number of instances to generate"""

    LAYERS: Final[str] = "layers"  #: Experimental parameter for the layers of the network (name, generator).
    INTERMEAN: Final[str] = "interMean"  #: Experimental parameter for the mean inter-layer edge participation.

    def __init__(self, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        super().__init__(params, limit)
        self._layers: Dict[str, NetworkGenerator] = {}

    def topology(self) -> str:
        """Return the topology flag for this generator.

        :returns: the topology marker ("ML")"""
        return "ML"
    
    def _generate(self, params: Dict[str, Any]) -> Graph:
        """Generate a multilayer network from a list of tuples (or lists) of layer name and underlying network generator (or generator identifier).

        :param params: experimental parameters
        :returns: the multilayer network"""
        g = self._generateLayerSeparated(g, params)
        g = self._addInterlayerEdges(g, params)
        return g

    def _generateLayerSeparated(self, params: Dict[str, Any]) -> Graph:
        """Generate a multilayer network from a list of tuples (or lists) of layer name and underlying network generator (or generator identifier).
        There are NO inter-layer edges in the graph returned by this function. This is useful for generating the layers separately.

        :param params: experimental parameters
        :returns: the multilayer network"""
        # extract the layers
        layers = params[self.LAYERS]
        if isinstance(layers, str):
            # then parse the string
            layers = json.loads(layers)
        if not isinstance(layers, dict):
            raise AttributeError("Layers must be a dict of (layer name : generator/generator name)")
        for (layer, generator) in layers.items():
            if isinstance(generator, str):
                # then a generator name, so match to the generator by class.__name__
                if generator not in _names:
                    raise AttributeError(f"Generator {generator} not found")
                generator = _names[generator](params, limit=self._remaining)
            elif not isinstance(generator, NetworkGenerator):
                raise AttributeError("Layers must be a dict of (layer name : generator/generator name)")
            # else is generator object so just use it
            self._layers[layer] = generator

        graphs = []
        for layer, generator in self._layers.items():
            # generate the network
            g = generator.generate()
            if g is None:
                raise AttributeError("Generator returned None")
            # add the layer attribute to the nodes
            for n in g.nodes():
                g.nodes[n]["layer"] = layer
            # add the graph to the list of graphs
            graphs.append(g)
        # combine the graphs into a single graph
        g = disjoint_union_all(graphs)
        return g
    
    def _addInterlayerEdges(self, g: Graph, params: Dict[str, Any]) -> Graph:
        """Add inter-layer edges to the graph. This is done in a configuration-model style 
        fashion where nodes from each layer are randomly connected together according to the
        distribution formed by the global mean inter-layer edge participation :attr:`INTERMEAN`.

        :param g: the graph to add inter-layer 
        :param params: experimental parameters
        :returns: the graph with inter-layer edges added"""
        # extract the inter-layer edge participation
        interMean = params[self.INTERMEAN]
        participation = rng.poisson(interMean, len(g.nodes()))
        nodes = []
        for n in g.nodes():
            # add the node to the list of nodes
            nodes.extend([n] * participation[n])
        if len(nodes) % 2 != 0:
            # then the number of nodes is odd, so add an extra node to the list
            nodes.append(rng.choice(g.nodes()))
        rng.shuffle(nodes)
        for i in range(0, len(nodes), 2):
            u, v = nodes[i], nodes[i + 1]
            if u != v and g.nodes[u]["layer"] != g.nodes[v]["layer"]:
                # then add an inter-layer edge
                g.add_edge(u, v, interlayer = True)
            # else (for now ignore?....)
        return g
        

class MultiplexNetwork(MultilayerNetwork):
    """Generate a multiplex network from a dictionary of layer name and the underlying network generator.
    Each generator is called individually and passed the same parameters, meaning that the parameters dictionary
    must contain the necessary parameters for each generator. Parameters may be decorated with the layer name to 
    allow the use of the same generator with different parameters across layers. This generator is just a special
    case of a multi-layer network where every inter-layer edge is marked as an identity edge and cannot be occupied
    by processes. It is up to the process to be aware of the multiplex nature of the substrate, otherwise the graph
    is treated as just any other. 

    A multiplex network is a network with multiple layers, each of which can have its own topology.
    Inter-layer edges are an identity relationship between nodes in each layer, if the implementation uses
    several layers.

    :param params: (optional) experiment parameters
    :param limit: (optional) maximum number of instances to generate"""

    def topology(self) -> str:
        """Return the topology flag for this generator.

        :returns: the topology marker ("MP")"""
        return "MP"
    
    def _generate(self, params):
        """Generate a multiplex network from a dictionary of layer name and the underlying network generator (or generator identifier).
        
        :param params: experimental parameters
        :returns: the multiplex network"""
        g = super()._generateLayerSeparated(params) # use the multilayer network generator 
        # then connect pairwise
        for i in range(len(self._layers.keys()) - 1): # from bottom to second
            layerOneName = list(self._layers.keys())[i]
            layerTwoName = list(self._layers.keys())[i + 1]
            layerOnePool = [n for n in g.nodes() if g.nodes[n]["layer"] == layerOneName]
            layerTwoPool = [n for n in g.nodes() if g.nodes[n]["layer"] == layerTwoName]
            # ASSUME SAME SIZE FOR NOW
            rng.shuffle(layerOnePool)
            rng.shuffle(layerTwoPool)
            for j in range(len(layerOnePool)):
                u = layerOnePool[j]
                v = layerTwoPool[j]
                g.add_edge(u, v, identity = True)
                # nodes wont be the same as guaranteed unique node labels
        return g



_names: dict[str, NetworkGenerator] = {
    FixedNetwork.__name__: FixedNetwork,
    ERNetwork.__name__: ERNetwork,
    BANetwork.__name__: BANetwork,
    ConfigurationModel.__name__: ConfigurationModel,
    ClusteredNetwork.__name__: ClusteredNetwork,
    MultilayerNetwork.__name__: MultilayerNetwork,
    MultiplexNetwork.__name__: MultiplexNetwork,
}
