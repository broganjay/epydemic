# Test multiple simultaneous instances of the same process
#
# Copyright (C) 2017--2024 Simon Dobson
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

from epydemic import *
from epyc import Experiment
import unittest
import networkx

class CoinfectionTest(unittest.TestCase):

    def testTwoProcesses(self):
        """Test we can run two process instances."""
        N = 10000
        kmean = 100

        # network
        params = dict()
        params[ERNetwork.N] = N
        params[ERNetwork.KMEAN] = kmean

        # first infection
        p1 = SIR("First disease")
        p1.setParameters(params,
                         {SIR.P_INFECT: 0.1,
                          SIR.P_INFECTED: 5.0 / N,
                          })

        # second infection
        p2 = SIR("Second disease")
        p2.setParameters(params,
                         {SIR.P_INFECT: 0.3,
                          SIR.P_INFECTED: 5.0 / N,
                          })

        # common removal rate
        params[SIR.P_REMOVE] = 0.005

        # run the processes together
        ps = ProcessSequence([p1, p2])

        # arbitrarily reduce max simulation time, as given enough time both would occupy the whole network anyway  
        # alternative to avoid would be to impose cross-immune interaction but left for another test
        ps.setMaximumTime(1)

        e = StochasticDynamics(ps, ERNetwork())
        rc = e.set(params).run(fatal=True)
        g = e.network()

        # bi: checking hit against size of final result inaccurate -- hit records most recent process only
        # so, the max value of len(hit1) + len(hit2) would be N, 10_000
        # switch to making use of results dict more ---
        # assert more removed from second process than first

        # we should have more nodes hit by the second infection than by the first
        # so less S nodes in p2 than in p1
        print(rc[Experiment.RESULTS])
        self.assertTrue(rc[Experiment.RESULTS][p2.decoratedNameInInstance(p2.SUSCEPTIBLE)] < rc[Experiment.RESULTS][p1.decoratedNameInInstance(p1.SUSCEPTIBLE)])
        # the total number of removed nodes should be less than 2 * N 
        self.assertTrue(rc[Experiment.RESULTS][p1.decoratedNameInInstance(p1.REMOVED)] + rc[Experiment.RESULTS][p2.decoratedNameInInstance(p2.REMOVED)] < 2 * N)
        # while we're here, check the aggregated values are correct too...
        self.assertTrue(rc[Experiment.RESULTS][p1.decoratedNameInInstance(p1.REMOVED)] + rc[Experiment.RESULTS][p2.decoratedNameInInstance(p2.REMOVED)] == rc[Experiment.RESULTS][SIR.REMOVED])

if __name__ == '__main__':
    unittest.main()
