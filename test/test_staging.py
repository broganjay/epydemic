# Test staging of processes in simulations of >1 process
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

from epydemic import *
import unittest
from tenacity import retry, stop_after_attempt
import json

class StagingTest(unittest.TestCase):

    D1_NAME = "d1"
    D2_NAME = "d2"
    INFECTION_NAME = "infect"
    MONITOR_NAME = "monitor"

    def setUp(self):
        '''Set up the experimental parameters and process.'''
        self._N = 5000
        self._kmean = 2
        phi = (self._kmean + 0.0) / self._N
        self._params = dict()
        self._networkGenerator = ERNetwork()
        self._params[ERNetwork.N] = self._N
        self._params[ERNetwork.KMEAN] = self._kmean
        self._params[SIR.P_INFECTED] = 1 / self._N
        self._params[SIR.P_INFECT] = 0.3
        self._params[SIR.P_REMOVE] = 0.05
        self._params[Monitor.DELTA] = 1
        self._d1 = SIR()
        self._d2 = SIR()
        self._monitor = Monitor()
        self._g = ERNetwork()
        self._e = StochasticDynamics(ProcessSequence({self.D1_NAME: self._d1, self.D2_NAME: self._d2, self.MONITOR_NAME: self._monitor}), self._g)

    def testFixedTimeStaging(self):
        '''Test that a process emerges at a fixed time specified.'''
        self._d2.setParameters(self._params,
                               {SIR.EMERGENCE_CONDITION: 100,
                                SIR.START_COMPARTMENT: SIR.SUSCEPTIBLE,})
        rc = self._e.set(self._params).run(fatal=True)
        results = rc[self._e.RESULTS]
        nInfected = results[self._monitor.decoratedNameInInstance(Monitor.timeSeriesForLocus(self._d2.decoratedNameInInstance(SIR.INFECTED)))]
        emergenceTime = results[self._monitor.decoratedNameInInstance(Monitor.OBSERVATIONS)].index(100.0)
        # assert that no infected nodes before emergenceTime
        self.assertSequenceEqual(nInfected[:emergenceTime], [0] * emergenceTime)
        # assert that at least one infected node after emergenceTime
        # assumes sufficient delta granularity (currently 1, so fine)
        self.assertGreater(nInfected[emergenceTime + 1], 0)

    @retry(stop=stop_after_attempt(3))
    def testConditionEquilibriumStaging(self):
        '''Test that the equilibrium staging works with d2 seeded after d1 has equilibrated.'''
        self._d2.setParameters(self._params, 
                               {SIR.EMERGENCE_CONDITION: "equilibrium",
                                SIR.EMERGENCE_TARGET: self.D1_NAME,
                                SIR.START_COMPARTMENT: SIR.SUSCEPTIBLE,})
        rc = self._e.set(self._params).run(fatal=True)
        results = rc[self._e.RESULTS]
        equiIndex = results[self._monitor.decoratedNameInInstance(Monitor.timeSeriesForLocus(self._d1.decoratedNameInInstance(SIR.INFECTED)))].index(0)
        # first check that d1 did itself seed, as test is meaningless otherwise
        self.assertGreater(equiIndex, 0)
        # then assert that no d2 infected nodes before equiIndex
        # time-steps consistent so can just index into d2 locus with same index
        nInfectedD2 = results[self._monitor.decoratedNameInInstance(Monitor.timeSeriesForLocus(self._d2.decoratedNameInInstance(SIR.INFECTED)))]
        self.assertSequenceEqual(nInfectedD2[:equiIndex], [0] * equiIndex)
        # then assert that at least one d2 infected node after equiIndex
        self.assertGreater(nInfectedD2[equiIndex + 1], 0)

        

if __name__ == '__main__':
    unittest.main()
