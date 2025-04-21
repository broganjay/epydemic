# Test interactions between two compartmented processes 
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
import json

class InteractionsTest(unittest.TestCase):

    D1_NAME = "d1"
    D2_NAME = "d2"
    INFECTION_NAME = "infect"

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
        self._d1 = SIR()
        self._d2 = SIR()
        self._g = ERNetwork()
        self._e = StochasticDynamics(ProcessSequence({self.D1_NAME: self._d1, self.D2_NAME: self._d2}), self._g)
        self._params['std_interactions'] = {}
        self._params['custom_interactions'] = {}
        self._params['std_interactions'][self.D1_NAME] = []
        self._params['std_interactions'][self.D2_NAME] = []

    def testCrossImmunity(self):
        '''Test a cross-immunity interaction between two compartmented processes.'''
        self._params['std_interactions'][self.D1_NAME].append({self.D2_NAME: ('causes_immunity', self.INFECTION_NAME)})
        self._params['std_interactions'][self.D2_NAME].append({self.D1_NAME: ('causes_immunity', self.INFECTION_NAME)})
        self._params['std_interactions'] = json.dumps(self._params['std_interactions'])
        self._params['custom_interactions'] = json.dumps(self._params['custom_interactions'])

        rc = self._e.set(self._params).run(fatal=True)
        results = rc[self._e.RESULTS]
        # normally max number of cumulative Rs is N * 2, but with cross-immunity
        # node can catch max 1 disease = N instead so check this first
        self.assertLessEqual(results[SIR.REMOVED], self._N)
        # then check that there is not any nodes which seem to have been infected by _both_ processes
        g = self._e.network()
        for n, data in g.nodes(data=True):
            history = data.get("history", {})
            d1History = history.get(self.D1_NAME, ([], CompartmentHistory()))
            d2History = history.get(self.D2_NAME, ([], CompartmentHistory()))
            d1Compartments, _ = d1History
            d2Compartments, _ = d2History
            if SIR.INFECTED in d1Compartments:
                self.assertNotIn(SIR.INFECTED, d2Compartments) # if infected by d1, not infected by d2
            if SIR.INFECTED in d2Compartments:
                self.assertNotIn(SIR.INFECTED, d1Compartments) # if infected by d2, not infected by d1
        
    def testInfectionPrecondition(self):
        '''Test an infection precondition interaction between two compartmented processes.
        In this test, a node must have contracted d1 before it can contract d2.'''
        self._params['std_interactions'][self.D2_NAME].append({self.D1_NAME: ('must_have_for_infection', self.INFECTION_NAME)})
        self._params['std_interactions'] = json.dumps(self._params['std_interactions'])
        self._params['custom_interactions'] = json.dumps(self._params['custom_interactions'])
        # to make it easier for d2 to spread (without using staging, which is tested separately) set the params
        # so that  ~half of the graph starts infected with d1, and d2 is seeded onto ~10% of the graph to ensure
        # it has the chance to spread
        self._d1.setParameters(self._params, {self._d1.P_INFECTED: 0.5})
        self._d2.setParameters(self._params, {self._d2.P_INFECTED: 0.1})
        rc = self._e.set(self._params).run(fatal=True)
        results = rc[self._e.RESULTS]
        # first check that the number of R@d2 is less than the number of R@d1
        self.assertLessEqual(results[self._d2.decoratedName(SIR.REMOVED)], results[self._d1.decoratedName(SIR.REMOVED)])
        # then check for each node that has been infected by d2, it has also been infected by d1
        for n, data in self._e.network().nodes(data=True):
            history = data.get("history", {})
            d1History = history.get(self.D1_NAME, ([], CompartmentHistory()))
            d2History = history.get(self.D2_NAME, ([], CompartmentHistory()))
            d1Compartments, _ = d1History
            d2Compartments, d2Timings = d2History
            if SIR.INFECTED == d2Compartments[0] and d2Timings[0] <= 0:
                # then initially infected as part of seeding, so ignore...
                continue
            if SIR.INFECTED in d2Compartments:
                self.assertIn(SIR.INFECTED, d1Compartments)
            




if __name__ == '__main__':
    unittest.main()
