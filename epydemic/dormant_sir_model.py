# SIR as a compartmented model
#
# Copyright (C) 2017--2022 Simon Dobson
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

import sys
from typing import Dict, Any
if sys.version_info >= (3, 8):
    from typing import Final
else:
    # backport compatibility with older typing
    from typing_extensions import Final
from epydemic import SIR, rng

class DormantSIR(SIR):

    def __init__(self, name: str = None):
        super().__init__(name)
        self.dormant = True
        self.emerged = False

    def build(self, params: Dict[str, Any]):
        super().build(params)
        for c in self._compartments:
            self.changeCompartmentInitialOccupancy(c, 0.0)
        self.params = params
        self.postEvent(1.0, None, self.emerge)


    def emerge(self, t, n):
        #### the target of a posted event - currently emerges at a fixed time
        if self.emerged:
            return
        [pInfected] = self.getParameters(self.params, [self.P_INFECTED])
        y = 0
        edges = list(self.network().edges)[:1000]
        rng.shuffle(edges)
        for e in edges:
            if rng.random() <= pInfected:
                self.infect(t, e)
        self.emerged = True  
        self.notifyEmerged(t)

    def postEquilibriumEvent(self, e, ef):
        self.dynamics().postEquilibriumEvent(self, e, ef)

    def notifyEmerged(self, t):
        self.dynamics().notifyEmerged(self.configuration(), t)