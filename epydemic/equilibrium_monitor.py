# Monitor the progress of an epidemic by just checking if each process is in equilibrium 

import sys
from typing import Dict, Any
if sys.version_info >= (3, 8):
    from typing import Final
else:
    # backport compatibility with older typing
    from typing_extensions import Final
from epydemic import Process

class EqMonitor(Process):

    DELTA: Final[str] = "epydemic.eq_monitor.time_delta"             #: Parameter for the time interval for observations.
    OBSERVATIONS: Final[str] = 'epydemic.eq_monitor.observations'    #: Result holding the times of the observations.
    TIMESERIES_STEM: Final[str] = "epydemic.eq_monitor.timeseries"   #: Stem for names of the timeseries for different loci.

    @staticmethod
    def timeSeriesForLocus(l: str) -> str:
        '''Generate the name of the time series corresponding to the
        given locus.

        :param l: the locus
        :returns: the result name'''
        return '{stem}-{locus}'.format(stem=EqMonitor.TIMESERIES_STEM, locus=l)
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.eqFound = False

    def reset(self):
        '''Reset the process.'''
        super().reset()
        self._timeSeries = None

    def build(self, params: Dict[str, Any]):
        '''Build the observation process.

        :param params: the experimental parameters

        '''
        super().build(params)

        # post a repeating event to observe the process
        delta = params[self.DELTA]
        self.postRepeatingEvent(0.0, delta, None, self.observe)

    def observe(self, t: float, e: Any):
        '''Observe the process.
        Was originally going to use a simple equilibrium check but abstraction
        broken, so instead just monitoring the number of SI loci - so only works with SIR models
        at the moment.
        
        :param t: the simulation time
        :param e: the event
        '''
        if self.eqFound:
            return
        # not do anything . . . 
        """
        for (n, l) in self.dynamics().loci().items():
            if l.name() == 'epydemic.sir.SI':
                if (len(l) == 0):
                    print("SI locus = 0!")
                    print("at time t = ", t)
                    self.eqFound = True
                    self.postEvent(t, n, self.alert, "second-epidemic")
        """
        # just check if particular process in equilibrium
        if self.dynamics().process().get("d1").atEquilibrium(t):
            print("at equilibrium")
            self.eqFound = True
            self.postEvent(t, self.dynamics().process().get("d1"), self.dynamics().process().get("d2").spontaneousInfection, "second-epidemic")