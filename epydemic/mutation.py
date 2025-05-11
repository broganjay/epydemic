# A record containing sufficient information to specify the mutation behaviour of a process
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

from typing import Any, Dict, Optional
from epydemic import rng

class MutationProfile:

    def __init__(self, mutatingAttrs: Optional[Dict[str, float]] = None):
        self._mutatingAttrs: Optional[Dict[str, float]] = mutatingAttrs
        self._storedParams: Dict[str, Any] = dict()
    
    def getMutatingAttrs(self) -> Dict[str, float]:
        """Return the attributes that are mutated by this profile and the % difference the new attribute can have."""
        return self._mutatingAttrs

    def __getitem__(self, key: str) -> float:
        """Return the variance allowed for a given attribute."""
        return self._mutatingAttrs[key]
    
    def __iter__(self):
        """Return an iterator over the mutation rates."""
        return iter(self._mutatingAttrs)
    
    def storeParams(self, params: Dict[str, Any]) -> None:
        """Store the parameters in the mutation profile.
        Used for caching params used to create model, so new model can be
        created with same params."""
        self._storedParams = params

    def getStoredParams(self) -> Dict[str, Any]:
        """Return the stored parameters."""
        return self._storedParams
    
    @staticmethod
    def mutateAttr(attrValue: float, mutationTolerance: float) -> float:
        """Mutate the given attribute by a random amount within the given tolerance (taken as a percentage)
        e.g with `currentValue=0.1` and `mutationTolerance=0.1`, the new value will be ±10% of 0.1 (between 0.09 and 0.11).
        Assumes that the distribution of possible values across this range is uniform, although this may not be strictly true in all cases.
        This method is focused on producing meaningful mutations rather than modelling any specific biological phenomenon.
        
        :param currentValue: the current value of the attribute
        :param mutationTolerance: the percentage tolerance for the mutation
        :return: the mutated value

        """
        lower = max(0, attrValue - (attrValue * mutationTolerance)) # ensure doesn't go below 0
        upper = min(1, attrValue + (attrValue * mutationTolerance)) # ensure doesn't go above 1
        return rng.uniform(lower, upper)