
# Simple exceptions used in testing
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

RETRIES: int = 3

class StochasticException(Exception):
    '''A custom exception indicating that something unexpected or unwanted occurred
    in the course of the test directly as a result of the stochastic nature of the processes tested,
    but is not regarded strictly as a failure. This exception is used to signal to the retry framework
    that a test should be repeated and not yet failed. The most common use of this is retrying in the event
    that the disease being tested is not successfully seeded onto the network -- not strictly a failure (as 
    entirely possible), but not part of a successful test either.'''

    DEFAULT_MESSAGE: str = "Some conditions to create the test failed (as sometimes happens with stochastic processes). The test will be retried up to a maximum of {RETRIES} times.".format(RETRIES=RETRIES)

    def __init__(self, message = DEFAULT_MESSAGE):
        super().__init__(message)