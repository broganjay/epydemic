import sys
from typing import Dict, Any, Tuple
if sys.version_info >= (3, 8):
    from typing import Dict, Tuple
else:
    # backport compatibility with older typing
    from typing_extensions import Dict
from epydemic import SIR, CompartmentedModel

IMMUNISES = 'epydemic.interactions.IMMUNISES'
SUSCEPTS = 'epydemic.interactions.SUSCEPTS'
HINDERS = 'epydemic.interactions.HINDERS'
BOOSTS = 'epydemic.interactions.BOOSTS'

class Interactions:

    def __init__(self, models: CompartmentedModel):
        self._models = models
        self._interactions = []

    def addModel(self, model: SIR):
        self._models.append(model)
        ## some rejigging needed

    def addInteraction(self, interaction: str, source: str, target: str, rate: float):
        if not source in self._models:
            raise ValueError(f"Source model {source} not in the list of models")
        if not target in self._models:
            raise ValueError(f"Target model {target} not in the list of models")
        print(f"Adding interaction {interaction} from {source} to {target} at rate {rate}")
        self._interactions.append((source, target, interaction, rate))
        ### rejigging needed
        self._integrateInteraction(interaction, source, target, rate)
        print(self._interactions)

    def _integrateInteraction(self, interaction, source, target, rate):
        ## rejigging needed
        """
        so start with makes vulnerable to
        this means that all removed nodes in the source model are equal to the
        infectious nodes in the target model
        add a new compartment to both models
        """
        pass