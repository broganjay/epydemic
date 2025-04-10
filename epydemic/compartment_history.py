class CompartmentHistory:

    _templates: dict = dict()

    def __new__(cls, compartments: tuple[str, ...] = tuple()) -> "CompartmentHistory":
        if compartments in cls._templates:  # if obj already exists
            return cls._templates[compartments]  # return the existing
        else:
            template: CompartmentHistory = super().__new__(cls)  # or create a new one
            cls._templates[compartments] = template
            return template

    def __init__(self, compartments: tuple[str, ...] = tuple()) -> None:
        if not hasattr(self, "_initialised"):  # prevent re-setting attrs
            self._compartments: tuple[str, ...] = compartments
            self._length: int = len(compartments)
            self._initialised: bool = True
        # else: nothing
        # (__init__ is called every time regardless of whether newly created or just passed reference to existing)

    def getCompartments(self) -> tuple[str, ...]:
        return self._compartments

    def getNumberOfCompartments(self) -> int:
        return self._length

    def __len__(self) -> int:
        return self._length

    def __contains__(self, compartment) -> bool:
        return compartment in self._compartments

    def __getitem__(self, compartment) -> str:
        return self._compartments[compartment]

    def getLatestCompartment(self) -> str:
        return self._compartments[-1] if self._length > 0 else None

    def __str__(self):
        return f"CompartmentHistory{self._compartments}"

    @staticmethod
    def updateHistory(history: "CompartmentHistory", nc: str) -> "CompartmentHistory":
        return CompartmentHistory(
            history.getCompartments() + (nc,)
        )  # add new compartment to history
