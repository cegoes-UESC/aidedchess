from typing import Any


class StateManager:

    state: dict = {
        "current_position": [0, 0],
        "running": True,
    }

    def __init__(self) -> None:
        pass

    def getState(self, state: str | None = None) -> any:

        if state is None:
            return self.state

        if state not in self.state:
            return None

        return self.state[state]

    def __setitem__(self, name: str, value: Any) -> None:
        self.state[name] = value


stateManager = StateManager()
