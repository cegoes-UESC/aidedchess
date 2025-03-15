class Key:
    KEY_LEFT = 81
    KEY_UP = 82
    KEY_RIGHT = 83
    KEY_DOWN = 84
    Q = ord("q")


class KeyManager:
    callbacks = {}

    def __init__(self):
        pass

    def onKey(self, key: Key, callback) -> None:

        if key not in self.callbacks:
            self.callbacks[key] = [callback]

        else:
            self.callbacks[key].append(callback)

    def processKey(self, key: Key) -> None:

        if key in self.callbacks:

            for c in self.callbacks[key]:
                if callable(c):
                    c()
                elif isinstance(c, tuple):
                    cls, func = c[0], c[1]
                    f = getattr(cls, func)
                    f()


key_manager = KeyManager()
