"""Shared test doubles."""


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})
