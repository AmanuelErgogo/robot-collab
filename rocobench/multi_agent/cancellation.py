"""Cancellation token used by synchronized multi-agent execution."""


class MultiAgentCancellationToken(object):
    def __init__(self) -> None:
        self._cancelled = False
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        return bool(self._cancelled)

    def cancel(self, reason: str = "") -> None:
        self._cancelled = True
        self.reason = str(reason)

    def throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError("Multi-agent execution cancelled: {}".format(self.reason))
