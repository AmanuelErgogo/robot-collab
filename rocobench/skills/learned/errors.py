"""Stable learned skill executor errors."""


class LearnedSkillError(RuntimeError):
    def __init__(self, code, message, evidence=None):
        RuntimeError.__init__(self, message)
        self.code = str(code)
        self.message = str(message)
        self.evidence = dict(evidence or {})


class PolicyRegistryError(LearnedSkillError):
    pass


class PolicyHandleError(LearnedSkillError):
    pass


class LearnedExecutionError(LearnedSkillError):
    pass

