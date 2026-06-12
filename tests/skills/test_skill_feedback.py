from rocobench.skills.compiler import SkillCompilationError
from rocobench.skills.models import PreparedSkillExecution, SkillCall, SkillPlan, SkillValidationResult, ValidationIssue
from prompting.skill_feedback import SkillFeedbackManager


class Validator:
    def __init__(self, result):
        self.result = result

    def validate(self, plan, obs):
        return self.result


class Compiler:
    def __init__(self, prepared=None, error=None):
        self.prepared = prepared
        self.error = error

    def compile(self, plan, obs):
        if self.error:
            raise self.error
        return self.prepared


class Geometry:
    def __init__(self, ready=True):
        self.ready = ready

    def give_feedback(self, plan):
        if self.ready:
            return True, "ok"
        return False, "geometry failed"


def make_plan():
    return SkillPlan([SkillCall("Alice", "WAIT", {}, "WAIT()")], "EXECUTE")


def test_feedback_requires_observation():
    manager = SkillFeedbackManager(Validator(SkillValidationResult.ok()), Compiler(), Geometry())

    ready, feedback = manager.give_feedback(make_plan())

    assert not ready
    assert "[NO_OBSERVATION]" in feedback


def test_feedback_validation_failure():
    result = SkillValidationResult.invalid([ValidationIssue("BAD", "bad plan")])
    manager = SkillFeedbackManager(Validator(result), Compiler(), Geometry())
    manager.update_obs(object())

    ready, feedback = manager.give_feedback(make_plan())

    assert not ready
    assert "[BAD]" in feedback


def test_feedback_compilation_failure():
    manager = SkillFeedbackManager(
        Validator(SkillValidationResult.ok()),
        Compiler(error=SkillCompilationError("COMPILATION_FAILED", "bad compile")),
        Geometry(),
    )
    manager.update_obs(object())

    ready, feedback = manager.give_feedback(make_plan())

    assert not ready
    assert "[COMPILATION_FAILED]" in feedback


def test_feedback_geometry_failure_does_not_attach_preparation():
    prepared = PreparedSkillExecution("rrt", "plan", ["compiled"])
    plan = make_plan()
    manager = SkillFeedbackManager(Validator(SkillValidationResult.ok()), Compiler(prepared), Geometry(ready=False))
    manager.update_obs(object())

    ready, feedback = manager.give_feedback(plan)

    assert not ready
    assert "geometry failed" in feedback
    assert plan.prepared_execution is None


def test_feedback_attaches_prepared_execution_on_success():
    prepared = PreparedSkillExecution("rrt", "plan", ["compiled"])
    plan = make_plan()
    manager = SkillFeedbackManager(Validator(SkillValidationResult.ok()), Compiler(prepared), Geometry())
    manager.update_obs(object())

    ready, _ = manager.give_feedback(plan)

    assert ready
    assert plan.prepared_execution is prepared
