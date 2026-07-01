Skills
======

A **skill** is a named, parameterised physical action that one agent can
perform. Skills are declared as :class:`~rocobench.skills.models.SkillSpec`
objects and stored in a :class:`~rocobench.skills.registry.SkillRegistry`.

Skill declaration
-----------------

.. autoclass:: rocobench.skills.models.SkillSpec
   :members:
   :undoc-members:

Skill invocation
-----------------

.. autoclass:: rocobench.skills.models.SkillCall
   :members:
   :undoc-members:

Registry
--------

.. autoclass:: rocobench.skills.registry.SkillRegistry
   :members:
   :undoc-members:

Built-in skills (Pack Grocery)
-------------------------------

.. automodule:: rocobench.skills.pack_grocery
   :members:
   :undoc-members:

Skill validation
-----------------

.. autoclass:: rocobench.skills.models.SkillValidationResult
   :members:
   :undoc-members:

.. automodule:: rocobench.skills.validation
   :members:
   :undoc-members:

Skill name normalisation
-------------------------

Skill names are always stored in upper-snake-case. The registry normalises
any name before lookup:

.. code-block:: python

   from rocobench.skills.models import normalize_skill_name

   normalize_skill_name("pick object")  # → "PICK_OBJECT"
   normalize_skill_name("Put-In-Box")   # → "PUT_IN_BOX"

.. autofunction:: rocobench.skills.models.normalize_skill_name

Learned skills
--------------

The ``rocobench.skills.learned`` sub-package provides infrastructure for
ACT / Diffusion / LeRobot policies loaded from checkpoints.

Key classes (require the ``roco`` conda environment to import):

``rocobench.skills.learned.registry.LearnedSkillRegistry``
   Registry that maps skill names to loaded policy checkpoints.
   Supports hot-swapping policies without restarting the simulator.

``rocobench.skills.learned.policy_handle.PolicyHandle``
   Thin wrapper around a loaded policy. Exposes ``predict(obs) -> action``
   and metadata (checkpoint path, training config, feature contract).

``rocobench.skills.learned.executor.LearnedSkillExecutorImpl``
   Concrete :class:`~rocobench.crie_bt.LearnedSkillExecutor` that calls
   ``PolicyHandle.predict`` on every simulator tick and converts the raw
   action tensor into a ``ExecutionFeedback`` using configurable success
   monitors.
