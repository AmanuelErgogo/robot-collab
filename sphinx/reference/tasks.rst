Tasks
=====

All tasks are MuJoCo environments with a shared interface. A task environment
must provide:

* ``reset() -> obs`` — reset the simulator, return initial observation.
* ``get_reward_done(obs) -> (reward, done)`` — check task completion.

Built-in tasks
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 18 52

   * - Task
     - Agents
     - Class
     - Description
   * - **Sandwich**
     - Chad, Dave
     - Sequential-Dependent
     - Assemble a bacon sandwich in strict recipe order:
       bread → bacon → cheese → tomato → bread. ~10 actions.
   * - **Pack Grocery**
     - Alice, Bob
     - Parallel-Independent
     - Pack grocery items into a box in any order.
   * - **Cabinet**
     - Alice, Bob, Carol (3)
     - Gated
     - One agent opens the cabinet door; the others place items inside.
   * - **Sort**
     - Alice, Bob, Carol (3)
     - Parallel-Independent
     - Sort objects by category into correct bins.
   * - **Sweep**
     - Chad, Dave
     - Continuous-Cooperative
     - Cooperatively sweep scattered objects to a target zone.
   * - **Rope**
     - Chad, Dave
     - Tightly-Coupled
     - Both arms grip the rope ends simultaneously and transport.

.. note::

   Task environments depend on MuJoCo and can only be fully imported inside the
   ``roco`` conda environment. Use
   ``conda run -n roco python -c "from rocobench.envs.task_sandwich import MakeSandwichTask; help(MakeSandwichTask)"``
   for live introspection.

Sandwich — ``MakeSandwichTask``
---------------------------------

**Module**: ``rocobench.envs.task_sandwich``

Assemble a bacon sandwich by stacking five items in strict recipe order.

.. code-block:: text

   Recipe: bread_slice1 → bacon → cheese → tomato → bread_slice2

- **Agents**: Chad (``ur5e_suction``, right arm), Dave (``humanoid``, left arm)
- **Actions per agent**: ``PICK <object>``, ``PLACE <target>``, ``WAIT``
- **Done condition**: Each item stacked on the previous one in recipe order
- **Coordination class**: Sequential-Dependent
- **Typical episode length**: ~10 subtasks

Key attributes:

``robot_name_map``
   ``{"ur5e_suction": "Chad", "humanoid": "Dave"}``

``reset() -> obs``
   Resets the MuJoCo scene. Food items (bacon, cheese, tomato) are shuffled
   randomly; bread positions are fixed.

``get_reward_done(obs) -> (reward, done)``
   Returns ``done=True`` when every item in ``recipe_order`` is correctly
   stacked on the item below it in the recipe sequence.

Pack Grocery — ``PackGroceryTask``
------------------------------------

**Module**: ``rocobench.envs.task_pack``

Pack grocery items from the table into a box in any order.

- **Agents**: Alice (``ur5e_robotiq``), Bob (``panda``)
- **Coordination class**: Parallel-Independent
- **Done condition**: All items inside the box

Cabinet — ``CabinetTask``
---------------------------

**Module**: ``rocobench.envs.task_cabinet``

Open the cabinet door first, then place items inside.

- **Agents**: Alice (``ur5e_robotiq``), Bob (``panda``), Carol (``ur5e_suction``)
- **Coordination class**: Gated (door must open before placement)
- **Done condition**: All target items placed inside the cabinet

Sort — ``SortOneBlockTask``
-----------------------------

**Module**: ``rocobench.envs.task_sort``

Sort objects by category into the correct coloured bins.

- **Agents**: Alice (``ur5e_robotiq``), Bob (``panda``), Carol (``ur5e_suction``)
- **Coordination class**: Parallel-Independent
- **Done condition**: All objects in their matching bins

Sweep — ``SweepTask``
-----------------------

**Module**: ``rocobench.envs.task_sweep``

Cooperatively sweep scattered objects into a target zone.

- **Agents**: Chad (``ur5e_robotiq``), Dave (``panda``)
- **Coordination class**: Continuous-Cooperative
- **Done condition**: All objects within the target boundary

Rope — ``MoveRopeTask``
-------------------------

**Module**: ``rocobench.envs.task_rope``

Both arms grip the rope ends simultaneously and transport it to a target.

- **Agents**: Chad (``ur5e_robotiq``), Dave (``panda``)
- **Coordination class**: Tightly-Coupled (simultaneous bimanual)
- **Done condition**: Rope midpoint within tolerance of the target position

Task coordination classes
--------------------------

Tasks are grouped by coordination structure. The class determines which
failure modes are most likely and which controller benefits most.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Implications for evaluation
   * - Sequential-Dependent
     - Wrong-order failures dominate. Feedback+BT gains are largest here.
       Open-loop TCR ≈ 0% because one subtask cannot complete a 10-step recipe.
   * - Parallel-Independent
     - Collision avoidance is the main challenge. All controllers perform
       similarly; Dialog mode can explicitly partition the workspace.
   * - Gated
     - Deadlock risk if the prerequisite is not recognised. Direct feedback
       helps but BT mediation adds explicit gate-detection logic.
   * - Tightly-Coupled
     - Requires synchronisation. Dialog mode is critical — agents negotiate
       timing before committing. Single-Robot mode is not feasible.
   * - Continuous-Cooperative
     - No natural step granularity for LLM planning. Only Direct Feedback
       and BT-Mediated controllers are meaningful.

Adding a new task
------------------

See :doc:`/tutorials/add_task` for a step-by-step walkthrough.
