class FakeObjectState:
    def __init__(self, name, contacts=None, xpos=None):
        self.name = name
        self.contacts = set(contacts or [])
        self.xpos = xpos or (0.0, 0.0, 0.0)


class FakeRobotState:
    def __init__(self, contacts=None):
        self.contacts = set(contacts or [])


class FakeObs:
    def __init__(self, objects=None, alice_contacts=None, bob_contacts=None):
        self.objects = objects or {
            "apple": FakeObjectState("apple"),
            "banana": FakeObjectState("banana"),
            "milk": FakeObjectState("milk"),
        }
        self.ur5e_robotiq = FakeRobotState(alice_contacts)
        self.panda = FakeRobotState(bob_contacts)


class FakePackEnv:
    def __init__(self):
        self.item_names = ["apple", "banana", "milk"]
        self.bin_slot_xposes = {
            "bin_front_left": (0.0, 0.0, 0.0),
            "bin_front_right": (1.0, 0.0, 0.0),
            "bin_back_left": (0.0, 1.0, 0.0),
        }
        self.robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
        self.robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}
        self.occupancy = {slot_name: None for slot_name in self.bin_slot_xposes}

    def get_agent_held_object(self, obs, agent_name):
        robot_state = getattr(obs, self.robot_name_map_inv[agent_name])
        for item_name in self.item_names:
            if item_name in robot_state.contacts:
                return item_name
        return None

    def get_slot_occupancy(self, obs):
        return dict(self.occupancy)

    def get_packed_slot_for_object(self, obs, object_name):
        for slot_name, occupant in self.occupancy.items():
            if occupant == object_name:
                return slot_name
        return None

    def describe_obs(self, obs):
        return "fake scene"
