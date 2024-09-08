import dspy


class ThinkSignature(dspy.Signature):
    """Given an instruction, previous thoughts, and previous reflections, return a new thought that will get you closer to responding with the best possible response."""
    instruction = dspy.InputField(desc="The instruction that you are trying to respond to.")
    previous_thoughts: list = dspy.InputField(desc="A list of previous thoughts.")
    previous_reflections: list = dspy.InputField(desc="A list of previous reflections.")
    new_thought = dspy.OutputField(desc="The new thought that will get you closer to responding with the best possible response.")

class ReflectSignature(dspy.Signature):
    """Given an instruction, previous thoughts, and previous reflections, return a new reflection that will critique the thoughts so far and get you closer to responding with the best possible response."""
    instruction = dspy.InputField(desc="The instruction that you are trying to respond to.")
    previous_thoughts: list = dspy.InputField(desc="A list of previous thoughts.")
    previous_reflections: list = dspy.InputField(desc="A list of previous reflections.")
    new_reflection = dspy.OutputField(desc="The new reflection that will get you closer to responding with the best possible response.")

class FinalResponseSignature(dspy.Signature):
    """Given an instruction, a list of thoughts, and a list of reflections, return the final response."""
    instruction = dspy.InputField(desc="The instruction that you are trying to respond to.")
    thoughts: list = dspy.InputField(desc="A list of thoughts.")
    reflections: list = dspy.InputField(desc="A list of reflections.")
    final_response = dspy.OutputField(desc="The final response.")

class ReflectionModule(dspy.Module):
    def __init__(self):
        self.think = dspy.TypedPredictor(ThinkSignature)
        self.reflect = dspy.TypedPredictor(ReflectSignature)
        self.give_final_response = dspy.TypedPredictor(FinalResponseSignature)

    def forward(self, instruction, n_thoughts = 5):
        thoughts = []
        reflections = []
        for i in range(n_thoughts):
            thought = self.think(instruction=instruction, previous_thoughts=thoughts, previous_reflections=reflections)
            thoughts.append(thought)
            reflection = self.reflect(instruction=instruction, previous_thoughts=thoughts, previous_reflections=reflections)
            reflections.append(reflection)
        final_response = self.give_final_response(instruction=instruction, thoughts=thoughts, reflections=reflections)
        return final_response
