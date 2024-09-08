import dspy

from chaintune.reflection import ReflectionModule
from chaintune.tracing import trace

if __name__ == "__main__":
    dspy.configure(lm=lm)
    lm = dspy.OpenAI(model="gpt-4o-mini", model_type="chat")


    reflection = ReflectionModule()

    # Trace the function
    traced_f = trace(reflection.forward, {"instruction": "What is the capital of France?"})

    # Call the traced function
    result = traced_f(instruction="What is the capital of France?")
    print("Final result:", result)

