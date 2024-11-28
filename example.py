import dspy

from chaintune.reflection import ReflectionModule
from chaintune.tracing import trace

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    lm = dspy.OpenAI(model="gpt-4o-mini", model_type="chat")
    dspy.configure(lm=lm)


    reflection = ReflectionModule()

    # Trace the function
    traced_f = trace(reflection.forward, {"instruction": "What is the capital of France?"})

    # Call the traced function
    result = traced_f(instruction="What is the capital of France?")
    print("Final result:", result)

