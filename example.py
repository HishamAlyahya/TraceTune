from chaintune import get_traced_sample

def llm(prompt):
    return f"Hi its me llm {prompt}"

def finetuned_llm(prompt):
    return f"Hi its me FINETUNED {prompt}"


def traced_func(question):
    c = 3
    context = []
    for i in range(c):
        doc = llm(str(i))
        context.append(doc)
    return context

inp = {"question": "What?"}
### Example usage
print("Execution Mode (without finetuned model):")
print("=========================================")
sample = get_traced_sample(program=traced_func, program_inputs=inp)
print(sample)
print("=========================================")
print("Inference Mode (with finetuned model):")
print("=========================================")
sample = get_traced_sample(program=traced_func, program_inputs=inp, finetuned_model=finetuned_llm)
print(sample)