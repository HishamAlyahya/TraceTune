#%%
import pdb
import random
import sys
import copy
import linecache
import dspy

class ExecutionTracker(pdb.Pdb):
    def __init__(self, traced_program, program_inputs, finetuned_model=None, llm_fns=None, *args, **kwargs):
        super(ExecutionTracker, self).__init__(*args, **kwargs)
        self.traced_program = traced_program
        self.program_inputs = program_inputs

        # this is set when the user wants to do all the calls that call an llm using a single finetuned model (inference mode)
        # if this is set to None, the program is just executed and traced.
        self.finetuned_model = finetuned_model
        # generated variables holds the values of variables that were generated by the finetuned model if it was set
        self.generated_variables = {}

        # this is a list of functions that are to be treated as llm functions during inference mode
        # if llm_fns = None, llm_fns will be all the dspy modules in the program (based on program self attributes)
        if llm_fns is not None:
            self.llm_fns = llm_fns
        else:
            traced_program_self = traced_program.__self__ # this assumes that the passed traced_program is a method of a class
            self.llm_fns = [fn for fn in dir(traced_program_self) if isinstance(getattr(traced_program_self, fn), dspy.Module)] 
        print(f"llm_fns: {self.llm_fns}")

        self.trace_string = ""

        self.prev_local_dict = None
        self.prev_line = None

        self.tracing_enabled = False
        self.first = True

        
    
    def first_write(self):
        self.trace_string += f"inputs:\n"
        for k, v in self.program_inputs.items():
            self.trace_string += f"{k} = {v}\n\n"

        self.trace_string += f"Execution:\n\n"
    
    def is_llm_call(self, line):
        # TODO: could be done more robustly by dealing with more reliable types rather than just string matching
        return any([fn in line for fn in self.llm_fns])


    def get_llm_fn_signature_schema(self, llm_fn: str, local_dict):
        llm_fn_module = getattr(local_dict["self"], llm_fn)
        signature = llm_fn_module.extended_signature if hasattr(llm_fn_module, "extended_signature") else llm_fn_module.signature
        signature_schema = signature.model_json_schema()
        
        return signature_schema

    
    def call_finetuned_model(self, line, current_local_dict):
        def get_llm_args_kwargs(inputs):
            llm_args = []
            llm_kwargs = {}
            inputs = inputs.replace("(", "").replace(")", "")
            inputs = inputs.split(",")
            inputs = [inp for inp in inputs if inp]
            for inp in inputs:
                if "=" in inp:
                    k,v = inp.split("=")
                    llm_kwargs[k.strip()] = eval(v, copy.deepcopy(current_local_dict))
                else:
                    llm_args.append(eval(inp, copy.deepcopy(current_local_dict)))
            return llm_args, llm_kwargs

        # first, parse the input names from the line
        import re
        inputs = re.findall(r'\(.*\)', line)[0]
        inputs = inputs[:-1] + "," + inputs[-1]

        # TODO: this is very hacky, we can do something fancier like parsing the stuff between the parentheses, injecting that into a dummy function, calling it, then getting the values of those variables.
        # this will be needed in case the inputs are not just variables but also function calls or other stuff
        # right now this just assumes the inputs are variables
        # mk: even more hacks to deal with positional and keyword arguments in get_llm_args_kwargs ...
        args, kwargs = get_llm_args_kwargs(inputs)
        args = ", ".join([repr(arg) for arg in args])
        kwargs = ", ".join([f"{k}={repr(v)}" for k,v in kwargs.items()])
        prompt = args + " " +  kwargs
        prompt = self.trace_string + line.strip()

        for fn_name in self.llm_fns:
            if fn_name in line:
                llm_fn = fn_name

        signature_schema = self.get_llm_fn_signature_schema(llm_fn=llm_fn, local_dict=current_local_dict)

        # take only the output fields
        inp_fields = {}
        for field_name, props in signature_schema["properties"].items():
            if props["__dspy_field_type"] == "output":
                inp_fields[field_name] = props

        variable_name = line.split("=")[0].strip()
        prediction = {}
        for inp_field, props in inp_fields.items():
            prompt += f"> {variable_name}.{inp_field} = {props['prefix']} "
            llm_output =  self.finetuned_model(prompt)[0]
            prediction[inp_field] = llm_output
            prompt += llm_output + "\n"
        llm_output = dspy.Prediction(prediction)
        return llm_output

    def diff_write(self, prev, current, line):
        diff = dict(set(current.items()) - set(prev.items()))   
        diff = sorted(diff.items())
            
        for k, v in diff:
            if k in self.program_inputs and self.program_inputs[k] == v:
                continue

            if k == "self":
                continue    

            # this bit of code makes sure that every prediction is printed in such a way that follows the steps the underlying module is doing to generate the output.
            # for example, when using dspy.ChainOfThought, both the rationale and the answer is ensured to be printed in the correct order
            # the "prefix" bit is what gets prefixed to the output of every dspy module. this key is always defined.
            # for example, if the output field "answer" is there in the prediction, it will always not include the prefix
            # this code ensure that instead of printing for example "the capital of France is Paris", it would print "Answer: the capital of France is Paris"
            if isinstance(v, dspy.Prediction):
                for inp, out in v.items():
                    for llm_fn in self.llm_fns:
                        if llm_fn in line:
                            signature_schema = self.get_llm_fn_signature_schema(llm_fn=llm_fn, local_dict=current)
                            prefix = signature_schema["properties"][inp]["prefix"]
                            self.trace_string += f"> {k}.{inp} = {prefix} {out}\n"
                            break
                self.trace_string += "\n"
                continue

            self.trace_string += f"> {k} = {v}\n\n"


    def trace_dispatch(self, frame, event, arg):
        if not self.tracing_enabled:
            return
        
        # step into only the first call (program(**program_inputs) call)
        if self.first and event == "call":
            self.set_step()
            self.first = False
            return super().trace_dispatch(frame, event, arg)

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        cur_line = linecache.getline(filename, lineno, frame.f_globals)
        # only trace lines
        if event != "line":
            return 
        current_local_dict = frame.f_locals
    

        for key, value in self.generated_variables.items():
            if isinstance(value, dspy.Prediction):
                frame.f_locals[key] = value
            else:
                exec(f"{key} = {repr(value)}", frame.f_globals, frame.f_locals)        

        
        # if we have reached the end of the trace, stop tracing
        if cur_line.strip() == "set_trace(end=True)":
            return

        new_dict = {}
        for k, v in current_local_dict.items():
            if isinstance(v, dspy.Module):
                new_dict[k] = v
                continue

            if isinstance(v, dspy.Prediction):
                new_dict[k] = v
                continue
            try:
                new_dict[k] = str(v)
            except:
                pass

        current_local_dict = new_dict
        
        if self.prev_local_dict is None:
            # this is the first call after setting our debugger 
            self.first_write()
        else:
            self.diff_write(prev=self.prev_local_dict, current=current_local_dict, line=self.prev_line)

        ###### INFERENCE MODE ######
        # TODO: naive way to do it but it could be done very cleanly and robustly
        if self.finetuned_model and self.is_llm_call(cur_line):

            # TODO: now it assumes only one variable is being assigned per line. We should handle cases where a, b = x, y. Some eval things could be done here
            k = cur_line.split("=")[0].strip()
            if f"{k} =" in cur_line:
                # we are in inference mode
                # we want to call the finetuned model here and replace v with the generation of the finetuned model
                v = self.call_finetuned_model(cur_line, current_local_dict)
                self.generated_variables[k] = v


        self.prev_local_dict = copy.copy(current_local_dict)
    
        self.prev_line = cur_line

        # don't write the first line (program() call)
        if self.first and event == "line":
            return
        
        self.trace_string += cur_line.strip() + "\n"

def get_traced_sample(program: callable, program_inputs: dict, finetuned_model: callable = None):
    execution_tracker = ExecutionTracker(program_inputs=program_inputs, finetuned_model=finetuned_model, traced_program=program)

    # TODO: it makes more sense for this to be a method of execution_tracker and just call execution_tracker.set_trace() but it doesn't work for some reason
    def set_trace(end=False):
        if end:
            execution_tracker.tracing_enabled = False
            execution_tracker.trace_file = None
            return 

        execution_tracker.rcLines.append("next")
        execution_tracker.set_trace(sys._getframe().f_back)
        execution_tracker.tracing_enabled = True

    set_trace()  # Start the debugger here
    
    program(**program_inputs)

    set_trace(end=True)

    return execution_tracker.trace_string

# %%
# Useful for quick debugging, ok to remove. 
class BasicMultiHop(dspy.Module):
  def __init__(self, passages_per_hop=3):
    self.retrieve = dspy.Retrieve(k=passages_per_hop)
    self.generate_query = dspy.ChainOfThought("context, question -> search_query")
    self.generate_answer = dspy.ChainOfThought("context, question -> answer")

  def forward(self, question):
    context = []

    for hop in range(2):
      query = self.generate_query(context=context, question=question)
      context += self.retrieve(query.search_query).passages

    # this was a single line: "return self.generate_answer(context=context, question=question)" 
    # spliting to two lines to allow tracing to capture the intermediate value, however there is a fundamental problem here ..
    # but works for now
    answer = self.generate_answer(context=context, question=question) 
    return answer
  
  
# prog = BasicMultiHop()
# lm = dspy.OpenAI(model="gpt-4o-mini", model_type="chat")
# colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# dspy.settings.configure(rm=colbertv2, lm=lm)
# import random
def finetuned_llm(prompt):
    random_queries = [
        "Who is the king of Saudi Arabia?",
        "How many people live in Spain?",
        "What's the riemann hypothesis?",
    ]
    return [f"Last 3 words before this are: [{', '.join(prompt.split()[-3:])}]. now, {random.choice(random_queries)}"]
# inp = {"question": "What is the capital of France??"}
# sample = get_traced_sample(prog.forward, inp )
# print(sample)
# # %%
# lm("hi")
# # %%
# lm("hi")

# # %%

# import dspy
# import dspy
# from dspy.evaluate import Evaluate
# from dspy.datasets.hotpotqa import HotPotQA
# from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
# class BasicMultiHop(dspy.Module):
#   def __init__(self, passages_per_hop=3):
#     self.retrieve = dspy.Retrieve(k=passages_per_hop)
#     self.generate_query = dspy.ChainOfThought("context, question -> search_query")
#     self.generate_answer = dspy.ChainOfThought("context, question -> answer")

#   def forward(self, question):
#     context = []

#     for hop in range(2):
#       print("before query")
#       query = self.generate_query(context=context, question=question).search_query
#       print("after query")
#       print("before retrieve")
#       context += self.retrieve(query).passages
#       print("after retrieve")

#     # this was a single line: "return self.generate_answer(context=context, question=question)" 
#     # spliting to two lines to allow tracing to capture the intermediate value, however there is a fundamental problem here ..
#     # but works for now
#     print("before answer")
#     answer = self.generate_answer(context=context, question=question) 
#     print("after answer")
#     return answer

# from dotenv import load_dotenv
# lm = dspy.OpenAI(model="gpt-4o-mini", model_type="chat")
# colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# dspy.settings.configure(rm=colbertv2, lm=lm)

# # %%
# prog = BasicMultiHop()
# inp = dict()
# inp["question"] = "What is the capital of France?"
# # %%
# sample = get_traced_sample(prog.forward, inp)
# # %%
# import copy

# # Define the local context
# context = "This is the context."
# question = "What is the question?"

# # Create a dictionary with the variables
# local_dict = {
#     "context": context,
#     "question": question,
#     "a": "This is the context.",
# }

# # String representation of the inputs
# inputs = "(a, context=context, question=question,)"

# # Evaluate the inputs using the dictionary
# kwargs = dict()
# inputs: tuple = eval(inputs, copy.deepcopy(local_dict))

# # Print the result
# print(inputs)  # Outputs: ('This is the context.', 'What is the question?')

# # %%
# # remove parentheses if they exist
# inputs.replace("(", "").replace(")", "").split(",")[0].split("=")
# # %%
# local_dict = {
#     "context": context,
#     "question": question,
#     "a": "This is the context.",
# }
# llm_args = []
# llm_kwargs = {}
# inputs = "(a, z=3, k=a, context=context, question=question,)"
# # no parentheses
# inputs = inputs.replace("(", "").replace(")", "")
# # split by comma
# inputs = inputs.split(",")
# inputs = [inp for inp in inputs if inp]
# for inp in inputs:
#     if "=" in inp:
#         k,v = inp.split("=")
#         llm_kwargs[k.strip()] = eval(v, copy.deepcopy(local_dict))
#     else:
#         llm_args.append(eval(inp, copy.deepcopy(local_dict)))
# print(llm_args)
# print(llm_kwargs)
# # make them a string 
# # %%
# args = ", ".join([repr(arg) for arg in llm_args])
# kwargs = ", ".join([f"{k}={repr(v)}" for k,v in llm_kwargs.items()])
# print(args + " " +  kwargs)

# %%
