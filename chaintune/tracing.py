import pdb
import random
import sys
import copy
import linecache
from inspect import signature
from dataclasses import dataclass


class ExecutionTracker(pdb.Pdb):
    def __init__(
        self,
        traced_program,
        program_inputs,
        finetuned_model=None,
        llm_fns=None,
        start_token="[LLM]",
        end_token="[/LLM]",
        ignored_vars=None,
        *args,
        **kwargs,
    ):
        super(ExecutionTracker, self).__init__(*args, **kwargs)
        self.traced_program = traced_program
        self.program_inputs = program_inputs

        self.start_token = start_token
        self.end_token = end_token

        self.ignored_vars = ignored_vars if ignored_vars else []
        # this is set when the user wants to do all the calls that call an llm using a single finetuned model (inference mode)
        # if this is set to None, the program is just executed and traced.
        self.finetuned_model = finetuned_model
        # generated variables holds the values of variables that were generated by the finetuned model if it was set
        self.generated_variables = {}

        # this is a list of functions that are to be treated as llm functions during inference mode
        # if llm_fns = None, llm_fns will be all the dspy modules in the program (based on program self attributes)
        self.llm_fns = llm_fns if llm_fns else []

        self.trace_string = ""

        self.prev_local_dict = None
        self.prev_line = None

        self.tracing_enabled = False
        self.steps = 1
        self.first = True

    def first_write(self):
        self.trace_string += f"Inputs:\n"
        for k, v in self.program_inputs.items():
            if k in self.ignored_vars:
                continue
            self.trace_string += f"{k} = {v}\n"
        self.trace_string += f"\nExecution:\n\n"

    def is_llm_call(self, line):
        # TODO: could be done more robustly by dealing with more reliable types rather than just string matching
        return any([fn in line for fn in self.llm_fns])

    def call_finetuned_model(self, line, current_local_dict):
        prompt = self.trace_string + line.strip()

        for fn_name in self.llm_fns:
            if fn_name in line:
                llm_fn = fn_name

        variable_name = line.split("=")[0].strip()
        prompt += f"> {variable_name} = {self.start_token}"
        llm_output = self.finetuned_model(prompt)
        return llm_output

    def diff_write(self, prev, current, line):
        diff = dict(set(current.items()) - set(prev.items()))
        diff = sorted(diff.items())
        if not diff:
            self.trace_string += f"> [No change in any variables]\n\n"
        for k, v in diff:
            if k in self.ignored_vars:
                continue

            if k in self.program_inputs and self.program_inputs[k] == v:
                continue

            for llm_fn in self.llm_fns:
                if llm_fn in line:
                    self.trace_string += (
                        f"> {k} = {self.start_token} {v} {self.end_token}\n\n"
                    )
                    break
                continue

            if llm_fn in line:
                break

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
            exec(f"{key} = {repr(value)}", frame.f_globals, frame.f_locals)

        # if we have reached the end of the trace, stop tracing
        if cur_line.strip() == "set_trace(end=True)":
            return

        new_dict = {}
        for k, v in current_local_dict.items():
            # convert all nonhashable values to strings
            try:
                hash(v)
                new_dict[k] = v
            except:
                new_dict[k] = str(v)
                pass

        current_local_dict = new_dict

        if self.prev_local_dict is None:
            # this is the first call after setting our debugger
            self.first_write()
        else:
            self.diff_write(
                prev=self.prev_local_dict,
                current=current_local_dict,
                line=self.prev_line,
            )

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

        # don't write the first line (func() call)
        if self.first and event == "line":
            return

        self.trace_string += f"Step {self.steps}: " + cur_line.strip() + "\n"
        self.steps += 1


@dataclass
class TracedProgramOutput:
    inputs: dict
    output: any
    trace_string: str
    llm_fns: list
    finetuned_model: callable


def trace(llm_fns=None, ignored_vars=None):
    def decorator(func):
        def wrapper(*args, finetuned_model=None, **kwargs):
            # get names of positional arguments
            for arg, param in zip(args, signature(func).parameters):
                kwargs[param] = arg

            # handle default arguments
            for param in signature(func).parameters.values():
                if param.default != param.empty and param.name not in kwargs:
                    kwargs[param.name] = param.default

            execution_tracker = ExecutionTracker(
                program_inputs=kwargs,
                finetuned_model=finetuned_model,
                traced_program=func,
                llm_fns=llm_fns,
                ignored_vars=ignored_vars,
            )

            def set_trace(end=False):
                if end:
                    execution_tracker.tracing_enabled = False
                    execution_tracker.trace_file = None
                    return

                execution_tracker.rcLines.append("next")
                execution_tracker.set_trace(sys._getframe().f_back)
                execution_tracker.tracing_enabled = True

            set_trace()  # Start the debugger here

            output = func(**kwargs)

            set_trace(end=True)

            return TracedProgramOutput(
                inputs=kwargs,
                output=output,
                trace_string=execution_tracker.trace_string,
                llm_fns=llm_fns,
                finetuned_model=finetuned_model,
            )

        return wrapper

    return decorator
