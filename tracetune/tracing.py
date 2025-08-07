import ast
import inspect
import os
import pdb
import sys
import copy
import linecache
from inspect import signature
from dataclasses import dataclass
import types
from typing import Any
import cloudpickle

class PauseExecutionException(Exception):
    def __init__(self, is_llm_call=False):
        self.is_llm_call = is_llm_call

def convert_to_hashable_dict(local_dict):
    new_dict = {}
    for k, v in local_dict.items():
        # convert all nonhashable values to strings
        try:
            hash(v)
            new_dict[k] = v
        except:
            new_dict[k] = str(v)
    return new_dict

class ExecutionTracker(pdb.Pdb):
    def __init__(
        self,
        traced_program,
        program_inputs,
        llm_fns=None,
        start_token="[LLM]",
        end_token="[/LLM]",
        ignored_vars=None,
        frame_history=None,
        initial_step_count=0,
        step_value=None,
        *args,
        **kwargs,
    ):
        # super(ExecutionTracker, self).__init__(*args, stdout=open(os.devnull, 'w'), **kwargs)
        super(ExecutionTracker, self).__init__(*args, **kwargs)
        self.traced_program = traced_program
        self.program_inputs = program_inputs

        # for resuming tracing
        self.frame_history = frame_history
        self.initial_step_count = initial_step_count

        self.start_token = start_token
        self.end_token = end_token

        self.ignored_vars = ignored_vars if ignored_vars else []

        # this is a list of functions that are to be treated as llm functions during inference mode
        # if llm_fns = None, llm_fns will be all the dspy modules in the program (based on program self attributes)
        self.llm_fns = llm_fns if llm_fns else []

        self.trace_string = ""

        self.prev_local_dict = None
        self.prev_line = None

        self.tracing_enabled = False
        self.steps = 1
        self.initial_step_count = initial_step_count
        self.first = True
        self.step_done = 0

        self._mocked_functions = {}

        self.step_value = step_value

    def first_write(self):
        self.trace_string += f"Inputs:\n"
        for k, v in self.program_inputs.items():
            if k in self.ignored_vars:
                continue
            self.trace_string += f"{k} = {v}\n"
        
        # handle default values in the function signature
        for k, v in inspect.signature(self.traced_program).parameters.items():
            if k in self.ignored_vars:
                continue
            if k not in self.program_inputs:
                self.trace_string += f"{k} = {v.default}\n"

        self.trace_string += f"\nExecution:\n\n"

    def is_llm_call(self, line):
        # TODO: could be done more robustly by dealing with more reliable types rather than just string matching
        return any([fn in line for fn in self.llm_fns])

    def diff_write(self, prev, current, line):
        diff = dict(set(current.items()) - set(prev.items()))
        diff = sorted(diff.items())

        # if there are no changes, only write the output if the current line is an assignment.
        # this makes it so that re-assignments of variables with the same value are rewritten.
        if not diff and self.steps > 1:
            # check if the line is an assignment
            try:
                tree = ast.parse(line.strip(), mode='exec')
            except:
                self.trace_string += f"> [No change in any variables]\n\n"
                return

            if isinstance(tree.body[0], ast.Assign):
                k = tree.body[0].targets[0].id
                v = current[k]
                if k in self.ignored_vars:
                    return
                if k in self.program_inputs and self.program_inputs[k] == v:
                    return

                self.trace_string += f"> {k} = {v}\n\n"
            else:
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

            if self.llm_fns and llm_fn in line:
                break

            try:
                v = ast.literal_eval(v)
                try:
                    prev_v = ast.literal_eval(prev[k])
                    assert isinstance(prev_v, list)
                except:
                    prev_v = []
                if isinstance(v, list) and len(v) > 0:
                    for i, item in enumerate(v):
                        if isinstance(prev_v, list) and len(prev_v) > i and prev_v[i] == item:
                            continue
                        else:
                            self.trace_string += f"> {k}[{i}] = {v[i]}\n"
                    self.trace_string += "\n"
                    break
            except Exception as e:
                pass

            self.trace_string += f"> {k} = {v}\n\n"

    def trace_dispatch(self, frame: types.FrameType, event: str, arg: Any):
        if not self.tracing_enabled:
            return

        # step into only the first call (program(**program_inputs) call)
        if self.first and event == "call":
            self.set_step()
            self.first = False
            self.steps += 1
            return super().trace_dispatch(frame, event, arg)

        # only trace lines
        if event != "line":
            return
        
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        cur_line = linecache.getline(filename, lineno, frame.f_globals)
        
        # if we have reached the end of the trace, stop tracing
        if cur_line.strip() == "set_trace(end=True)":
            return
        
        # first, restore the original functions of previous steps
        for fn in self._mocked_functions:
            if fn in frame.f_globals:   
                frame.f_globals[fn] = self._mocked_functions[fn]
        self._mocked_functions = {}

        current_local_dict = convert_to_hashable_dict(frame.f_locals)
        if self.steps == self.initial_step_count + 1:
            self.diff_write(
                prev=self.prev_local_dict,
                current=current_local_dict,
                line=self.prev_line,
            )

            self.frame_history.append(Frame(frame))
            self.trace_string += f"Step {self.steps - 1}: {linecache.getline(frame.f_code.co_filename, frame.f_lineno, frame.f_globals).strip()}\n>"
            if self.is_llm_call(cur_line):
                var_name = cur_line.split("=")[0].strip()
                self.trace_string += f" {var_name} = {self.start_token} "
            raise PauseExecutionException(is_llm_call=self.is_llm_call(cur_line))


        if self.prev_local_dict is None:
            # this is the first call after setting our debugger
            self.first_write()
        else:
            current_local_dict = convert_to_hashable_dict(frame.f_locals)
            self.diff_write(
                prev=self.prev_local_dict,
                current=current_local_dict,
                line=self.prev_line,
            )

        # if we are continuing the trace, we need to mock the functions that are being called
        if self.steps <= self.initial_step_count and self.steps > 1:
            # then, mock the functions that are being called

            error_parsing = False
            target = None
            # parse ast of cur_line, if it fails, there is no function to mock on this line
            try:
                ast_node = ast.parse(cur_line.strip()).body[0]
            except Exception as e:
                error_parsing = True
            #TODO: assumes only one target per assign
            if not error_parsing:

                # get the target of the assignment
                if isinstance(ast_node, ast.Assign):
                    assert len(ast_node.targets) == 1, "Only one target per assign currently supported for Assign nodes"
                    target = ast_node.targets[0]
                # supports += for list append
                elif isinstance(ast_node, ast.AugAssign) and isinstance(ast_node.op, ast.Add):
                    target = ast_node.target

                if target and isinstance(target, ast.Name) and isinstance(ast_node.value, ast.Call):
                    # if the function is not mocked yet, mock it
                    original_func = frame.f_globals[ast_node.value.func.id]
                    if ast_node.value.func.id not in self._mocked_functions:
                        self._mocked_functions[ast_node.value.func.id] = original_func

                    if self.steps < self.initial_step_count:
                        v = self.frame_history[self.steps-1].locals[target.id]
                        if isinstance(ast_node, ast.Assign):
                            frame.f_globals[ast_node.value.func.id] = lambda *args, **kwargs: v
                        # for list append
                        else:
                            if ast_node.value.func.id in self.frame_history[self.steps-2].globals:
                                new_items_len = len(self.frame_history[self.steps-1].locals[target.id]) - len(self.frame_history[self.steps-2].locals[target.id])
                                frame.f_globals[ast_node.value.func.id] = lambda *args, **kwargs: v[-new_items_len:]
                            else:
                                frame.f_globals[ast_node.value.func.id] = lambda *args, **kwargs: v

                    elif self.steps == self.initial_step_count and self.step_value is not None:
                        frame.f_globals[ast_node.value.func.id] = lambda *args, **kwargs: self.step_value

        self.prev_local_dict = copy.copy(current_local_dict)
        self.prev_line = cur_line

        # # don't write the first line (func() call)
        if self.first and event == "line":
            return

        self.trace_string += f"Step {self.steps - 1}: " + cur_line.strip() + "\n"
        self.steps += 1


@dataclass
class TracedProgramOutput:
    inputs: dict
    output: any
    trace_string: str
    llm_fns: list
    execution_tracker: ExecutionTracker

class Frame:
    def __init__(self, frame: types.FrameType):
        self.filename = frame.f_code.co_filename
        self.function = frame.f_code.co_name
        self.lineno = frame.f_lineno
        self.globals = {}
        for key, value in frame.f_globals.items():
            try:
                cloudpickle.dumps(value)
                self.globals[key] = copy.deepcopy(value)
            except :
                self.globals[key] = str(value)

        self.locals = {}
        for key, value in frame.f_locals.items():
            try:
                cloudpickle.dumps(value)
                self.locals[key] = copy.deepcopy(value)
            except:
                self.locals[key] = str(value)

class TracedProgramState:
    def __init__(self, func, llm_fns, ignored_vars=None, start_token="[LLM]", end_token="[/LLM]", step_count=1, frame_history=None, inputs: dict = None, output=None) -> None:
        self.func = func
        self.llm_fns = llm_fns
        self.ignored_vars = ignored_vars if ignored_vars else []
        self.step_count = step_count
        self.frame_history: list[Frame] = frame_history if frame_history else []
        self.inputs = inputs if inputs else {}
        self.output = output
        self.is_llm_call = False
        self.start_token = start_token
        self.end_token = end_token
        self.trace_string = ""

    def step(self, step_value=None):
        # Create a new execution tracker for each step
        execution_tracker = ExecutionTracker(
            program_inputs=self.inputs,
            traced_program=self.func,
            llm_fns=self.llm_fns,
            ignored_vars=self.ignored_vars,
            frame_history=self.frame_history,
            initial_step_count=self.step_count,
            step_value=step_value,
            start_token=self.start_token,
            end_token=self.end_token,
        )

        def set_trace(end=False):
            if end:
                execution_tracker.tracing_enabled = False
                execution_tracker.trace_file = None
                return

            execution_tracker.rcLines.append("next")
            execution_tracker.set_trace(sys._getframe().f_back)
            execution_tracker.tracing_enabled = True
        output = None
        try:
            set_trace()  # Start the debugger here

            output = self.func(**self.inputs)
            
            set_trace(end=True)
            self.output = output

        except PauseExecutionException as e:
            self.is_llm_call = e.is_llm_call
        # except Exception as e:
        #     print(f"Error setting trace: {e}")
        self.step_count += 1
        self.trace_string = execution_tracker.trace_string
        return self
    
    @property
    def is_done(self):
        return self.output is not None


def trace(llm_fns=None, ignored_vars=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg, param in zip(args, signature(func).parameters):
                kwargs[param] = arg

            for param in signature(func).parameters.values():
                if param.default != param.empty and param.name not in kwargs:
                    kwargs[param.name] = param.default

            return TracedProgramState(func, llm_fns, ignored_vars, inputs=kwargs).step()
        return wrapper
    return decorator
