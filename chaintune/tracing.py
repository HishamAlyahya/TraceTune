import sys
import inspect

def trace_function(frame, event, arg):
    if event == "line":
        # Get the current line number and source code
        lineno = frame.f_lineno
        code = frame.f_code
        filename = code.co_filename
        line = inspect.getsourcelines(code)[0][lineno - (inspect.getsourcelines(code)[1] + 1)].strip()
        
        # Get the local variables in the current frame
        local_vars = frame.f_locals.copy()
        
    return trace_function

def trace_function(frame, event, arg, trace_output, target_func):
    if frame.f_code.co_name != target_func.__name__:
        return  # Skip tracing if it's not the target function

    if event == "line":
        # Get the current line number and source code
        lineno = frame.f_lineno
        code = frame.f_code
        line = inspect.getsourcelines(code)[0][lineno - (inspect.getsourcelines(code)[1] + 1)].strip()
        
        # Get the local variables in the current frame
        local_vars = frame.f_locals.copy()

        # Append line info and local variables to the trace output
        trace_output.append((lineno, line, local_vars))

    return lambda f, e, a: trace_function(f, e, a, trace_output, target_func)


def trace(func, inputs):
    trace_output = []

    def wrapped_function(*args, **kwargs):
        sys.settrace(lambda f, e, a: trace_function(f, e, a, trace_output, func))
        try:
            # Call the target function
            result = func(*args, **kwargs)
        finally:
            sys.settrace(None)  # Disable tracing after the function is done
        return trace_output

    return wrapped_function