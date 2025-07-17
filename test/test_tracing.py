from tracetune.tracing import TracedProgramState

counter = 0
mock_counters = {
    'generate_search_term_thought': 0,
    'generate_search_term': 0,
    'retrieve': 0,
    'generate_answer': 0
}

def reset_counters():
    global counter
    counter = 0
    for key in mock_counters:
        mock_counters[key] = 0

def generate_search_term_thought(question, context=[]):
    global counter
    counter += 1
    return f"test {counter}"

# Mock functions with counters
def mock_generate_search_term_thought(question, context=[]):
    mock_counters['generate_search_term_thought'] += 1
    return f"mock_thought_{mock_counters['generate_search_term_thought']}"

def mock_generate_search_term(question, reasoning, context=[]):
    mock_counters['generate_search_term'] += 1
    return f"mock_search_term_{mock_counters['generate_search_term']}"

def mock_retrieve(query: str, k: int = 1):
    mock_counters['retrieve'] += 1
    return [f"mock_context_{mock_counters['retrieve']}_{query}"]

def mock_generate_answer(question, context):
    mock_counters['generate_answer'] += 1
    return f"mock_answer_{mock_counters['generate_answer']}"

def multi_hop(question, hops=3):
    dummy_initial_thought = mock_generate_search_term_thought(question, [])
    context = []

    for hop in range(hops):
        thought = mock_generate_search_term_thought(question, context)
        search_term = mock_generate_search_term(question, context, thought)
        context += mock_retrieve(search_term)

    answer = mock_generate_answer(question, context)
    
    return answer

class CumulativeAssert:
    def __init__(self, program_state):
        self.program_state = program_state
        self.asserts = []

    def __call__(self, check_str):
        self.asserts.append(check_str)
        for a in self.asserts:
            assert a in self.program_state.trace_string

def test_tracing_no_step_values():
    reset_counters()
    traced_multi_hop = TracedProgramState(func=multi_hop, llm_fns=["mock_generate_search_term_thought", "mock_generate_search_term", "mock_generate_answer"], inputs={"question": "What is the capital of France?"})
    asserts = CumulativeAssert(traced_multi_hop)
    
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("question = What is the capital of France?")
    asserts("hops = 3")
    asserts("dummy_initial_thought = mock_generate_search_term_thought(question, [])")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"dummy_initial_thought = {traced_multi_hop.start_token} mock_thought_1 {traced_multi_hop.end_token}")
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = []")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 0")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"thought = {traced_multi_hop.start_token} mock_thought_2 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"search_term = {traced_multi_hop.start_token} mock_search_term_1 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")
    
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1']")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 1")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"thought = {traced_multi_hop.start_token} mock_thought_3 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"search_term = {traced_multi_hop.start_token} mock_search_term_2 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1', 'mock_context_2_mock_search_term_2']")
    asserts("for hop in range(hops):")


    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 2")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"thought = {traced_multi_hop.start_token} mock_thought_4 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"search_term = {traced_multi_hop.start_token} mock_search_term_3 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1', 'mock_context_2_mock_search_term_2', 'mock_context_3_mock_search_term_3']")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("[No change in any variables]")
    asserts("answer = mock_generate_answer(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"answer = {traced_multi_hop.start_token} mock_answer_1 {traced_multi_hop.end_token}")
    asserts("return answer")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()

    assert traced_multi_hop.is_done
    assert not traced_multi_hop.is_llm_call
    assert traced_multi_hop.output == "mock_answer_1"

def test_tracing_first_step_with_value():
    reset_counters()
    traced_multi_hop = TracedProgramState(func=multi_hop, llm_fns=["mock_generate_search_term_thought", "mock_generate_search_term", "mock_generate_answer"], inputs={"question": "What is the capital of France?"})
    asserts = CumulativeAssert(traced_multi_hop)
    
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("question = What is the capital of France?")
    asserts("hops = 3")
    asserts("dummy_initial_thought = mock_generate_search_term_thought(question, [])")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step("test_thought_1")
    asserts(f"dummy_initial_thought = {traced_multi_hop.start_token} test_thought_1 {traced_multi_hop.end_token}")
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = []")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 0")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"thought = {traced_multi_hop.start_token} mock_thought_1 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"search_term = {traced_multi_hop.start_token} mock_search_term_1 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")
    
    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1']")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 1")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step("test_thought_2")
    asserts(f"thought = {traced_multi_hop.start_token} test_thought_2 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step("test_search_term_1")
    asserts(f"search_term = {traced_multi_hop.start_token} test_search_term_1 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1', 'mock_context_2_test_search_term_1']")
    asserts("for hop in range(hops):")


    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("hop = 2")
    asserts("thought = mock_generate_search_term_thought(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"thought = {traced_multi_hop.start_token} mock_thought_2 {traced_multi_hop.end_token}")
    asserts("search_term = mock_generate_search_term(question, context, thought)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"search_term = {traced_multi_hop.start_token} mock_search_term_2 {traced_multi_hop.end_token}")
    asserts("context += mock_retrieve(search_term)")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("context = ['mock_context_1_mock_search_term_1', 'mock_context_2_test_search_term_1', 'mock_context_3_mock_search_term_2']")
    asserts("for hop in range(hops):")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts("[No change in any variables]")
    asserts("answer = mock_generate_answer(question, context)")

    assert traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()
    asserts(f"answer = {traced_multi_hop.start_token} mock_answer_1 {traced_multi_hop.end_token}")
    asserts("return answer")

    assert not traced_multi_hop.is_llm_call
    assert not traced_multi_hop.is_done
    traced_multi_hop.step()

    assert traced_multi_hop.is_done
    assert not traced_multi_hop.is_llm_call
    assert traced_multi_hop.output == "mock_answer_1"