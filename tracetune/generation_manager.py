import inspect
import torch
from typing import Callable, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from tracing import TracedProgramState
import concurrent.futures

def _process_single_rollout(response: str, program_state: TracedProgramState, prev_trace_string: str, initial_prompt: str) -> Tuple[TracedProgramState, bool, str]:
        """Process a single rollout by stepping the program state until it is at an LLM call or done."""
        while not program_state.is_llm_call and not program_state.is_done:
            program_state.step()
        
        is_active = not program_state.is_done
        start = len(prev_trace_string) + len(response) - len(initial_prompt)
        new_trace_string = program_state.trace_string[start:]
        return program_state, is_active, new_trace_string

def _step_program_state_with_value(response: str, program_state: TracedProgramState) -> TracedProgramState:
        program_state.step(response.split(program_state.end_token)[0].strip())
        return program_state

def _step_program_state(program_state) -> TracedProgramState:
        """Helper function to step a program state - must be at module level for pickling."""
        program_state.step()
        return program_state


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    max_program_workers: int = 8


@dataclass
class TraceTuneGenerationConfig:
    source_code_path: str
    function_name: str
    function_args: Dict[str, Any]
    llm_input_key: str
    llm_fns: List[str]
    start_token: str = "[LLM]"
    end_token: str = "[/LLM]"
    ignored_vars: List[str] = None

class TraceTuneGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        program: Callable,
        llm_input_key: str,
        is_validation: bool = False,
        ignored_vars: List[str] = None,
        llm_fns: List[str] = None,
        start_token: str = "[LLM]",
        end_token: str = "[/LLM]",
        program_args: Dict[str, Any] = None,
    ):
        """
        Args:
            program_args: Dict[str, Any] = None
                Inputs to the program.
                If None, the program will be traced with no inputs.
                If not None, the program will be traced with the given inputs.
            llm_input_key: str
                The key of the LLM input argument in the program.
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.program = program
        self.ignored_vars = ignored_vars if ignored_vars else []
        self.llm_fns = llm_fns if llm_fns else []
        self.start_token = start_token
        self.end_token = end_token
        self.llm_input_key = llm_input_key
        self.program_args = program_args

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [
            resp.split(self.end_token)[0] + self.end_token 
            if self.end_token in resp 
            else resp + self.end_token
            for resp in responses_str
        ]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str


    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _info_masked_concatenate_with_padding(self, 
                input_ids: torch.Tensor, 
                input_ids_with_info_mask: torch.Tensor,
                responses_ids: torch.Tensor,
                info: bool = None,
                pad_to_left: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate input IDs with responses IDs, optionally with info mask."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [input_ids, responses_ids]
        if info:
            info_mask = torch.full(responses_ids.size(), pad_id, dtype=responses_ids.dtype, device=responses_ids.device) # information mask
            tensors_with_mask = [input_ids_with_info_mask, info_mask]
        else:
            tensors_with_mask = [input_ids_with_info_mask, responses_ids]
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info


    def _rollout_programs(self, program_states: List[TracedProgramState], prev_trace_strings: List[str], responses_str: Optional[List[str]] = None) -> Tuple[List[TracedProgramState], torch.Tensor, torch.Tensor]:
        """
        Rollout programs until they are at an LLM call or done. Returns new trace ids and active mask.

        Args:
            program_states: List[TracedProgramState]
                The program states to rollout.
            prev_trace_strings: List[str]
                The previous trace strings.
            responses_str: Optional[List[str]]
                The LLM response string batch (if None, it means we are at the first step)
        """
        if responses_str is None:
            responses_str = ['' for _ in range(len(program_states))]
            
        inputs = zip(responses_str, program_states, prev_trace_strings, [self.initial_prompt for _ in range(len(program_states))])

        new_trace_strings = [''] * len(program_states)
        new_active_mask = [True] * len(program_states)
        new_program_states = [None] * len(program_states)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_program_workers) as executor:
            futures = {executor.submit(_process_single_rollout, *args): idx for idx, args in enumerate(inputs)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                program_state, is_active, trace_string = future.result()
                new_active_mask[idx] = is_active
                new_trace_strings[idx] = trace_string
                new_program_states[idx] = program_state

        return new_program_states, self._batch_tokenize(new_trace_strings), torch.tensor(new_active_mask, dtype=torch.bool)
    
    
    def _step_programs(self, program_states: List[TracedProgramState], responses_str: List[str]) -> List[TracedProgramState]:
        """Step programs until they are at an LLM call or done. Returns new program states."""

        inputs = zip(responses_str, program_states)

        new_program_states = [None] * len(program_states)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_program_workers) as executor:
            futures = {executor.submit(_step_program_state_with_value, *args): idx for idx, args in enumerate(inputs)}
            for future in concurrent.futures.as_completed(futures):
                program_state = future.result()
                new_program_states[futures[future]] = program_state

        return new_program_states

    def _initialize_program_states(self, detokenized_inputs: List[str]) -> List[TracedProgramState]:
        program_states: List[TracedProgramState] = [
            TracedProgramState(func=self.program, ignored_vars=self.ignored_vars, llm_fns=self.llm_fns, inputs={**self.program_args, self.llm_input_key: input})
            for input in detokenized_inputs
        ]
        new_program_states = [None] * len(program_states)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_program_workers) as executor:
            futures = {executor.submit(_step_program_state, program_state): idx for idx, program_state in enumerate(program_states)}
            for future in concurrent.futures.as_completed(futures):
                new_program_states[futures[future]] = future.result()
            
        return new_program_states

    @property
    def initial_prompt(self):
        prompt = f"""\
Your task is to follow the execution of the following function, step by step:

{inspect.getsource(self.program)}

Any updated variables will be outputted in the following format after the execution of each step:
> <variable_name> = <variable_value>

There are some functions whose outputs are your job to fill in. The values of these function are wrapped in {self.start_token} and {self.end_token} tags. For example:
> <variable_name> = {self.start_token} [YOUR OUTPUT HERE] {self.end_token}

These functions are:
{self.llm_fns}

Here is an example of the first few steps of the execution of this function on an example input:
Inputs:
question = What is the capital of France?
hops = 3
topk = 3

Execution:

Step 1: context = []
> context = []

Step 2: for hop in range(hops):
> hop = 0

Step 3: thought = generate_search_term_thought(question, context)
> thought = {self.start_token} [YOUR THOUGHT HERE] {self.end_token}

Step 4: search_term = generate_search_term(question, context, thought)
> search_term = {self.start_token} [YOUR SEARCH TERM HERE] {self.end_token}

Step 5: context += retrieve(search_term)
> context[0] = [RETRIEVED PASSAGE 1]
> context[1] = [RETRIEVED PASSAGE 2]
> context[2] = [RETRIEVED PASSAGE 3]

Step 6: for hop in range(hops):
> hop = 1
...

Step N: answer = generate_answer(question, context)
> answer = {self.start_token} [YOUR ANSWER HERE] {self.end_token}

---

Now, you will be given a new input followed by the execution trace of the function.

"""
        return prompt
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)

        # initialize program states for each input in the batch
        detokenized_inputs = self.tokenizer.batch_decode(initial_input_ids, skip_special_tokens=True)
        program_states = self._initialize_program_states(detokenized_inputs)

        initial_trace_strings = [self.initial_prompt + program_state.trace_string for program_state in program_states]

        rollings = gen_batch
        
        rollings.batch['input_ids'] = self._batch_tokenize(initial_trace_strings)
        rollings.batch['attention_mask'] = self.tensor_fn.create_attention_mask(rollings.batch['input_ids'])
        rollings.batch['position_ids'] = self.tensor_fn.create_position_ids(rollings.batch['attention_mask'])

        program_states, new_trace_ids, active_mask = self._rollout_programs(program_states=program_states, prev_trace_strings=initial_trace_strings)

        rollings = self._update_rolling_state(rollings=rollings, cur_responses=new_trace_ids)

        program_states = self._initialize_program_states(detokenized_inputs)

        output_ids = rollings.batch['input_ids'][:, -self.config.max_start_length:]

        # initial tokens are info masked
        output_ids_info_mask = torch.full(output_ids.size(), self.tokenizer.pad_token_id, dtype=output_ids.dtype, device=output_ids.device) # information mask

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            rollings = self._update_rolling_state(rollings=rollings, cur_responses=responses_ids)

            output_ids, output_ids_info_mask = self._info_masked_concatenate_with_padding(
                input_ids=output_ids,
                input_ids_with_info_mask=output_ids_info_mask,
                responses_ids=responses_ids,
                info=False,
                pad_to_left=False
            )

            ##### TT STEP #####
            prev_trace_strings = [self.initial_prompt + program_state.trace_string for program_state in program_states]
            program_states = self._step_programs(program_states=program_states, responses_str=responses_str)

            ##### TT ROLLOUT #####
            program_states, new_trace_ids, active_mask = self._rollout_programs(program_states=program_states, prev_trace_strings=prev_trace_strings, responses_str=responses_str)

            ##### TT UPDATE ROLLING STATE WITH NEW TRACE #####
            rollings = self._update_rolling_state(rollings=rollings, cur_responses=new_trace_ids)
           
            ##### UPDATE OUTPUT WITH NEW TRACE WITH INFO MASK #####
            output_ids, output_ids_info_mask = self._info_masked_concatenate_with_padding(
                input_ids=output_ids,
                input_ids_with_info_mask=output_ids_info_mask,
                responses_ids=new_trace_ids,
                info=True,
                pad_to_left=False
            )

        
        meta_info['program_states'] = program_states
        
        return self._compose_final_output(output_ids=output_ids, output_ids_info_mask=output_ids_info_mask, meta_info=meta_info)

    def _compose_final_output(self, output_ids: torch.Tensor,
                            output_ids_info_mask: torch.Tensor,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""        
        final_output = {
            "responses": output_ids,
            "responses_with_info_mask": output_ids_info_mask,
            "prompts": output_ids[:, []],
            "input_ids": output_ids,
            "attention_mask": self.tensor_fn.create_attention_mask(output_ids),
            "info_mask": self.tensor_fn.create_attention_mask(output_ids_info_mask),
        }
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
    