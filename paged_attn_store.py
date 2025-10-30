import os
import torch
import random
from fms.models import get_model
# - call intto ContinuosuBatchingSpyreModelRunner.forward
# - this import done before the optimized model forward
import fms.utils.spyre.paged  # noqa # pylint: disable=unused-import
from fms.utils.generation import pad_input_ids
from typing import List, Dict, Callable
from copy import copy

BLOCK_SIZE: int = 64
_MAX_BATCH: int = 2
_MAX_CONTEXT_LENGTH: int = 128
NUM_BLOCKS = (_MAX_BATCH * _MAX_CONTEXT_LENGTH) // BLOCK_SIZE
MAX_NEW_TOKENS = 8

def prepare_inputs(
    input_ids: torch.tensor,
    block_numbers: List,
    kwargs=None,
    is_prefill: bool = True,
    compile: bool = True,
):

    # assume there is no batch for simplicity
    assert input_ids.shape[0] == 1

    if is_prefill:

        # input_ids, kwargs = adjust_inputs_to_batch(input_ids)
        input_ids, kwargs = pad_input_ids(input_ids, min_pad_length=BLOCK_SIZE)

        # this is the true number of left pads when computing paged attention using a paged kv-cache
        # it may include whole empty pages
        left_padded_prompt_mask = (kwargs["position_ids"] == 0).sum(dim=1) - 1

        # this is the context length for each sequence without pads
        context_lengths_without_pads = (kwargs["position_ids"] != 0).sum(dim=1) + 1

        # this is the context length for each sequence with no empty pages (padded to multiple of 64)
        context_lengths = BLOCK_SIZE * (
            (context_lengths_without_pads + BLOCK_SIZE - 1) // BLOCK_SIZE
        )

        # left_padded_prompt_mask - empty_slots + context_lengths
        current_tkv_mask = torch.fill(context_lengths, torch.max(context_lengths))

        current_tkv = context_lengths[0]
        block_table = [block_numbers.pop(0) for _ in range(current_tkv // BLOCK_SIZE)]
        slot_mapping = []
        for pos_i in range(current_tkv):
            # we may have already popped a block, so index to the proper block
            block_number = block_table[pos_i // BLOCK_SIZE]

            block_offset = pos_i % BLOCK_SIZE
            slot = block_number * BLOCK_SIZE + block_offset
            slot_mapping.append(slot)

        input_ids = input_ids[0][-current_tkv:].unsqueeze(0).clone()
        slot_mapping = (
            torch.tensor(slot_mapping[-current_tkv:], dtype=torch.int64)
            .unsqueeze(0)
            .clone()
        )
        position_ids = (
            kwargs["position_ids"][0][-current_tkv:].unsqueeze(0).clone()
        )

        # This view will result in a discontiguous tensor (creates a new graph during compile)
        # For this reason, we must explicitly make contiguous
        mask = (
            kwargs["mask"][:, -current_tkv:, -current_tkv:]
            .unsqueeze(0)
            .contiguous()
        )

        if compile:

            # batch dynamic
            torch._dynamo.mark_static(input_ids, 0)
            torch._dynamo.mark_static(slot_mapping, 0)
            torch._dynamo.mark_static(position_ids, 0)
            torch._dynamo.mark_static(mask, 0)

            # seq dynamic
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.mark_dynamic(slot_mapping, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(mask, 2)
            torch._dynamo.mark_dynamic(mask, 3)
    else:
        mask = None
        position_ids = kwargs["position_ids"][:, -1:] + 1
        current_tkv_mask = kwargs['current_tkv_mask']
        pos_i = current_tkv_mask[0]
        block_table = kwargs['block_table']
        if isinstance(block_table, torch.Tensor):
            block_table = block_table[0].tolist()

        if pos_i % BLOCK_SIZE == 0:
            block_number = block_numbers.pop(0)
            block_table.append(block_number)

        current_tkv_mask = current_tkv_mask + 1
        block_offset = pos_i % BLOCK_SIZE
        slot = block_table[-1] * BLOCK_SIZE + block_offset
        slot_mapping = torch.tensor([[slot]], dtype=torch.int64)
        left_padded_prompt_mask = kwargs['left_padded_prompt_mask']

        block_table = torch.tensor(
            [
                (
                    [b_seq[0]]
                    * (
                        max(1, max([len(b) for b in [block_table]]))
                        - len(b_seq)
                    )
                )
                + b_seq
                for b_seq in [block_table]
            ],
            dtype=torch.int64,
        )

        if compile:

            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(block_table, 0)
            torch._dynamo.mark_dynamic(slot_mapping, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(current_tkv_mask, 0)
            torch._dynamo.mark_dynamic(left_padded_prompt_mask, 0)

            torch._dynamo.mark_static(input_ids, 1)  # always 1
            torch._dynamo.mark_dynamic(block_table, 1)
            torch._dynamo.mark_static(slot_mapping, 1)  # always 1
            torch._dynamo.mark_static(position_ids, 1)  # always 1

    kwargs['mask'] = mask
    kwargs['position_ids'] = position_ids
    kwargs['slot_mapping'] = slot_mapping
    kwargs['block_table'] = block_table
    kwargs['current_tkv_mask'] = current_tkv_mask
    kwargs['left_padded_prompt_mask'] = left_padded_prompt_mask
    
    # kwargs["use_cache"] = use_cache
    # only_last_token = kwargs.get("only_last_token", False)

    kwargs['attn_name'] = 'spyre_paged_attn'
    # kwargs['past_key_value_states'] = past_key_value_states

    return input_ids, kwargs

def randomize_input_ids(inputs: Dict, vocab_size: int, seed: int = 42):
    torch.manual_seed(42)
    inputs_mod = copy(inputs)
    inputs_mod['input_ids'] = torch.randint(
        0, high=vocab_size,
        size=inputs['input_ids'].shape
    )
    torch._dynamo.mark_dynamic(inputs_mod['input_ids'] , 0)
    torch._dynamo.mark_static(inputs_mod['input_ids'] , 1)
    return inputs_mod

# ---- WARMUP -----


# NOTE: there is some problem with this. 
# - in Spyre notice there this will not actually trigger
# a compilation, only a small number of lines will printput
# INFO:torch_sendnn.backends.sendnn_backend:Checking for dynamic shapes...
# INFO:torch_sendnn.backends.sendnn_backend:We have dynamic shapes for input 0!
# and nothing happens

def prefill(
    model: torch.nn.Module,
    inputs: Dict,
    past_key_value_states: List,
    block_numbers: List,
    input_kwargs: Dict = {},
    load_kvs: List = None,
):

    input_ids, kwargs = prepare_inputs(
        inputs['input_ids'],
        block_numbers=block_numbers,
        is_prefill=True,
        **input_kwargs,
    )

    _prefill_kwargs = {
        k:v for k,v in kwargs.items() if 
        not k in [
            'block_table',
            'current_tkv_mask',
            'left_padded_prompt_mask',
            # 'slot_mapping',
        ]
    }

    if load_kvs is not None:
        _prefill_kwargs['load_kvs'] = load_kvs

    logits, cache = model(
        input_ids,
        past_key_value_states=past_key_value_states,
        use_cache=True,
        only_last_token=False,
        **_prefill_kwargs,
    )

    return (
        logits[:, 0, :] if logits is not None else None, # NOTE WHY WHY WHY?
        cache,
        kwargs
    )

def decode(
    model: torch.nn.Module,
    kwargs: Dict,
    logits: torch.Tensor,
    max_new_tokens: int,
    cache: List,
    block_numbers: List,
    input_kwargs: Dict = {},
):

    results = []
    for _ in range(max_new_tokens):
        next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
        results.append(next_val[0,0].item())

        input_ids, kwargs = prepare_inputs(
            next_val,
            block_numbers=block_numbers,
            kwargs=kwargs,
            is_prefill=False,
            **input_kwargs,
        )

        logits, cache = model(
            input_ids,
            past_key_value_states=cache,
            use_cache=True,
            only_last_token=False,
            **kwargs,
        )
        logits = logits[:, -1, :]

    next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
    results.append(next_val[0,0].item())

    return results

def paged_attn_store(
    cache: List,
    current_kv_cache: List,
    slot_mapping: torch.Tensor,
    pas_kernel: Callable, # this would be the compiled pas kernel
):
    new_kv_cache = []
    for (key, val), (key_store, val_store) in zip(cache, current_kv_cache):

        kvs, vvs = pas_kernel(
            key.transpose(1, 2), val.transpose(1, 2), 
            key_store, val_store, 
            slot_mapping,
        )
        new_kv_cache.append((kvs, vvs))

    return new_kv_cache 

def extract_kvs_from_cache(result_key_cache, result_value_cache, slot_mapping):

    # NOTE: this is not really correct
    keys = torch.zeros((len(slot_mapping), BLOCK_SIZE) + tuple(result_key_cache.shape[-2:]))
    vals = torch.zeros((len(slot_mapping), BLOCK_SIZE) + tuple(result_key_cache.shape[-2:]))
    for seq_i, slot_mapping_seq in enumerate(slot_mapping):
        for tok_i, slot in enumerate(slot_mapping_seq):
            block_number = slot.item() // BLOCK_SIZE
            position = slot.item() % BLOCK_SIZE

            keys[seq_i, tok_i, :, :] = result_key_cache[block_number, position, :, :]
            vals[seq_i, tok_i, :, :] = result_value_cache[block_number, position, :, :]
            
    return keys, vals


if __name__ == '__main__':

    # different test modes
    # - normal decode path
    TEST_MODE_REG = 'regular'

    # - using the paged attention store
    TEST_MODE_PAS = 'pas' # 

    # - by modifying fms.modules.attention.MultiHeadAttention
    #   to read in the key-values from an extra argument 
    TEST_MODE_MOD = 'fake'

    import sys
    TEST_MODE = TEST_MODE_REG
    if len(sys.argv) > 1:
        TEST_MODE = sys.argv[1]
        assert TEST_MODE in {TEST_MODE_REG, TEST_MODE_PAS, TEST_MODE_MOD}

    print ('TESTING MODE:', TEST_MODE)

    os.environ['TORCH_SENDNN_LOG'] = 'INFO'

    # DEBUGGING
    # os.environ['TORCH_COMPILE_DEBUG'] = '1'
    # os.environ['TORCH_LOGS'] = "+dynamo,inductor"
    # os.environ['TORCH_LOGS'] = "+dynamo"
    # os.environ['DEE_DUMP_GRAPHS'] = "1"
    # import torch._logging
    # torch._logging.set_logs(dynamo=True, cudagraph_static_inputs=True, autotuning=True)


    os.environ.setdefault("DATA_PREC", "fp16")
    os.environ.setdefault("FLEX_OVERWRITE_NMB_FRAME", "1")
    os.environ.setdefault("DTCOMPILER_KEEP_EXPORT", "true")

    MODEL='/mnt/models/tiny-granite-3.3-8b'
    os.environ['COMPILATION_MODE'] = 'offline_decoder'
    os.environ['VLLM_DT_MAX_BATCH_SIZE'] = '2'
    os.environ['VLLM_DT_MAX_CONTEXT_LEN'] = '512'
    os.environ['VLLM_DT_MAX_BATCH_TKV_LIMIT'] = '1024'

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
    # prompt = template.format("Write a bubble sort function in python.")
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

    prompt = template.format(
        "Provide a list of instructions for preparing chicken soup."
    )
    # prompt = "<|endoftext|>i need a copy paste tool for zellij. its relaly hard to grab scrollback output cleanly, idk how to output it easily for use outside the terminal. can you help? "
    inputs = tokenizer(prompt, return_tensors="pt")

    # VLLM - 
    # - in model_loader.spyre.FmsModelBase.load_weights
    DTYPE=torch.float16

    # - call get model to load odel
    model = get_model(
        architecture='hf_pretrained',
        variant=MODEL,
        model_path=None,
        device_type="cpu",
        data_type=DTYPE,
        source=None,
        distributed_strategy=None,
        group=None,
        linear_config={"linear_type": "torch_linear"},
        fused_weights=False,
    )

    if TEST_MODE in {TEST_MODE_PAS, TEST_MODE_MOD}:

        # use a cpu model to simulate the prefill in the 
        # different matchine
        model_validation = get_model(
            architecture='hf_pretrained',
            variant=MODEL,
            model_path=None,
            device_type="cpu",
            data_type=DTYPE,
            source=None,
            distributed_strategy=None,
            group=None,
            linear_config={"linear_type": "torch_linear"},
            fused_weights=False,
        )
        model_validation.load_state_dict(model.state_dict())
        model_validation.eval()

    # - set these 
    model.eval()
    torch.set_grad_enabled(False)

    # DONT DO THIS (BAD THINGS WILL HAPPEN)
    # torch._dynamo.config.assume_static_by_default = False
    # torch._dynamo.config.dynamic_shapes = True
    # torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.accumulated_cache_size_limit = 160
    torch._dynamo.config.cache_size_limit = 160

    # Lazy import to avoid load torch_sendnn runtime before it is really
    # necessary. This solve issues of running forked tests that share
    # some resources from parent to children which can have problems
    # of caching even though the test run in isolated subprocesses.
    from torch_sendnn import torch_sendnn

    # - call torch.compile
    model = torch.compile(
        model, backend='sendnn', 
        options={"sendnn.dynamic": True}
    )

    # compile paged attn store in offline mode
    _old_val = os.environ['COMPILATION_MODE'] 
    os.environ['COMPILATION_MODE'] = 'offline'
    pas_kernel = torch.compile(
        fms.utils.spyre.paged.paged_attn_store,
        backend="sendnn"
    )
    os.environ['COMPILATION_MODE'] = _old_val

    # - in spyre_worker.SpyreWorker._warmup_spyre_dynamic_size

    # this setting is required to mark a dimension of size 1 as dynamic
    # for pytorch >= 2.7.1 (needed to support batch size 1 for decodes)
    from torch.fx.experimental import _config as config
    config.backed_size_oblivious = True

    from torch_sendnn import warmup_mode

    # build the blocks
    kvheads = model.config.kvheads
    head_size = model.config.emb_dim // model.config.nheads

    past_key_value_states = [
        (
            torch.zeros(
                NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=DTYPE
            ),
            torch.zeros(
                NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=DTYPE
            ),
        )
        for _ in range(model.config.nlayers)
    ]
    BLOCK_NUMBERS = [2, 0, 1, 3]

    # - warming up
    print("WARMING UP")
    if os.environ.get("DEBUG", "False") == "True":
        print ("DEBUG MODE")
        model = model_validation ## DEBUG

    if TEST_MODE == TEST_MODE_MOD:
        # We simulate a prefill on a different matchine.
        # - this prefill is run on a different model instance
        #   on the cpu.
        # - this will generate the logits and cache to be sent
        #   to the decode machine
        print ('SIMULATING PREFILL ON DIFFERENT MATCHINE')
        logits, cache_prefill, kwargs_prefill = prefill(
            model_validation,
            inputs, 
            past_key_value_states=past_key_value_states, # can try to even change this
            block_numbers=[1,2,3,0], # chose a different set of block numbers
        )
        kvs = []
        for K, V in cache_prefill:
            k, v = extract_kvs_from_cache(K, V, kwargs_prefill['slot_mapping'])
            torch._dynamo.mark_static(k, 0)
            torch._dynamo.mark_dynamic(k, 1)
            torch._dynamo.mark_static(v, 0)
            torch._dynamo.mark_dynamic(v, 1)
            kvs.append((k,v))
    with warmup_mode():
        if TEST_MODE == TEST_MODE_MOD:
            # In this strategy, we compile the model in a "modified"
            # prefill path (which accepts an extra argument "load_kvs")
            # - this will result in the model bypassing short circuiting 
            # the keys, values computation, with that loaded from the 
            # "load_kvs" argument.
            # - in other words, this simulates a pass-through prefill
            #   which is used to load the kv values
            # - the inputs are used only to internally compute the correct
            #   slot mapping.
            # - the logits are not taken from here
            # - pass random input ids through this prefill to ensure that
            #   we are not using the same input ids to compute the prefill
            print ("PASS THROUGH PREFILL TO LOAD KV VALUES")
            _, cache, kwargs = prefill(
                model,
                randomize_input_ids(inputs, model.config.src_vocab_size),
                past_key_value_states=past_key_value_states,
                block_numbers=BLOCK_NUMBERS,
                load_kvs=kvs,
            )
        else:
            logits, cache, kwargs = prefill(
                model,
                inputs, 
                past_key_value_states=past_key_value_states,
                block_numbers=BLOCK_NUMBERS,
            )
        print ("RUN DECODE")
        results = decode(
            model,
            kwargs,
            logits, 
            max_new_tokens=1,
            cache=cache,
            block_numbers=BLOCK_NUMBERS,
        )

    # - first inference after warmup
    # - triggers compile
    print("RUNNING INFERENCE")
    BLOCK_NUMBERS = [2, 0, 1, 3]
    if TEST_MODE == TEST_MODE_MOD:
        _, cache, kwargs = prefill(
            model,
            randomize_input_ids(inputs, model.config.src_vocab_size),
            past_key_value_states=past_key_value_states,
            block_numbers=BLOCK_NUMBERS,
            load_kvs=kvs,
        )
    else:
        logits, cache, kwargs = prefill(
            model,
            inputs, 
            past_key_value_states=past_key_value_states,
            block_numbers=BLOCK_NUMBERS,
        )
        logits_prefill = logits

    results = decode(
        model,
        kwargs,
        logits, 
        max_new_tokens=1,
        cache=cache,
        block_numbers=BLOCK_NUMBERS,
    )
    print (tokenizer.decode(results))

    # - test
    print ("RUNNING TEST")
    BLOCK_NUMBERS = [2, 0, 1, 3]
    prompt = "This is a new instruction that has not yet been seen. Please introduce yourself and tell me a joke."
    inputs_new = tokenizer(prompt, return_tensors="pt")
    # inputs_new = inputs
    if TEST_MODE == TEST_MODE_MOD:
        logits_new, cache_prefill_new, kwargs_prefill_new = prefill(
            model_validation,
            inputs_new, 
            past_key_value_states=past_key_value_states, # can try to even change this
            block_numbers=[1,2,3,0], # maybe not needed
        )
        kvs_new = []
        for K, V in cache_prefill_new:
            k, v = extract_kvs_from_cache(K, V, kwargs_prefill_new['slot_mapping'])
            torch._dynamo.mark_dynamic(k, 1)
            torch._dynamo.mark_dynamic(v, 1)
            kvs_new.append((k,v))
    elif TEST_MODE == TEST_MODE_PAS:
        # - run the model forward with the 
        #   different model instance that is on the 
        #   cpu.
        input_ids, kwargs_new = prepare_inputs(
            inputs_new['input_ids'],
            block_numbers=BLOCK_NUMBERS,
            is_prefill=True,
        )
        logits_new, kvs = model_validation(
            input_ids,
            position_ids=kwargs_new['position_ids'],
            use_cache=True,
            only_last_token=True,
        )
        # - call the paged attention kernel to 
        #   store the key values
        cache_new = paged_attn_store(
            kvs,
            past_key_value_states,
            slot_mapping=kwargs_new['slot_mapping'],
            pas_kernel=pas_kernel,
        )

    if TEST_MODE == TEST_MODE_MOD:
        # - randomize inputs to ensure we are not 
        #   replaying the prefill
        _, cache_new, kwargs_new = prefill(
            model, 
            randomize_input_ids(inputs_new, model.config.src_vocab_size),
            past_key_value_states=past_key_value_states,
            block_numbers=BLOCK_NUMBERS,
            load_kvs=kvs_new,
        )
    elif TEST_MODE == TEST_MODE_REG:
        # - run the regular prefill step
        logits_new, cache_new, kwargs_new = prefill(
            model, inputs_new, 
            past_key_value_states=past_key_value_states,
            block_numbers=BLOCK_NUMBERS,
        )

    print ("LOGITS")
    print(logits_new)

    results_new = decode(
        model,
        kwargs_new,
        logits_new, 
        max_new_tokens=20,
        cache=cache_new,
        block_numbers=BLOCK_NUMBERS,
    )
    print ("FINAL DECODE")
    print (tokenizer.decode(results_new))