import ctypes
from ctypes import c_void_p, c_char_p, c_int, c_uint, c_bool, c_uint32, c_size_t, c_float
from typing import List, AsyncGenerator
import asyncio
import concurrent.futures

# Load the shared library
lib = ctypes.CDLL("/home/blackroot/Desktop/YALS/lib/deno_cpp_binding.so")  # Replace with the actual path

# Define structures
class llama_model(ctypes.Structure):
    pass

class llama_logit_bias(ctypes.Structure):
    _fields_ = [("token", ctypes.c_int32),
                ("bias", ctypes.c_float)]

# Function bindings
lib.LoadModel.argtypes = [c_char_p, c_int]
lib.LoadModel.restype = c_void_p

lib.InitiateCtx.argtypes = [c_void_p, c_uint, c_uint]
lib.InitiateCtx.restype = c_void_p

lib.CreateReadbackBuffer.argtypes = []
lib.CreateReadbackBuffer.restype = c_void_p

lib.ReadbackNext.argtypes = [c_void_p]
lib.ReadbackNext.restype = c_void_p

lib.WriteToReadbackBuffer.argtypes = [c_void_p, c_char_p]
lib.WriteToReadbackBuffer.restype = None

lib.IsReadbackBufferDone.argtypes = [c_void_p]
lib.IsReadbackBufferDone.restype = c_bool

# Sampler functions
lib.MakeSampler.argtypes = []
lib.MakeSampler.restype = c_void_p

lib.DistSampler.argtypes = [c_void_p, c_uint32]
lib.DistSampler.restype = c_void_p

lib.GrammarSampler.argtypes = [c_void_p, ctypes.POINTER(llama_model), c_char_p, c_char_p]
lib.GrammarSampler.restype = c_void_p

lib.GreedySampler.argtypes = [c_void_p]
lib.GreedySampler.restype = c_void_p

lib.InfillSampler.argtypes = [c_void_p, ctypes.POINTER(llama_model)]
lib.InfillSampler.restype = c_void_p

lib.LogitBiasSampler.argtypes = [c_void_p, ctypes.POINTER(llama_model), c_size_t, ctypes.POINTER(llama_logit_bias)]
lib.LogitBiasSampler.restype = c_void_p

lib.MinPSampler.argtypes = [c_void_p, c_float, c_size_t]
lib.MinPSampler.restype = c_void_p

lib.MirostatSampler.argtypes = [c_void_p, c_int, c_uint32, c_float, c_float, c_int]
lib.MirostatSampler.restype = c_void_p

lib.MirostatV2Sampler.argtypes = [c_void_p, c_uint32, c_float, c_float]
lib.MirostatV2Sampler.restype = c_void_p

lib.PenaltiesSampler.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_bool, c_bool]
lib.PenaltiesSampler.restype = c_void_p

lib.SoftmaxSampler.argtypes = [c_void_p]
lib.SoftmaxSampler.restype = c_void_p

lib.TailFreeSampler.argtypes = [c_void_p, c_float, c_size_t]
lib.TailFreeSampler.restype = c_void_p

lib.TempSampler.argtypes = [c_void_p, c_float]
lib.TempSampler.restype = c_void_p

lib.TempExtSampler.argtypes = [c_void_p, c_float, c_float, c_float]
lib.TempExtSampler.restype = c_void_p

lib.TopKSampler.argtypes = [c_void_p, c_int]
lib.TopKSampler.restype = c_void_p

lib.TopPSampler.argtypes = [c_void_p, c_float, c_size_t]
lib.TopPSampler.restype = c_void_p

lib.TypicalSampler.argtypes = [c_void_p, c_float, c_size_t]
lib.TypicalSampler.restype = c_void_p

lib.XtcSampler.argtypes = [c_void_p, c_float, c_float, c_size_t, c_uint32]
lib.XtcSampler.restype = c_void_p

lib.Infer.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_uint]
lib.Infer.restype = None

lib.InferToReadbackBuffer.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_char_p, c_uint]
lib.InferToReadbackBuffer.restype = None

class LogitBias:
    def __init__(self, token: int, bias: float):
        self.token = token
        self.bias = bias

class SamplerBuilder:
    def __init__(self, lib, model: c_void_p):
        self.lib = lib
        self.sampler = self.lib.MakeSampler()
        self.model = model

    def dist_sampler(self, seed: int) -> 'SamplerBuilder':
        self.sampler = self.lib.DistSampler(self.sampler, seed)
        return self

    def grammar_sampler(self, model: c_void_p, grammar: str, root: str) -> 'SamplerBuilder':
        grammar_bytes = grammar.encode('utf-8') + b'\0'
        root_bytes = root.encode('utf-8') + b'\0'
        self.sampler = self.lib.GrammarSampler(
            self.sampler, model, ctypes.c_char_p(grammar_bytes), ctypes.c_char_p(root_bytes)
        )
        return self

    def greedy(self) -> 'SamplerBuilder':
        self.sampler = self.lib.GreedySampler(self.sampler)
        return self

    def infill_sampler(self, model: c_void_p) -> 'SamplerBuilder':
        self.sampler = self.lib.InfillSampler(self.sampler, model)
        return self

    def logit_bias_sampler(self, logit_bias: List[LogitBias]) -> 'SamplerBuilder':
        n_bias = len(logit_bias)
        bias_array = (llama_logit_bias * n_bias)()
        for i, bias in enumerate(logit_bias):
            bias_array[i].token = bias.token
            bias_array[i].bias = bias.bias
        self.sampler = self.lib.LogitBiasSampler(self.sampler, self.model, n_bias, bias_array)
        return self

    def min_p_sampler(self, min_p: float, min_keep: int) -> 'SamplerBuilder':
        self.sampler = self.lib.MinPSampler(self.sampler, min_p, min_keep)
        return self

    def mirostat_sampler(self, n_vocab: int, seed: int, tau: float, eta: float, m: int) -> 'SamplerBuilder':
        self.sampler = self.lib.MirostatSampler(self.sampler, n_vocab, seed, tau, eta, m)
        return self

    def mirostat_v2_sampler(self, seed: int, tau: float, eta: float) -> 'SamplerBuilder':
        self.sampler = self.lib.MirostatV2Sampler(self.sampler, seed, tau, eta)
        return self

    def penalties_sampler(self, n_vocab: int, eos_token: int, nl_token: int, penalty_last_n: int,
                          penalty_repeat: float, penalty_freq: float, penalty_present: float,
                          penalize_nl: bool, ignore_eos: bool) -> 'SamplerBuilder':
        self.sampler = self.lib.PenaltiesSampler(
            self.sampler, n_vocab, eos_token, nl_token, penalty_last_n,
            penalty_repeat, penalty_freq, penalty_present, penalize_nl, ignore_eos
        )
        return self

    def softmax_sampler(self) -> 'SamplerBuilder':
        self.sampler = self.lib.SoftmaxSampler(self.sampler)
        return self

    def tail_free_sampler(self, z: float, min_keep: int) -> 'SamplerBuilder':
        self.sampler = self.lib.TailFreeSampler(self.sampler, z, min_keep)
        return self

    def temp_sampler(self, temp: float) -> 'SamplerBuilder':
        self.sampler = self.lib.TempSampler(self.sampler, temp)
        return self

    def temp_ext_sampler(self, temp: float, dynatemp_range: float, dynatemp_exponent: float) -> 'SamplerBuilder':
        self.sampler = self.lib.TempExtSampler(self.sampler, temp, dynatemp_range, dynatemp_exponent)
        return self

    def top_k(self, num: int) -> 'SamplerBuilder':
        self.sampler = self.lib.TopKSampler(self.sampler, num)
        return self

    def top_p(self, p: float, min_keep: int) -> 'SamplerBuilder':
        self.sampler = self.lib.TopPSampler(self.sampler, p, min_keep)
        return self

    def typical_sampler(self, typical_p: float, min_keep: int) -> 'SamplerBuilder':
        self.sampler = self.lib.TypicalSampler(self.sampler, typical_p, min_keep)
        return self

    def xtc_sampler(self, xtc_probability: float, xtc_threshold: float, min_keep: int, seed: int) -> 'SamplerBuilder':
        self.sampler = self.lib.XtcSampler(self.sampler, xtc_probability, xtc_threshold, min_keep, seed)
        return self

    def build(self) -> c_void_p:
        return self.sampler

class ReadbackBuffer:
    def __init__(self, lib):
        self.lib = lib
        self.buffer_ptr = self.lib.CreateReadbackBuffer()

    def _read_next(self) -> str | None:
        string_ptr = self.lib.ReadbackNext(self.buffer_ptr)
        if string_ptr is None:
            return None
        return ctypes.cast(string_ptr, ctypes.c_char_p).value.decode('utf-8')

    def _is_done(self) -> bool:
        return self.lib.IsReadbackBufferDone(self.buffer_ptr)

    @staticmethod
    async def _sleep(ms: int):
        await asyncio.sleep(ms / 1000)  # Convert milliseconds to seconds

    async def read(self) -> AsyncGenerator[str, None]:
        while True:
            next_string = self._read_next()
            if next_string is None:
                await self._sleep(10)
                continue
            yield next_string
            if self._is_done():
                break

# Example usage
def load_model(model_path, num_gpu_layers):
    return lib.LoadModel(model_path.encode(), num_gpu_layers)

def initiate_ctx(llama_model, context_length, num_batches):
    return lib.InitiateCtx(llama_model, context_length, num_batches)

def __infer_to_readback(llama_model: c_void_p, sampler: c_void_p, context: c_void_p,
                      readback_buffer: ReadbackBuffer, prompt: str, number_tokens_to_predict: int):
    prompt_bytes = prompt.encode('utf-8')
    c_prompt = ctypes.c_char_p(prompt_bytes)

    return lib.InferToReadbackBuffer(llama_model, sampler, context, readback_buffer.buffer_ptr,
                                     c_prompt, c_uint(number_tokens_to_predict))

async def async_infer_to_readback(llama_model: c_void_p, sampler: c_void_p, context: c_void_p,
                                  readback_buffer: ReadbackBuffer, prompt: str, number_tokens_to_predict: int):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, __infer_to_readback, llama_model, sampler, context,
                                   readback_buffer, prompt, number_tokens_to_predict)

async def main():
    model_path = "/home/blackroot/Desktop/YALS/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    num_gpu_layers = 999
    context_length = 2048
    num_batches = 32

    model = load_model(model_path, num_gpu_layers)
    ctx = initiate_ctx(model, context_length, num_batches)

    sampler_builder = SamplerBuilder(lib, model)
    sampler = (sampler_builder
               .temp_sampler(0.8)
               .top_k(40)
               .top_p(0.95, 1)
               .dist_sampler(1337)
               .build())

    readback_buffer = ReadbackBuffer(lib)

    inference_task = asyncio.create_task(async_infer_to_readback(model, sampler, ctx, readback_buffer, "Wello Herld", 100))

    async for text in readback_buffer.read():
        print(text, end='', flush=True)

    await inference_task

if __name__ == "__main__":
    asyncio.run(main())

