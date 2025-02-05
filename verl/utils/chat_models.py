import os
import ray
import time
import openai
import anthropic
from together import Together
from tqdm import tqdm
from vllm import LLM, SamplingParams
import openai
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


from transformers import AutoTokenizer
from data.model_prompt_templates.chat_model_templates import get_chat_model_templates
# from src.utils.language_models import VLLM
from src.utils.main_utils import *


def get_chat_model(model_name, config=None):
    model_config = {"model_name": model_name}
    if config is not None:
        model_config.update(config)

    if "is_together" in model_config and model_config["is_together"]:
        return TogetherAIModel(model_config)
    elif "claude" in model_name:
        return ClaudeModel(model_config)
    elif "gpt" in model_name or "o1" in model_name:
        return GPTModel(model_config)
    else:
        return HFChatModel(model_config)


class Model:
    def __init__(self, config):
        self.model_name = config["model_name"]
        print(f"==> Loading model: {self.model_name} <==")

    def _init_default_config(self, model_specific_default_config, config):
        self.config = {
            "tqdm": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_tokens": 512,
            "is_show_prompt": False,
            "n": 1,
        }
        self.config.update(model_specific_default_config)
        self.config.update(config)

    def update_config(self, config):
        self.config.update(config)

    def generate(self, prompt):
        """
        Generate model responses for a single prompt.
        """
        return self.generate_messages([{"role": "user", "content": prompt}])

    def batch_generate(self, prompts):
        """
        Generate model responses for a list of prompts.
        """
        list_of_messages = [[{"role": "user", "content": p}] for p in prompts]
        return self.batch_generate_messages(list_of_messages)

    def get_config(self):
        return self.config


class HFChatModel(Model):
    def __init__(self, config):
        Model.__init__(self, config)
        self._load_config(config)

        self.model = VLLM.remote(self.model_name, **self.config)
        ray.get(self.model.is_initialized.remote())  # init ray model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_config(self, config):
        """
        Load the default config and update it with the user provided config.
        """
        model_specific_default_config = {
            "n_devices": 8,
            "return_full_outputs": False,
            "use_beam_search": False,
            "n": 1,
        }
        self._init_default_config(model_specific_default_config, config)

    def generate_messages(self, messages):
        """
        Generate model responses for a single set of messages.
        """
        return self.batch_generate_messages([messages])[0]

    def batch_generate_messages(self, list_of_messages):
        """
        Generate model responses for a batch of messages.
        """
        list_of_completions = self.model.batch_generate_messages.remote(messages=list_of_messages,
                                                                        tokenizer=self.tokenizer,
                                                                        use_tqdm=self.config["tqdm"],
                                                                        return_full_outputs=self.config[
                                                                            "return_full_outputs"],
                                                                        temperature=self.config["temperature"],
                                                                        top_p=self.config["top_p"],
                                                                        max_tokens=self.config["num_tokens"],
                                                                        is_show_prompt=self.config["is_show_prompt"],
                                                                        n=self.config["n"])
        list_of_completions = ray.get(list_of_completions)
        return list_of_completions


class GPTModel(Model):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(self, config):
        Model.__init__(self, config)
        if "top_logprobs" not in config:
            config["top_logprobs"] = None
        if "logprobs" not in config:
            config["logprobs"] = None
        self._load_config(config)
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _load_config(self, config):
        """
        Load the default config and update it with the user provided config.
        """
        model_specific_default_config = {
            "system_message": get_chat_model_templates("gpt")["system_message"],
        }
        self._init_default_config(model_specific_default_config, config)

    def generate_messages(self, messages):
        """
        Generate model responses for a single set of messages.
        """
        validate_messages_format(messages)

        for _ in range(self.API_MAX_RETRY):
            try:
                if "o1" in self.config["model_name"]:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.config["temperature"],
                        # logprobs=self.config["logprobs"],
                        # top_logprobs=self.config["top_logprobs"],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.config["num_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                    )
                print(response)
                if not self.config["logprobs"]:
                    output = response.choices[0].message.content
                else:
                    output = response.choices[0]
                break
            except Exception as e:
                print("ERROR:", e, type(e))
                time.sleep(self.API_RETRY_SLEEP)
                if type(e) == openai.BadRequestError:
                    return "I'm sorry but the prompt is invalid. Please try again."
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batch_generate_messages(self, list_of_messages):
        """
        Generate model responses for a batch of messages.
        """
        # add system message if not None
        system_message = self.config["system_message"]
        for messages in list_of_messages:
            if system_message is not None:
                messages = [
                    {"role": "system", "content": system_message}] + messages
            else:
                messages = messages

        if self.config["is_show_prompt"]:
            print("=" * 30, "An example of formatted prompt", "=" * 30)
            print(list_of_messages[0])
            print("=" * 92)

        if self.config["tqdm"]:
            list_of_messages = tqdm(list_of_messages)

        return [self.generate_messages(messages) for messages in list_of_messages]


class TogetherAIModel(Model):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(self, config):
        Model.__init__(self, config)
        self._load_config(config)
        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

    def _load_config(self, config):
        """
        Load the default config and update it with the user provided config.
        """
        model_specific_default_config = {
        }
        self._init_default_config(model_specific_default_config, config)

    def generate_messages(self, messages):
        """
        Generate model responses for a single set of messages.
        """
        validate_messages_format(messages)

        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.config["num_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                )
                output = response.choices[0].message.content
                break
            except Exception as e:
                print("ERROR:", e, type(e))
                time.sleep(self.API_RETRY_SLEEP)
                if type(e) == openai.BadRequestError:
                    return "I'm sorry but the prompt is invalid. Please try again."
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batch_generate_messages(self, list_of_messages):
        """
        Generate model responses for a batch of messages.
        """
        # add system message if not None
        system_message = self.config["system_message"]
        for messages in list_of_messages:
            if system_message is not None:
                messages = [
                    {"role": "system", "content": system_message}] + messages
            else:
                messages = messages

        if self.config["is_show_prompt"]:
            print("=" * 30, "An example of formatted prompt", "=" * 30)
            print(list_of_messages[0])
            print("=" * 92)

        if self.config["tqdm"]:
            list_of_messages = tqdm(list_of_messages)

        return [self.generate_messages(messages) for messages in list_of_messages]


class ClaudeModel(Model):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(self, config):
        Model.__init__(self, config)
        if "system_message" not in config:
            config["system_message"] = None
        self._load_config(config)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get('CLAUDE_API_KEY'))

    def _load_config(self, config):
        """
        Load the default config and update it with the user provided config.
        """
        model_specific_default_config = {
        }
        self._init_default_config(model_specific_default_config, config)

    def generate_messages(self, messages):
        """
        Generate model responses for a single set of messages.
        """
        validate_messages_format(messages)

        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.config["num_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                )
                output = response.content[0].text
                break
            except Exception as e:
                print("ERROR:", e, type(e))
                time.sleep(self.API_RETRY_SLEEP)
                if type(e) == openai.BadRequestError:
                    return "I'm sorry but the prompt is invalid. Please try again."
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batch_generate_messages(self, list_of_messages):
        """
        Generate model responses for a batch of messages.
        """
        # add system message if not None
        system_message = self.config["system_message"]
        for messages in list_of_messages:
            if system_message is not None:
                messages = [
                    {"role": "system", "content": system_message}] + messages
            else:
                messages = messages

        if self.config["is_show_prompt"]:
            print("=" * 30, "An example of formatted prompt", "=" * 30)
            print(list_of_messages[0])
            print("=" * 92)

        if self.config["tqdm"]:
            list_of_messages = tqdm(list_of_messages)

        return [self.generate_messages(messages) for messages in list_of_messages]


@ray.remote
class VLLM:
    def __init__(self,
                 model_name_or_path,
                 n_devices=1,
                 **model_kwargs):
        self.model_name = model_name_or_path
        self.model = self.load_vllm_model(
            model_name_or_path, num_gpus=n_devices, **model_kwargs)

    def _apply_chat_template(self, messages: list,
                             tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        formatted_prompts = [
            tokenizer.apply_chat_template(p,
                                          tokenize=False,
                                          add_generation_prompt=True) for p in messages
        ]
        return formatted_prompts

    def batch_generate_messages(self,
                                messages: list,
                                tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
                                use_tqdm: bool = False,
                                # Whether to return the direct vllm output objects
                                n: int = 1,
                                return_full_outputs: bool = False,
                                temperature: float = 1.0,
                                top_p: float = 1.0,
                                max_tokens: int = 2048,
                                logprobs: int = 0,
                                prompt_logprobs: int = 0,
                                is_show_prompt: bool = True,
                                **sampling_args
                                ):

        formatted_prompts = self._apply_chat_template(messages, tokenizer)

        if is_show_prompt:
            print("=" * 30, "An example of formatted prompt", "=" * 30)
            print(formatted_prompts[0])
            print("=" * 92)

        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            **sampling_args
        )
        outputs = self.model.generate(
            prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        if return_full_outputs:
            return outputs

        results = [it.outputs[0].text for it in outputs]
        return results

    def batch_generate_prompts(self,
                               prompts: list[str],
                               do_chat_formatting: bool = False,
                               system_message: str = None,
                               tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
                               use_tqdm: bool = False,
                               # Whether to return the direct vllm output objects
                               n: int = 1,
                               return_full_outputs: bool = False,
                               temperature: float = 1.0,
                               top_p: float = 1.0,
                               max_tokens: int = 2048,
                               logprobs: int = 0,
                               prompt_logprobs: int = 0,
                               is_show_prompt: bool = True,
                               **sampling_args):
        if do_chat_formatting:
            assert tokenizer is not None, "Chat formatting requires tokenizer"
            if system_message is not None:
                conversation_prompts = [[{'role': 'system', 'content': system_message}, {'role': 'user', 'content': p}]
                                        for p in prompts]
            else:
                conversation_prompts = [
                    [{'role': 'user', 'content': p}] for p in prompts]
            formatted_prompts = [tokenizer.apply_chat_template(
                p, tokenize=False) for p in conversation_prompts]
        else:
            formatted_prompts = prompts

        if is_show_prompt:
            print("=" * 30, "An example of formatted prompt", "=" * 30)
            print(formatted_prompts[0])
            print("=" * 92)

        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            **sampling_args
        )
        outputs = self.model.generate(
            prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        if return_full_outputs:
            return outputs

        results = [it.outputs[0].text for it in outputs]
        return results

    def load_vllm_model(self,
                        model_name_or_path,
                        dtype='auto',
                        trust_remote_code=True,  # False
                        download_dir=None,
                        revision=None,
                        quantization=None,  # "FP8"
                        num_gpus=1,
                        # tokenizer_args
                        use_fast_tokenizer=True,
                        pad_token=None,
                        eos_token=None,
                        gpu_memory_utilization=0.9,
                        enforce_eager=True,
                        max_num_seqs=1,
                        **kwargs
                        ):

        model = LLM(model=model_name_or_path,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    quantization=quantization,
                    gpu_memory_utilization=gpu_memory_utilization,  # to avoid CUDA out of memory
                    enforce_eager=enforce_eager,  # to avoid CUDA out of memory
                    max_num_seqs=max_num_seqs,  # to avoid CUDA out of memory
                    tokenizer_mode="auto" if use_fast_tokenizer else "slow",
                    tensor_parallel_size=num_gpus,
                    distributed_executor_backend="ray",  # for loading multiple ray actors
                    )

        if pad_token:
            model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
        if eos_token:
            model.llm_engine.tokenizer.tokenizer.eos_token = eos_token

        return model

    def set_tokenizer_truncation_side(self, side):
        self.model.llm_engine.tokenizer.tokenizer.truncation_side = side

    def is_initialized(self):
        print(f"==> Initialized {self.model_name} <==")


if __name__ == "__main__":
    list_of_messages = [
        [
            {"role": "user", "content": "What is your name?"},
            # {"role": "assistant", "content": "My name is Loki."}
        ],

        [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
    ]

    # config = {
    #     # "model_name": "google/gemma-2-9b-it",
    #     # "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    #     # "model_name": "meta-llama/Llama-2-7b-chat-hf",
    #     # "model_name": "mistralai/Mistral-Nemo-Instruct-2407",
    #     "model_name": "Qwen/Qwen2-7B-Instruct",
    #     # "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "is_show_prompt": True,
    # }
    # chat_model = HFChatModel(config)

    # config = {
    #     "model_name": "gpt-4o-mini-2024-07-18",
    # }
    # chat_model = GPTModel(config)

    # list_of_completions = chat_model.batch_generate(list_of_messages)
    # print(list_of_completions)

    config = {
        # "model_name": "google/gemma-2-9b-it",
        # "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        # "model_name": "meta-llama/Llama-2-7b-chat-hf",
        # "model_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "model_name": "claude-3-haiku-20240307",
        # "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "is_show_prompt": True,
    }

    chat_model = ClaudeModel(config)

    messages = list_of_messages[0]
    completion = chat_model.generate(messages[0]["content"] * 100)
    print(completion)
