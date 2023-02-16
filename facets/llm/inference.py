from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, AutoConfig
import torch


class GPTJ:
    def __init__(self, multi_gpu=True, precision=torch.float32, cache_dir=None,
                 device_name="cuda:0"):
        checkpoint = "EleutherAI/gpt-j-6B"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.multi_gpu = multi_gpu

        if multi_gpu:
            print(f"Loading GPT-J with multiple GPUs")

            # if sharded_checkpoint is None:
            #     raise ValueError("Sharded checkpoint is required for multi_gpu")

            # from accelerate import init_empty_weights
            # from accelerate import load_checkpoint_and_dispatch
            #
            # config = AutoConfig.from_pretrained(checkpoint)
            #
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_config(config)
            #
            # self.model = load_checkpoint_and_dispatch(
            #     model, checkpoint_path, device_map="auto", no_split_module_classes=["GPTJBlock"]
            # )

            self.model = GPTJForCausalLM.from_pretrained(
                checkpoint, torch_dtype=precision, low_cpu_mem_usage=True, device_map="auto", revision="sharded",
                cache_dir=cache_dir,
            )

        else:
            self.device = device_name if torch.cuda.is_available() else "cpu"
            print(f"Loading GPT-J onto {self.device}.")

            self.model = GPTJForCausalLM.from_pretrained(
                checkpoint, torch_dtype=precision, low_cpu_mem_usage=True, cache_dir=cache_dir
            ).to(self.device)
        print("Done.")

    def generate(self, input_text: str, max_length: int = 2048):
        if self.multi_gpu:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(0)
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=max_length
        )
        return self.tokenizer.batch_decode(tokens)[0]
