import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        print(">> Initializing model...")

        model_path = "/models_cache/phi-1_5/models--microsoft--phi-1_5/snapshots/675aa382d814580b22651a30acb1a585d7c25963"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.eval()

        # 🔧 Fix: Set pad_token if missing
        if self.tokenizer.pad_token is None:
            print(">> Setting pad_token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure model uses correct pad_token_id
        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        print(">> Model loaded successfully.")

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_text")
            input_bytes = input_tensor.as_numpy()  # shape: (1,)
            input_texts = [x.decode("utf-8").strip() for x in input_bytes]

            print(">> Input text:", input_texts[0])

            # Tokenize input with padding settings
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=100
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Remove prompt from output (token-based, robust)
            output_texts = []
            for i, input_text in enumerate(input_texts):
                input_ids = inputs["input_ids"][i]
                output_ids = outputs[i]
                # Remove the prompt tokens from the generated sequence
                generated_ids = output_ids[len(input_ids):]
                gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                output_texts.append(gen_text)

                print(">> Output text:", gen_text)

            output_array = np.array([x.encode("utf-8") for x in output_texts], dtype=np.object_)
            response_tensor = pb_utils.Tensor("output_text", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[response_tensor]))

        return responses
