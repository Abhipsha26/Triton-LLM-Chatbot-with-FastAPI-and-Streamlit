import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model initialized!")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids = input_tensor.as_numpy()[0]  # np array of int32

            # Decode token IDs back to string
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            out_tensor = pb_utils.Tensor(
                "output_ids",
                np.array([decoded_text.encode('utf-8')], dtype=np.object_)
            )
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up model...")

