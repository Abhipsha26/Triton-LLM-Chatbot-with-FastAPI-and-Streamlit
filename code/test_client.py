import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

# Load tokenizer for Phi model (adjust if you're using a different one)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Step 1: Take a natural language question
question = "Once upon a time?"

# Step 2: Tokenize the question into input IDs
tokens = tokenizer.encode(question, return_tensors=None)
input_ids = np.array(tokens, dtype=np.int64)

# Step 3: Connect to Triton server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Step 4: Format input for Triton
inputs = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
inputs.set_data_from_numpy(input_ids)

# Step 5: Request output
outputs = httpclient.InferRequestedOutput("output_text")

# Step 6: Run inference
response = client.infer(model_name="phi4", inputs=[inputs], outputs=[outputs])

# Step 7: Decode output
output = response.as_numpy("output_text")
if isinstance(output[0], bytes):
    print("Answer:", output[0].decode("utf-8"))
else:
    print("Answer:", output[0])

