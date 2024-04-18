# Demo
from vlmeval.config import supported_VLM

model = supported_VLM['DifyVision']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
