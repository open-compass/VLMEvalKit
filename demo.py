# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['mPLUG-Owl3']()

# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.