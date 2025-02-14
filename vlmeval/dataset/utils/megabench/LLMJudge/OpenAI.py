import base64
import requests
import logging
from .base_model import BaseModel
from PIL import Image, ImageFile
from io import BytesIO
from mimetypes import guess_type
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OpenAI(BaseModel):

    def create_media_content(self, file_path, is_demo=False):
        if self._is_video_file(file_path):
            # Handle video processing with the frame subsampling logic
            video_content = [{"type": "text", "text": self.prompts["video_start"]}]
            video_content.extend(self.process_video(file_path, is_demo))
            video_content.append({"type": "text", "text": self.prompts["video_end"]})
            return video_content
        else:
            # Handle image processing otherwise
            return [self.create_image_content(file_path)]

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    def encode_image(self, image_path, max_side=None):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)
        if not max_side and self.max_side:
            max_side = self.max_side

        if self.resize and max(image.size) > self.max_side:
            image = self._resize_image(image)
        encoded_image = self._encode_image(image, image_format)

        return encoded_image, mime_type

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    def prepare_context(self):
        global_description = self.query_data.get("global_description", "")
        global_images = self.query_data.get("global_images", [])
        content = self._process_text_and_media(global_description, global_images)

        example_info = self.query_data["example_info"]
        example_content = self.prepare_example_content(example_info)
        content.extend(example_content)
        return content

    def prepare_example_content(self, example_info):
        example_text = example_info["example_text"]
        example_media_paths = example_info["image_paths"]
        return self._process_text_and_media(example_text, example_media_paths, is_example=True)

    def prepare_query_content(self, query_info):
        query_text = query_info.get("query_text", "")
        image_paths = query_info.get("image_paths", [])
        query_content = self._process_text_and_media(query_text, image_paths)
        return query_content

    @property
    def url(self) -> str:
        """The server URL."""
        return "https://api.openai.com/v1/chat/completions"

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self._set_sampling_config(0)
        context = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
            tqdm(
                query_data["queries"],
                desc=f"{task_name} - Processing queries",
                unit="query",
                position=position,
            )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)

            query_content = self.prepare_query_content(query_info)

            if not exceed_image_quota:
                messages = self.prepare_system_message()
                messages.append({"role": "user", "content": context + query_content})
                query_payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                }

                response_data = None
                while response_data is None:
                    response = requests.post(
                        self.url,
                        headers=headers,
                        json=query_payload,
                    )
                    try:
                        response_ = response.json()
                    except requests.exceptions.JSONDecodeError as e:
                        logging.info(f"Can't parse output: {e}, retry...")
                    if "error" in response_:
                        error_info = response_["error"]
                        logging.info(
                            f"Got error with type: {error_info['type']}. Message: {error_info['message']}"
                        )
                        logging.info(f"Retry...")
                    else:
                        response_data = response_
                        break

                total_tokens = response_data.get("usage", {}).get("total_tokens", "N/A")

                # Extracting the 'content' field from the response
                if response_data and "choices" in response_data:
                    choices = response_data["choices"]
                    if choices and "message" in choices[0]:
                        message_content = choices[0]["message"]["content"]
                        if self.print_response:
                            logging.info(
                                f"response: {message_content}; tokens:{total_tokens}"
                            )
                else:
                    message_content = ""
            else:
                message_content = (
                    "Exceed the specified max number of images, skip..."
                )

            # save the correct answer here as well for later scoring
            query_response.append(
                {
                    "response": message_content,
                    "correct_answer": query_info["query_answer"],
                }
            )

        return query_response
