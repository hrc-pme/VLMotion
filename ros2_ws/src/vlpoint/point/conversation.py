"""VLPoint client conversation helpers.

Derived from ``ros2_ws/src/vlpoint/point/conversation.py`` in VLMotion commit
``57ff1f4``. Only the client functionality used by the White Point GUIs is
kept; controller and model-worker code must run on the inference computer.
"""

import base64
import dataclasses
from enum import Enum, auto
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if messages and isinstance(messages[0][1], tuple):
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            result = self.system + self.sep
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    result += role + ": " + message + self.sep
                else:
                    result += role + ":"
            return result

        if self.sep_style == SeparatorStyle.TWO:
            separators = (self.sep, self.sep2)
            result = self.system + separators[0]
            for index, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    result += role + ": " + message + separators[index % 2]
                else:
                    result += role + ":"
            return result

        if self.sep_style == SeparatorStyle.MPT:
            result = self.system + self.sep
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    result += role + message + self.sep
                else:
                    result += role
            return result

        if self.sep_style == SeparatorStyle.PLAIN:
            separators = (self.sep, self.sep2)
            result = self.system
            for index, (_, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    result += message + separators[index % 2]
            return result

        raise ValueError("Unsupported separator style: %s" % self.sep_style)

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, out_width=640, out_height=480):
        width, height = image.size
        transform = np.eye(3)

        # Preserve the camera frame so returned VLM pixels align with depth.
        if image_process_mode in ("Original", "None"):
            return image.copy(), transform

        aspect_ratio = width / height
        out_aspect_ratio = out_width / out_height
        new_image = image
        if image_process_mode == "Pad":
            background_color = (122, 116, 104)
            if aspect_ratio > out_aspect_ratio:
                new_height = int(width / out_aspect_ratio)
                new_image = Image.new(image.mode, (width, new_height), background_color)
                new_image.paste(image, (0, (new_height - height) // 2))
                transform[1, 2] = (height - new_height) // 2
            elif aspect_ratio < out_aspect_ratio:
                new_width = int(height * out_aspect_ratio)
                new_image = Image.new(image.mode, (new_width, height), background_color)
                new_image.paste(image, ((new_width - width) // 2, 0))
                transform[0, 2] = (width - new_width) // 2
        elif image_process_mode == "Crop":
            if aspect_ratio > out_aspect_ratio:
                new_width = int(height * out_aspect_ratio)
                new_image = image.crop(
                    ((width - new_width) // 2, 0, (width + new_width) // 2, height))
                transform[0, 2] = (width - new_width) // 2
            elif aspect_ratio < out_aspect_ratio:
                new_height = int(width / out_aspect_ratio)
                new_image = image.crop(
                    (0, (height - new_height) // 2, width, (height + new_height) // 2))
                transform[1, 2] = (height - new_height) // 2
        else:
            raise ValueError("Invalid image_process_mode: %s" % image_process_mode)

        width, height = new_image.size
        transform[0, 0] = width / out_width
        transform[1, 1] = height / out_height
        return new_image.resize((out_width, out_height)), transform

    def get_images(self, image_format="PNG"):
        images, encoded_images, transforms = [], [], []
        for role, message in self.messages[self.offset:]:
            if ("user" in role.lower() or "human" in role.lower()) and isinstance(message, tuple):
                _, image, process_mode = message
                image, transform = self.process_image(image, process_mode)
                buffer = BytesIO()
                image.save(buffer, format=image_format)
                images.append(image)
                encoded_images.append(base64.b64encode(buffer.getvalue()).decode())
                transforms.append(transform)
        return images, encoded_images, transforms

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[role, message] for role, message in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [
                [role, message[0] if isinstance(message, tuple) else message]
                for role, message in self.messages
            ],
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
}
