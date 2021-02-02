# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import base64
from PIL import Image
import io


def img_from_base64(imagestring):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(imagestring)))
        return img.convert('RGB')
    except ValueError:
        return None



