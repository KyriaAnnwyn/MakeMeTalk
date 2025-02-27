import instructor
from pydantic import BaseModel, Field
from enum import Enum
from openai import OpenAI
import base64
import json
from PIL import Image
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from typing import Union
import imghdr

load_dotenv(override=True)

NUM_TRIES = 2
PROXY: str = os.getenv("OPENAI_PROXY_SOCKS")
OPEN_AI_API_KEYS = os.getenv("OPENAI_API_KEY")
OPENAI_RESPONSE_TIMEOUT_SECONDS = 25

_http_client = None
_model_instance = None
_client = None

if PROXY:
    _http_client = httpx.Client(
        proxies=PROXY, timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )
else:
    _http_client = httpx.Client(
        timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )
   
_model_instance = OpenAI(
    api_key=OPEN_AI_API_KEYS,
    http_client=_http_client
)
_client = instructor.from_openai(_model_instance)

class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return [c.value for c in cls]

class SkinColour(ExtendedEnum):
    PALE  = "pale"
    WHITE = "white"
    BEIGE = "beige"
    OLIVE = "olive"
    TAN   = "tan"
    DARK_BROWN = "dark brown"
    UNKNOWN = "unknown"

def minusTone(skin_tone: str) -> str: 
    skin_tones = SkinColour.values()
    for idx, st in enumerate(skin_tones[:-1]):
        if skin_tone == st:
            return skin_tones[max(0, idx - 1)]

    return ""

class HairCut(Enum):
    BALD = "no hair, bald"
    SHORT = "short hairstyle"
    MEDIUMSTRAIGHT = "medium straight hair"
    MEDIUMCURLY = "medium curly hair"
    MEDIUMWAVY = "medium wavy hair"
    LONGSTRAIGHT = "long straight hair"
    LONGCURLY = "long curly hair"
    LONGWAVY = "long wavy hair"
    PONYTAIL = "ponytail"
    UNKNOWN = "unknown"


class HairColor(Enum):
    BALD = ""
    BLACK = "black hair"
    BROWN = "brown hair"
    BLOND = "blond hair"
    RED = "red hair"
    GREY = "grey hair"
    WHITE = "white hair"
    UNKNOWN = "unknown"

class BeardType(Enum):
    NOBEARD = ""
    SHORTBARD = "short beard"
    LONGBARD = "long beard"
    UNKNOWN = "unknown"

class FaceComplection(Enum):
    ROUND = "round face"
    FULL = "full face"
    THIN = "thin face"
    NORMAL = ""
    UNKNOWN = "unknown"

class EyeColour(Enum):
    BLUE = "blue eyes"
    BROWN = "brown eyes"
    GREEN = "green eyes"
    GREY = "grey eyes"
    UNKNOWN = "unknown"

class Gender(Enum):
    MALE = "man"
    FEMALE = "woman"
    UNKNOWN = "unknown"  

class Race(Enum):
    INDIAN = "indian"
    MONGOLOID = "mongoloid"
    MIXED = "mixed"
    #AFRICAN = "african"
    CAUSASIAN = "caucasian"
    LATINO = "latino"
    NEGROID = "negroid"
    
class BodyHeight(Enum):
    SMALL = "small"
    TALL = "tall"
    NORMAL = ""
    UNKNOWN = "unknown"

class ViewType(Enum):
    PORTRAIT = "portrait view"
    HADNSHOULDERS = "head and shoulders view"
    HALFBODY = "upper body portrait view"
    FULLBODY = "full body view"
    UNKNOWN = "unknown"

class BodyBuild(Enum):
    FAT = "fat body"
    LARGE = "large body"
    PLUMP = "plump body"
    AVERAGE= "" #"average body build"
    STRONG = "strong, muscular, wide body build"
    SLIM = "slim"
    UNKNOWN = "unknown"

    #slim
    #fit
    #athletic
    #average
    #muscular
    #chubby
    #large
    #full-figured
    #medium
    #curvy
    

class Face(BaseModel):
    gender: Union[Gender, None] = Field(
        description="Correctly assign gender of person on the image", default=None
    )
    skin_colour: Union[SkinColour, None] = Field(
        description="Correctly assign colour of skin of person on the image", default=None
    )
    haircut: Union[HairCut, str] = Field(
        description="Correctly assign haircut type of person on the image", default=''
    )
    haircolor: Union[HairColor, None] = Field(
        description="Correctly assign hair colour of person on the image", default=None
    )
    beard: Union[BeardType, None] = Field(
        description="Correctly assign beard type of person on the image", default=None
    )
    face_complection: Union[FaceComplection, None] = Field(
        description="Correctly assign face build of person on the image", default=None
    )
    eye_colour: Union[EyeColour, None] = Field(
        description="Correctly assign eye colour of person on the image", default=None
    )
    race: Union[Race, None] = Field(
        description="Correctly assign race of person on the image", default=None
    )

class Body(BaseModel):
    height: Union[BodyHeight, None] = Field(
        description="Correctly assign body height of person on the image", default=None
    )
    view: ViewType = Field(
        description="Correctly assign view of person on the image"
    )
    body_build: Union[BodyBuild, None] = Field(
        description="Correctly assign body build of person on the image", default=None
    ) 
    
    
class PersonApperance(BaseModel):
    face: Face = Field(
        description="Look carefully at the main person on the image and describe its face"
    )
    body: Body = Field(
        description="Look carefully at the main person on the image and describe its body"
    )

def get_description(image_path):
    global _client

    fld = os.path.dirname(image_path)
    if os.path.getsize(image_path) > 20000000:
        print(f"File size: {os.path.getsize(image_path)} - scaling")
        img = Image.open(image_path)
        w,h = img.size
        img = img.resize((int(w/2), int(h/2)))
        image_path = fld + "/tmp.png"
        img.save(image_path)
    img_format = imghdr.what(image_path)
    if img_format not in ['png', 'jpeg', 'gif', 'webp']:
        img = Image.open(image_path)
        image_path = fld + "/tmp.png"
        img.save(image_path)

    encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    
    tr = 0
    success = False
    while tr < NUM_TRIES and not success:
        try:
            resulting_json = _client.chat.completions.create(
                #model="gpt-4o-mini-2024-07-18", 
                model="gpt-4o-2024-05-13", 
                response_model=PersonApperance,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "You are portrait painter. Given the image of the person your goal is to carefully describe its appearance. Make description as exact as you can."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                        ]
                    }
                ]
            )
            success = True
        except Exception as e:
            print(f"OpenAI appearance request failed: try {tr}, reason: {e}")
            tr += 1

    if success:
        return json.loads(resulting_json.model_dump_json())
    else:
        return None

def get_description_no_scheme(image_path):
    global _client

    fld = os.path.dirname(image_path)
    if os.path.getsize(image_path) > 20000000:
        print(f"File size: {os.path.getsize(image_path)} - scaling")
        img = Image.open(image_path)
        w,h = img.size
        img = img.resize((int(w/2), int(h/2)))
        image_path = fld + "/tmp.png"
        img.save(image_path)
    img_format = imghdr.what(image_path)
    if img_format not in ['png', 'jpeg', 'gif', 'webp']:
        img = Image.open(image_path)
        image_path = fld + "/tmp.png"
        img.save(image_path)

    encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

    if PROXY:
        http_client = httpx.Client(
            proxies=PROXY, timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
        )
    else:
        http_client = httpx.Client(
            timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
        )

    client_2 = OpenAI(
        api_key=OPEN_AI_API_KEYS,
        http_client=http_client
    )

    
    tr = 0
    success = False
    while tr < NUM_TRIES and not success:
        try:
            resulting_json = client_2.chat.completions.create(
                #model="gpt-4o-mini-2024-07-18", 
                model="gpt-4o-2024-05-13", 
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "You are portrait painter. Given the image of the person your goal is to carefully describe its appearance. Make description as exact as you can. What color is the skin of the person, why you decided so?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                        ]
                    }
                ]
            )
            success = True
        except Exception as e:
            print(f"OpenAI appearance request failed: try {tr}, reason: {e}")
            tr += 1

    if success:
        print(resulting_json.choices[0].message.content)
        return json.loads(resulting_json.model_dump_json())
    else:
        return None

def appearance_mean(dt: list[dict]) -> dict:
    app_dict_out = {}
    app_dict_out['face'] = {}
    app_dict_out['body'] = {}

    for idx, d in enumerate(dt):
        for key, value in d['face'].items():
            if value is None or value == 'unknown':
                dt[idx]['face'][key] = ''
        for key, value in d['body'].items():
            if value is None or value == 'unknown':
                dt[idx]['body'][key] = ''

    if len(dt) > 0:
        for key in dt[0]['face'].keys():
            key_counts = Counter(d['face'][key] for d in dt)
            app_dict_out['face'][key] = key_counts.most_common(1)[0][0]
        for key in dt[0]['body'].keys():
            if key == 'view':
                continue
            try:
                key_counts = Counter(d['body'][key] for d in dt if d['body']['view'] != 'portrait view' and d['body'][
                    'view'] != 'head and shoulders view')
                app_dict_out['body'][key] = key_counts.most_common(1)[0][0]
            except:
                app_dict_out['body'][key] = ""

        #app_dict_out['face']['skin_colour'] = ""
        return app_dict_out
    else:
        return dict()

def prepare_appearance_4gen(ap_emb: dict) -> dict:
    if ap_emb['face']['skin_colour'] == 'dark brown':
        ap_emb['face']['skin_colour'] = 'black'
    if ap_emb['body']['body_build'] == 'plump body':
        ap_emb['body']['body_build'] = 'fat body'
    if ap_emb['face']['skin_colour']:
        ap_emb['face']['skin_colour'] += " skin"

    return ap_emb

def get_prompt_avatar(app_dict: dict) -> str:
    return "".join(
        [value + ", " for (key, value) in app_dict['face'].items() if value != '' and key != 'gender' and key != "race"]) + "".join(
        [value + ", " for (key, value) in app_dict['body'].items() if value != '' and key != "body_build"])


def get_prompt_post(app_dict: dict) -> str:
    return " dressed suitable for situation, high quality clothes, " + \
        ", ".join(value.strip() for key, value in app_dict['face'].items() if value.strip() != '' and key not in ["gender", "race"]) + ", " + \
        ", ".join(value.strip() for value in app_dict['body'].values() if value.strip() != '')

def get_appearance(id_image):
    appearance_embeddings = []
    image_path_list = id_image
    if isinstance(id_image, str):
        image_basename_list = os.listdir(id_image)
        image_path_list = sorted([os.path.join(id_image, basename) for basename in image_basename_list 
                                        if ".jpeg" in basename.lower() 
                                        or ".jpg" in basename.lower() 
                                        or ".png" in basename.lower() 
                                        or ".webp" in basename.lower()
                                        or ".avif" in basename.lower()])
    for idimg in image_path_list:
        dt = get_description(idimg)
        appearance_embeddings.append(dt)
    appearance_emb = appearance_mean(appearance_embeddings)
    appearance_emb = prepare_appearance_4gen(appearance_emb)
    appearance_emb_text = get_prompt_post(appearance_emb)

    return appearance_emb_text


if __name__ == "__main__":
    image_path = "/home/docet/Pictures/PromowomanSources/1638098493_26-koshka-top-p-s-kotom-i-devushkoi-29.jpg"
    image_path = "/home/docet/Projects/openai_vision/AvatarImgs/Avatar_62/08726ec2-dc34-48c9-9a7d-225ff83077b4.png"
    image_path = "/home/s.korobkova/TestData/SkinColor/PersonD/2cb36c50-cf2e-4653-9de8-9e44db6d22cf.jpg"
    #result = get_description_no_scheme(image_path=image_path)
    result = get_description(image_path=image_path)

    print(result)
    #print(json.loads(result.model_dump_json()))
    #dt = json.loads(result.model_dump_json())
    #dt['face']['skin_colour'] += " skin"

    #res_prompt_add = "".join([value + ", "  for (key, value) in dt['face'].items() if value != '']) + "".join([value + ", "  for (key, value) in dt['body'].items() if value != '']) 
    #print(res_prompt_add)