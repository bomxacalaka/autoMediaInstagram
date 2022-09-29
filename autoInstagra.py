from instagrapi import Client
from instagrapi.types import StoryMention, StoryMedia, StoryLink, StoryHashtag
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageDraw
from PIL import ImageFont
from fontTools.ttLib import TTFont
from instagrapi import Client
from instagrapi.types import StoryMention, StoryMedia, StoryLink, StoryHashtag
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import random
import time

# get your bloom token at https://huggingface.co/bigscience/bloom/tree/main
bloom_token = ""
instagram_user_name = ""
instagram_password = ""

inference = InferenceApi("bigscience/bloom",token=bloom_token) # logs in bloom

# parameters to generate the text
def infer(prompt,
          max_length = 200,
          top_k = 0,
          num_beams = 0,
          no_repeat_ngram_size = 2,
          top_p = 0.9,
          seed=random.randint(0, 1000),
          temperature=0.7,
          greedy_decoding = False,
          return_full_text = False):
    

    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams,
        "return_full_text":return_full_text
    }
    
    s = time.time()
    response = inference(prompt, params=params)
    proc_time = time.time()-s
    return response

# loads random words from local file
with open("words.txt", "r") as hi:
    reply = hi.read()

# gets 2 random words from the file
prompt = random.choice(reply.split(" ")) + " " + random.choice(reply.split(" "))

# give those 2 words to bloom and get a text response
resp = infer(prompt)
resp = resp[0]['generated_text']

# splits the text into lines to fit the picture better
resp2 = ''
o = 0
for i in (resp.split(" ")):
    if o > 4:
        resp2 += "\n"
        o = 0
    o += 1
    resp2 += i + " "
resp = resp2

# loads fonts to add to picture
font = ImageFont.truetype(font='font.ttf', size=50)

# loads random picture from picsum
url = "https://picsum.photos/1000/1000/?blur"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# adds text to picture
I1 = ImageDraw.Draw(img)
I1.text((20, 100), resp[:100], font=font, fill=(0, 0, 0))
I1.text((20, 95), resp[:100], font=font, fill=(255, 255, 255))

img.save('image.jpg')

# logs into instagram using cookies if they are available
cl = Client()
try:
    cl.load_settings("cookies.json")
except:
    pass
cl.login(instagram_user_name, instagram_password)
cl.dump_settings("cookies.json")

# uploads the picture to instagram
media = cl.photo_upload(
"image.jpg",
resp,
extra_data={
"custom_accessibility_caption": "ai generated text",
"like_and_view_counts_disabled": 0,
"disable_comments": 0,
})

# outputs date and time
import datetime
now = datetime.datetime.now()
print(str(now))