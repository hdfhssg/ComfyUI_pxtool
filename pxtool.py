import random
import json
import os
from .ch_lib import model_action_civitai
import re
import csv
from folder_paths import models_dir
from pathlib import Path
import numpy as np
model_folder_path = Path(models_dir)
root_dir = model_folder_path.parent
root_dir = root_dir / "custom_nodes/ComfyUI_pxtool"


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.loads(f.read())

def format_str(string):
    string = re.sub(r'(?<!\\)\(', r'\\(', string)
    string = re.sub(r'(?<!\\)\)', r'\\)', string)
    return string.strip().replace("_", " ")

def format_tag_str(string):
    # Step 1: 转义所有未被转义的原始括号
    string = re.sub(r'(?<!\\)\(', r'\\(', string)
    string = re.sub(r'(?<!\\)\)', r'\\)', string)
    # Step 2: 处理多层嵌套转义括号（任意层数）
    # 匹配连续两个及以上的转义括号（如 \\(\\、\\\\)\\\\) 等）
    string = re.sub(
        r'((\\[()]){2,})',  # 匹配两个及以上连续的转义括号
        lambda m: m.group(0)[1] * (len(m.group(0)) // 2),  # Fixed lambda function
        string  # 添加目标字符串参数
    )
    # Step 3: 处理包含冒号的括号（恢复原始形式）
    string = re.sub(r'\\\(([^:]*:[^)]*)\\\)', r'(\1)', string)
    string = re.sub(r'\\\)([^:]*:[^)]*)\\\(', r')\1(', string)

    # Step 4: 替换下划线为空格并去除首尾空格
    return string.strip().replace("_", " ")

def remove_duplicate_tags(tags_tuple):
    seen = set()  # 全局基名记录集合

    def get_base(tag):
        # 清理所有非字母数字、下划线、空格和冒号的字符
        trimmed = re.sub(r'[^\w\s:]', '', tag)
        # 去除开头和结尾的非字母数字及下划线
        trimmed = re.sub(r'^[^a-zA-Z0-9_]*', '', trimmed)
        trimmed = re.sub(r'[^a-zA-Z0-9_]*$', '', trimmed)
        # 取冒号前的部分并去除空格
        base = trimmed.split(':')[0].strip()
        return base.lower()  # 基名不区分大小写

    def process_tags(tags_str):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        result = []
        for tag in tags:
            base = get_base(tag)
            if base not in seen:
                seen.add(base)
                result.append(tag)
        # 处理末尾逗号
        joined = ",".join(result)
        if tags_str.endswith(","):
            if not joined.endswith(","):
                joined += ","
        return joined

    return tuple(process_tags(tags_str) for tags_str in tags_tuple)


def add_artist(chose_artists,artist_pref, random_artists, artist, min_weights=1, max_weights=5, 
                            lower_weight=False, higher_weight=False, medium=0.5,random_artist_weight=False,format_artists=True):
    if format_artists:
        artist = format_str(artist)
    if artist_pref:
        artist = f"artist:{artist}"
    if random_artist_weight:
        # 随机权重,值在0.5-1.5之间，正态分布，选在1附近的概率大，只要2位小数,1, 0.25
        num = round(np.random.normal(1, 0.25), 2)
        num = max(0.5, min(1.3, num))
        artist = f"({artist}:{num})"

    if random_artists:
        num = np.random.randint(min_weights, max_weights)
        if lower_weight and higher_weight:
            symbol = np.random.choice([["[", "]"], ["{", "}"]])
            artist = symbol[0] * num + artist + symbol[1] * num
        elif lower_weight:
            if np.random.random() < medium:
                artist = "[" * num + artist + "]" * num
        elif higher_weight:
            if np.random.random() < medium:
                artist = "{" * num + artist + "}" * num

        chose_artists += f"{artist},"

    return chose_artists

def add_Tag(chose_artists,random_weight, random_artists, artist, min_weights=1, max_weights=5, 
                            random_artist_weight=False,format_tags=True):
    # 是否format_str
    if format_tags:
        artist = format_str(artist)
    if random_artist_weight:
        # 随机权重,值在0.5-1.5之间，正态分布，选在1附近的概率大，只要2位小数
        num = round(np.random.normal(1, 0.25), 2)
        num = max(0.5, min(1.3, num))
        artist = f"({artist}:{num})"

    if random_weight:
        num = np.random.randint(min_weights, max_weights+1)
        artist = "(" * num + artist + ")" * num
    if random_artists:
        chose_artists += f"{artist},"

    return chose_artists

def add_position(prompt,chose_artists, position="最后面"):

    if position == "最后面":
        return (
            f"{str(prompt)}{str(chose_artists)}",
            str(chose_artists),
        )
    elif position == "最前面":
        return (
            f"{str(chose_artists)}{str(prompt)}",
            str(chose_artists),
        )


def random_artists_json(
    prompt,
    position,
    random_artists,
    artist_pref,
    lower_weight,
    higher_weight,
    max_artists,
    max_weights,
    min_artists,
    min_weights,
    seed,
    random_artist_weight,
    artist_seed,
    format_artists=True,
):
    random.seed(seed)
    np.random.SeedSequence(artist_seed)
    medium = 0.5
    path_json = os.path.join(root_dir, "artists.json")
    artists: dict = read_json(path_json)
    chose_artists = ""
    for _ in range(random.randint(min_artists, max_artists)):
        while (artist := artists[random.choice(list(artists.keys()))]) in chose_artists:
            pass
        chose_artists = add_artist(chose_artists,artist_pref, random_artists, artist, min_weights, max_weights, lower_weight, higher_weight, medium,random_artist_weight,format_artists)

    #chose_artists = format_str(chose_artists)
    #prompt = format_tag_str(prompt)
    chose_artists = add_position(prompt,chose_artists, position)
    return chose_artists
# 读取B列trigger以及C列count，只读取count>1000的，返回字典
def read_csv(file_path,max_count=1000):  
    artists = {}  
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  # 跳过标题行
        for line in reader:  # 跳过标题行
            _, trigger, count = line[:3]
            count = int(count)
            artists[trigger] = count
            if count < max_count:
                break
        return artists


def random_artists_csv(
    file,
    max_count,
    prompt,
    position,
    random_artists,
    artist_pref,
    lower_weight,
    higher_weight,
    max_artists,
    max_weights,
    min_artists,
    min_weights,
    seed,
    random_artist_weight,
    artist_seed,
    format_artists=True,
):
    random.seed(seed)
    np.random.SeedSequence(artist_seed)
    medium = 0.5
    #artists: dict = read_csv("./custom_nodes/ComfyUI_pxtool/danbooru_artist.csv")
    full_path = os.path.join(root_dir, file)
    artists_dict: dict = read_csv(full_path,max_count)
    artists = list(artists_dict.keys())
    frequencies = list(artists_dict.values())
    chose_artists = ""
    for _ in range(random.randint(min_artists, max_artists)):
        while (artist := random.choices(artists, weights=frequencies)[0]) in chose_artists:
            pass
        chose_artists = add_artist(chose_artists,artist_pref, random_artists, artist, min_weights, max_weights, lower_weight, higher_weight, medium,random_artist_weight,format_artists)
    #chose_artists = format_str(chose_artists)
    #prompt = format_tag_str(prompt)
    chose_artists = add_position(prompt,chose_artists, position)
    return chose_artists



# 这是comfyui的python插件
# C站助手，输出是root_dir即扫码文件夹
class CivitaiHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                 "directory": ("STRING", {"default": "./models"}),
                "checkpoints": ("BOOLEAN", {"default": True}),
                "loras": ("BOOLEAN", {"default": True}),
                "unet": ("BOOLEAN", {"default": True}),
                "diffusion_models": ("BOOLEAN", {"default": True}),
                "max_size_preview": ("BOOLEAN", {"default": True}),
                "skip_nsfw_preview": ("BOOLEAN", {"default": False}),

            }
        }
    
    FUNCTION = "civitai_helper"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI-pxtool"

    def civitai_helper(self, directory, checkpoints, loras, unet, diffusion_models, max_size_preview, skip_nsfw_preview):
        scan_model_types = ["ckp", "lora"]
        folders = {}
        if checkpoints:
            folders["ckp"] = os.path.join(directory, "checkpoints")
        if loras:
            folders["lora"] = os.path.join(directory, "loras")
        if unet:
            folders["unet"] = os.path.join(directory, "unet")
        if diffusion_models:
            folders["diffusion_models"] = os.path.join(directory, "diffusion_models")
        
        model_action_civitai.scan_model(scan_model_types, max_size_preview, skip_nsfw_preview, folders)
        return (str("扫描完成"),)

# noob随机画师串生成器
class RandomArtists:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt": ("STRING", {"default": "1girl,"}),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "position": (["最前面", "最后面"],),
                "random_artists": ("BOOLEAN", {"default": True}),
                "random_artist_weight": ("BOOLEAN", {"default": False}),
                "artist_weight_seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "year": (["None", "year 2022", "year 2023", "year 2024"],),
                "artist_pref": ("BOOLEAN", {"default": False}),
                "lower_weight": ("BOOLEAN", {"default": False}),
                "higher_weight": ("BOOLEAN", {"default": False}),
                "max_artists": ("INT", {"default": 5, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_artists": ("INT", {"default": 2,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "max_weights": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_weights": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "format_artists": ("BOOLEAN", {"default": True}),
            }
        }
    
    FUNCTION = "random_artists"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def random_artists(self, prompt, position, random_artists, year, artist_pref, lower_weight, higher_weight, 
                       max_artists, max_weights, min_artists, min_weights, seed, random_artist_weight, artist_weight_seed,format_artists):
        if year != "None":
            prompt = year + "," + prompt
        tag = random_artists_json(prompt, position, random_artists, artist_pref, lower_weight, higher_weight, 
                                  max_artists, max_weights, min_artists, min_weights, seed, random_artist_weight, artist_weight_seed,format_artists)
        return remove_duplicate_tags(tag)
# noob随机画师串生成器，高级
class RandomArtistsAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt": ("STRING", {"default": "1girl,"}),
                "file": (["danbooru_artist.csv", "e621_artist.csv"],),
                "max_count": ("INT", {"default": 1000, "min": 0, "max": 0xffffffffffffffff, "step": 100}),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "position": (["最前面", "最后面"],),
                "random_artists": ("BOOLEAN", {"default": True}),
                "random_artist_weight": ("BOOLEAN", {"default": False}),
                "artist_weight_seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "year": (["None", "year 2022", "year 2023", "year 2024"],),
                "artist_pref": ("BOOLEAN", {"default": False}),
                "lower_weight": ("BOOLEAN", {"default": False}),
                "higher_weight": ("BOOLEAN", {"default": False}),
                "max_artists": ("INT", {"default": 5, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_artists": ("INT", {"default": 2,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "max_weights": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_weights": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "format_artists": ("BOOLEAN", {"default": True}),
            }
        }
    
    FUNCTION = "random_artists_advanced"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"
    def random_artists_advanced(self, prompt, file, max_count, seed, position, random_artists, year, 
                                artist_pref, lower_weight, higher_weight, max_artists, max_weights, 
                                min_artists, min_weights,random_artist_weight,artist_weight_seed,format_artists):
        if year != "None":
            prompt = year + "," + prompt
        tag = random_artists_csv(file, max_count, prompt, position, random_artists, artist_pref, 
                                 lower_weight, higher_weight, max_artists, max_weights, min_artists, 
                                 min_weights, seed,random_artist_weight,artist_weight_seed,format_artists)
        return remove_duplicate_tags(tag)

def read_character_csv(file_path, max_count=1000):
    artists = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  # 跳过标题行
        for line in reader:  # 跳过标题行
            character, _, trigger, core_tags, count = line[:5]
            count = int(count)
            artists[character] = {"triggers": trigger, "count": count, "core_tags": core_tags}
            if count <= max_count:
                break
        return artists


# 角色串tag生成器
class DanbooruCharacterTag:
    @classmethod
    def INPUT_TYPES(s):
        artists1: dict = read_character_csv(os.path.join(root_dir, "danbooru_character.csv"))
        Character_list = list(artists1.keys())
        return {
            "required":{
                "character": (Character_list,),
                "return_type": (["tag + trigger", "tag", "trigger"],),
            }
        }
    
    FUNCTION = "character_tag"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def character_tag(self, character, return_type):
        artists1: dict = read_character_csv(os.path.join(root_dir, "danbooru_character.csv"))
        if return_type == "tag":
            return (artists1[character]["core_tags"]+",",)
        elif return_type == "trigger":
            return (artists1[character]["triggers"]+",",)
        else:
            return (artists1[character]["core_tags"]+","+artists1[character]["triggers"]+",",)

def read_e621_character_csv(file_path, max_count=1000):
    artists = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  # 跳过标题行
        for line in reader:  # 跳过标题行
            character, _, trigger, count = line[:4]
            count = int(count)
            artists[character] = {"triggers": trigger, "count": count}
            if count <= max_count:
                break
        return artists

# 角色串tag生成器
class E621CharacterTag:
    @classmethod
    def INPUT_TYPES(s):
        artists1: dict = read_e621_character_csv(os.path.join(root_dir, "e621_character.csv"))
        Character_list = list(artists1.keys())
        return {
            "required":{
                "character": (Character_list,),
            }
        }
    
    FUNCTION = "character_tag"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def character_tag(self, character):
        artists1: dict = read_e621_character_csv(os.path.join(root_dir, "e621_character.csv"))
        return (artists1[character]["triggers"]+",",)


def read_tag_csv(file_path,max_count=1000):  
    artists = {}  
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  # 跳过标题行
        next(reader) # 跳过1girl
        for line in reader:  # 跳过标题行
            trigger, count = line
            count = int(count)
            artists[trigger] = count
            if count < max_count:
                break
        return artists

def random_tag_csv(
    prompt,
    file,
    max_count,
    position,
    random_artists,
    random_weight,
    max_artists,
    max_weights,
    min_artists,
    min_weights,
    seed,
    random_artist_weight,
    tag_seed,
    format_tags=True,
):
    random.seed(seed)
    np.random.SeedSequence(tag_seed)
    full_path = os.path.join(root_dir, file)
    artists_dict: dict = read_tag_csv(full_path,max_count)
    artists = list(artists_dict.keys())
    frequencies = list(artists_dict.values())
    chose_artists = ""
    keywords = ["1girl", "2girls", "3girls", "4girls", "5girls","6+girls", "sisters", "1other", "multiple_girls","1boy", "2boys", "3boys", "4boys", "5boys","6+boys", "multiple_boys","solo", "duo", "trio", "group"]

    for _ in range(random.randint(min_artists, max_artists)):
        while (artist := random.choices(artists, weights=frequencies)[0]) in ([a for a in chose_artists.split(",") if a] + keywords):
            pass
        chose_artists = add_Tag(chose_artists,random_weight, random_artists, artist, min_weights, max_weights,random_artist_weight,format_tags)
    prompt = format_tag_str(prompt)
    chose_artists = add_position(prompt,chose_artists, position)
    return chose_artists


# 随机提示词生成器
class RandomTag:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt": ("STRING", {"default": "1girl,"}),
                "file": (["sfw_tags.csv","full_tags.csv", "nsfw_tags.csv"],),
                "max_count": ("INT", {"default": 50000, "min": 0, "max": 0xffffffffffffffff, "step": 1000}),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "girl_tag" : (["None","1girl", "2girls", "3girls", "4girls", "5girls","6+girls", "sisters", "1other", "multiple_girls"],),
                "boy_tag": ([ "None","1boy", "2boys", "3boys", "4boys", "5boys","6+boys", "multiple_boys"],),
                "multiple_tag": (["None","solo", "duo", "trio", "group"],),
                "prefix": ("BOOLEAN", {"default": True}),
                "position": (["最后面", "最前面"],),
                "random_tag": ("BOOLEAN", {"default": True}),
                "random_Tag_weight": ("BOOLEAN", {"default": False}),
                "random_weight": ("BOOLEAN", {"default": False}),
                "tag_weight_seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                    ),
                "year": (["None", "year 2022", "year 2023", "year 2024"],),
                "max_tag": ("INT", {"default": 30, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_tag": ("INT", {"default": 10,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "max_weights": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "min_weights": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "format_tags": ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "random_tag"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def random_tag(self, prompt, file, max_count, seed, position, random_tag,year, random_weight, 
                    max_tag, max_weights, min_tag, min_weights, prefix, girl_tag,boy_tag,multiple_tag,random_Tag_weight,tag_weight_seed,format_tags):
        if multiple_tag != "None":
            prompt = prompt.replace("solo,", "")
            prompt = multiple_tag + "," + prompt
        if boy_tag != "None":
            prompt = prompt.replace("1boy,", "")
            prompt = boy_tag + "," + prompt
        if girl_tag != "None":
            prompt = prompt.replace("1girl,", "")
            prompt = girl_tag + "," + prompt
        if year != "None":
            prompt = year + "," + prompt
        if prefix:
            prompt = "masterpiece, best quality, newest, absurdres, highres, safe," + prompt
        tag =random_tag_csv(prompt,file,max_count,position, random_tag,random_weight,max_tag,max_weights,
                            min_tag,min_weights,seed,random_Tag_weight,tag_weight_seed,format_tags)
        return remove_duplicate_tags(tag)

# 质量标签添加器，masterpiece > best quality > high quality / good quality > normal quality > low quality / bad quality > worst quality
class QualityTag:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt": ("STRING", {"default": "1girl,"}),
                "quality": (["None","masterpiece", "best quality", "high quality", "good quality", "normal quality", "low quality", "bad quality", "worst quality"],{"default":"masterpiece"}),
                "safe": (["None","general", "sensitive", "nsfw", "explicit"],),
                "aesthetic": (["None","very awa", "worst aesthetic"],),
                "time": (["None","newest", "recent", "mid", "early", "old"],),
                "position": (["最前面", "最后面"],),
            }
        }
    
    FUNCTION = "quality_tag"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def quality_tag(self, prompt,quality, safe, aesthetic, time, position):
        quality_tags = ""
        if aesthetic != "None":
            quality_tags += f"{aesthetic},"
        if safe != "None":
            quality_tags += f"{safe},"
        if time != "None":
            quality_tags +=f"{time},"
        if quality != "None":
            quality_tags += f"{quality},"
        if position == "最后面":
            tags = (f"{format_str(str(prompt))}{quality_tags}"),format_str(str(quality_tags))
            return remove_duplicate_tags(tags)
        elif position == "最前面":
            tags = (f"{quality_tags}{format_str(str(prompt))}"),format_str(str(quality_tags))
            return remove_duplicate_tags(tags)

def read_txt(file_path):
    with open(file_path, "r") as f:
        data = f.read()
        data = data.split("\n")
        return data

# 负面提示词生成器
class NegativeTag:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt": ("STRING", {"default": 
                                      "nsfw, worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro,"}),
            "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
            "girl_tag" : (["None","1girl", "2girls", "3girls", "4girls", "5girls","6+girls", "sisters", "1other", "multiple_girls"],),
            "boy_tag": ([ "None","1boy", "2boys", "3boys", "4boys", "5boys","6+boys", "multiple_boys"],),
            "multiple_tag": (["None","solo", "duo", "trio", "group"],),
            "random_tag": ("BOOLEAN", {"default": True}),
            "random_Tag_weight": ("BOOLEAN", {"default": False}),
            "random_weight": ("BOOLEAN", {"default": False}),
            "tag_weight_seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
            "old": ("BOOLEAN", {"default": True}),
            "max_tag": ("INT", {"default": 30, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "min_tag": ("INT", {"default": 10,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "max_weights": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "min_weights": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "format_tags": ("BOOLEAN", {"default": True}),
            }
        }
    
    FUNCTION = "negative_tag"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "ComfyUI-pxtool"

    def negative_tag(self, prompt, seed,random_weight, random_tag, old, max_tag, max_weights, min_tag, 
                     min_weights,random_Tag_weight,girl_tag,boy_tag,multiple_tag,tag_weight_seed,format_tags):
        random.seed(seed)
        np.random.SeedSequence(tag_weight_seed)
        if multiple_tag != "None":
            prompt = prompt.replace("solo,", "")
            multiple_tag = add_Tag("",random_weight, random_tag, multiple_tag, min_weights, max_weights,random_Tag_weight,format_tags)
            prompt = multiple_tag + "," + prompt
        if boy_tag != "None":
            prompt = prompt.replace("1boy,", "")
            prompt = boy_tag + "," + prompt
        if girl_tag != "None":
            prompt = prompt.replace("1girl,", "")
            prompt = girl_tag + "," + prompt
        artists_dict: dict = read_txt(os.path.join(root_dir, "NegativeTag.txt"))
        chose_artists = ""
        for _ in range(random.randint(min_tag, max_tag)):
            while (artist := random.choice(list(artists_dict))) in chose_artists:
                pass
            chose_artists = add_Tag(chose_artists,random_weight, random_tag, artist, min_weights, max_weights,random_Tag_weight,format_tags)
        if old:
            chose_artists += "old,"
        tags = (f"{str(prompt)}{str(chose_artists)}"),str(chose_artists)
        return remove_duplicate_tags(tags)

class PX_Seed:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("seed",)
    FUNCTION = "seedint"
    CATEGORY = "ComfyUI-pxtool"

    def seedint(self, seed):
        return (seed,)

NODE_CLASS_MAPPINGS = {
    "CivitaiHelper": CivitaiHelper,
    "RandomArtists": RandomArtists,
    "DanbooruCharacterTag": DanbooruCharacterTag,
    "E621CharacterTag": E621CharacterTag,
    "RandomTag": RandomTag,
    "RandomArtistsAdvanced": RandomArtistsAdvanced,
    "QualityTag": QualityTag,
    "NegativeTag": NegativeTag,
    "PX_Seed": PX_Seed,
}