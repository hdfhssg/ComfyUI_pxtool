import os
import json
import random
import base64
import io
from PIL import Image
import requests
from pathlib import Path
import shutil
import textwrap
from collections import namedtuple
from aiohttp import web
import folder_paths
from folder_paths import models_dir
model_folder_path = Path(models_dir)
root_dir = model_folder_path.parent
root_dir = root_dir / "custom_nodes/ComfyUI_pxtool/JSON"
import node_helpers
import re
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
def get_config2(path, open_mode='r'):
    file = os.path.join(root_dir, path)
    try:
        with open(file, open_mode, encoding='utf-8') as f:
            as_dict = json.load(f) 
    except FileNotFoundError as e:
        print(f"{e}\n{file} not found, check if it exists or if you have moved it.")
    return as_dict

def copy_json_file(source_path: str, destination_path: str, overwrite: bool = False):
        """
        複製JSON檔案並確認其格式正確
        Parameters:
        source_path (str): 來源JSON檔案的路徑
        destination_path (str): 目標位置的路徑
        overwrite (bool): 若為True則覆寫已存在的檔案，預設為False
        Returns:
        bool: 複製成功返回True，失敗返回False
        """
        try:
            # 確認來源檔案存在
            file = Path(os.path.join(root_dir, source_path))
            if not file.exists():
                print(f"错误：来源 '{source_path}' 不存在")
                return False
            
            # 檢查目標檔案是否已存在
            dest = Path(os.path.join(root_dir, destination_path))
            if dest.exists() and not overwrite:
                return False
            
            # 讀取並驗證JSON格式
            #with open(file, 'r', encoding='utf-8') as file:
            #    json.load(file)  # 確認JSON格式正確
            
            # 建立目標資料夾（如果不存在）
            #dest.parent.mkdir(parents=True, exist_ok=True)
            
            # 複製檔案
            shutil.copy2(os.path.join(root_dir, source_path), os.path.join(root_dir, destination_path))
            print(f"成功：档案已复制 '{dest}'")
            return True
        except json.JSONDecodeError:
            print(f"错误：'{source_path}' 不是有效的JSON档案")
            return False
        except Exception as e:
            print(f"错误：复制过程发成错误- {str(e)}")
            return False

def load_settings():
    settings_file = "settings.json"
    character_file = "character.json"
    action_file = "action.json"
    custom_settings_file = "custom_settings.json"
    custom_character_file = "custom_character.json"
    custom_action_file = "custom_action.json"

    # Read saved settings
    settings = get_config2(settings_file)
    character = get_config2(character_file)
    action = get_config2(action_file)
    
    copy_json_file(settings_file,custom_settings_file)
    copy_json_file(character_file,custom_character_file)
    copy_json_file(action_file,custom_action_file)

    try:
        settings = get_config2(custom_settings_file)
    except:
        print(f"错误：自定设置 '{custom_settings_file}' 不存在")

    try:
        character = get_config2(custom_character_file)
    except:
        print(f"错误：自定人物 '{custom_character_file}' 不存在")

    try:
        action = get_config2(custom_action_file)
    except:
        print(f"错误：自定动作 '{custom_action_file}' 不存在")

    #設定
    hm_config_1 = "custom_character.json"
    hm_config_2 = "custom_action.json"
    #hm_config_7 = "wai_character.json"
    #hm_config_8 = "wai_character2.json"
    
    #if(chk_character(hm_config_7) == False):
        #print("角色檔1:" + settings["wai_json_url1"] + " 下載中")
        #download_json(settings["wai_json_url1"], os.path.join(root_dir, "wai_character.json"))
        #print("角色檔1 下載完成")

    #if(chk_character(hm_config_8) == False):
    #    print("角色檔2:" + settings["wai_json_url2"] + " 下載中")
    #    download_json(settings["wai_json_url2"], os.path.join(root_dir, "wai_character2.json"))
    #    print("角色檔2 下載完成")

    hm_config_1_component = get_config2(hm_config_1)
    #for item in get_character(hm_config_7):
    #    hm_config_1_component.update({item : item})
    num_parts = 10
    hm_config_1_img = []
    for i in range(num_parts):            
        for item in get_config2(f"output_{i+1}.json"):
            hm_config_1_img.append(item)
            #key = list(item.keys())[0]
            #hm_config_1_component.update({key : key})

    hm_config_1_img = sorted(hm_config_1_img, key=lambda x: list(x.keys())[0])
    for item in hm_config_1_img:
        key = list(item.keys())[0]
        hm_config_1_component.update({key : key})

    hm_config_2_component = get_config2(hm_config_2)

    #hm_config_1_img = get_characterimg(hm_config_8)
    #for item in get_characterimg(hm_config_8):
    #    hm_config_1_img.append(item)
    
    localizations = "zh_TW.json"
    localizations_component = get_config2(localizations)
    relocalizations_component = {value: key for key, value in localizations_component.items()}

    return hm_config_1_component, hm_config_2_component, localizations_component        
    
class CharacterSelectLoader:
    def __init__(self, *args, **kwargs):
        # components that pass through after_components

        self.settings_file = "settings.json"
        self.character_file = "character.json"
        self.action_file = "action.json"
        self.custom_settings_file = "custom_settings.json"
        self.custom_character_file = "custom_character.json"
        self.custom_action_file = "custom_action.json"

        # Read saved settings
        self.settings = self.get_config2(self.settings_file)
        self.character = self.get_config2(self.character_file)
        self.action = self.get_config2(self.action_file)
        
        self.copy_json_file(self.settings_file,self.custom_settings_file)
        self.copy_json_file(self.character_file,self.custom_character_file)
        self.copy_json_file(self.action_file,self.custom_action_file)

        try:
            self.settings = self.get_config2(self.custom_settings_file)
        except:
            print(f"错误：自定设置 '{self.custom_settings_file}' 不存在")

        try:
            self.character = self.get_config2(self.custom_character_file)
        except:
            print(f"错误：自定人物 '{self.custom_character_file}' 不存在")

        try:
            self.action = self.get_config2(self.custom_action_file)
        except:
            print(f"错误：自定动作 '{self.custom_action_file}' 不存在")

        #設定
        self.hm_config_1 = "custom_character.json"
        self.hm_config_2 = "custom_action.json"
        #self.hm_config_7 = "wai_character.json"
        #self.hm_config_8 = "wai_character2.json"
        
        #if(self.chk_character(self.hm_config_7) == False):
            #print("角色檔1:" + self.settings["wai_json_url1"] + " 下載中")
            #self.download_json(self.settings["wai_json_url1"], os.path.join(root_dir, "wai_character.json"))
            #print("角色檔1 下載完成")

        #if(self.chk_character(self.hm_config_8) == False):
        #    print("角色檔2:" + self.settings["wai_json_url2"] + " 下載中")
        #    self.download_json(self.settings["wai_json_url2"], os.path.join(root_dir, "wai_character2.json"))
        #    print("角色檔2 下載完成")

        self.hm_config_1_component = self.get_config2(self.hm_config_1)
        #for item in self.get_character(self.hm_config_7):
        #    self.hm_config_1_component.update({item : item})
        num_parts = 10
        self.hm_config_1_img = []
        for i in range(num_parts):            
            for item in self.get_config2(f"output_{i+1}.json"):
                self.hm_config_1_img.append(item)
                #key = list(item.keys())[0]
                #self.hm_config_1_component.update({key : key})

        self.hm_config_1_img = sorted(self.hm_config_1_img, key=lambda x: list(x.keys())[0])
        for item in self.hm_config_1_img:
            key = list(item.keys())[0]
            self.hm_config_1_component.update({key : key})

        self.hm_config_2_component = self.get_config2(self.hm_config_2)

        #self.hm_config_1_img = self.get_characterimg(self.hm_config_8)
        #for item in self.get_characterimg(self.hm_config_8):
        #    self.hm_config_1_img.append(item)
        
        self.localizations = "zh_TW.json"
        self.localizations_component = self.get_config2(self.localizations)
        self.relocalizations_component = {value: key for key, value in self.localizations_component.items()}

    
    def get_config(self, path, open_mode='r'):
        file = os.path.join(root_dir, path)
        try:
            with open(file, open_mode) as f:
                as_dict = json.load(f) 
        except FileNotFoundError as e:
            print(f"{e}\n{file} not found, check if it exists or if you have moved it.")
        return as_dict 
    def get_config2(self, path, open_mode='r'):
        file = os.path.join(root_dir, path)
        try:
            with open(file, open_mode, encoding='utf-8') as f:
                as_dict = json.load(f) 
        except FileNotFoundError as e:
            print(f"{e}\n{file} not found, check if it exists or if you have moved it.")
        return as_dict
    
    def copy_json_file(self, source_path: str, destination_path: str, overwrite: bool = False):
        """
        複製JSON檔案並確認其格式正確
        Parameters:
        source_path (str): 來源JSON檔案的路徑
        destination_path (str): 目標位置的路徑
        overwrite (bool): 若為True則覆寫已存在的檔案，預設為False
        Returns:
        bool: 複製成功返回True，失敗返回False
        """
        try:
            # 確認來源檔案存在
            file = Path(os.path.join(root_dir, source_path))
            if not file.exists():
                print(f"错误：来源 '{source_path}' 不存在")
                return False
            
            # 檢查目標檔案是否已存在
            dest = Path(os.path.join(root_dir, destination_path))
            if dest.exists() and not overwrite:
                return False
            
            # 讀取並驗證JSON格式
            #with open(file, 'r', encoding='utf-8') as file:
            #    json.load(file)  # 確認JSON格式正確
            
            # 建立目標資料夾（如果不存在）
            #dest.parent.mkdir(parents=True, exist_ok=True)
            
            # 複製檔案
            shutil.copy2(os.path.join(root_dir, source_path), os.path.join(root_dir, destination_path))
            print(f"成功：档案已复制 '{dest}'")
            return True
        except json.JSONDecodeError:
            print(f"错误：'{source_path}' 不是有效的JSON档案")
            return False
        except Exception as e:
            print(f"错误：复制过程发成错误- {str(e)}")
            return False
    
    def func_setting(self, oldprompt,fv0,fv1):
        self.allfuncprompt = ""
        oldprompt = oldprompt.replace(self.settings["nsfw"], "")
        oldprompt = oldprompt.replace(self.settings["more_detail"], "")
        oldprompt = oldprompt.replace(self.settings["less_detail"], "")
        oldprompt = oldprompt.replace(self.settings["quality"], "")
        oldprompt = oldprompt.replace(self.settings["character_enhance"], "")
        if(fv0):
            self.allfuncprompt += self.settings["nsfw"]
        if(fv1):
            self.allfuncprompt += self.settings["quality"]
        oldprompt += self.allfuncprompt
        return oldprompt
    def base64_to_pil(self, base64_str):
        """將 base64 字串轉換為 PIL Image"""
        if "base64," in base64_str:  # 處理 data URL 格式
            base64_str = base64_str.split("base64,")[1]
    
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image

    def hm1_setting(self, selection, oldprompt):
        if(selection == ""):
            selection = "random"
        value = "data:image/jpeg;base64,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"
        index = 0
        i=0
        for item in self.hm_config_1_img:
            i+=1
            if(item.get(selection,'') != ''):
                value = item.get(selection)
                index = i

        #self.base64_to_pil(self.hm_config_1_img[0].get('hatsune miku'))
        return [self.base64_to_pil(value), oldprompt, index, self.relocalizations_component[selection]]
    def pil2tensor(self, image):
        """将PIL图像转换为ComfyUI兼容的张量格式"""
        import numpy as np
        import torch
        if image:
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image_np = np.array(image).astype(np.float32) / 255.0
            return torch.from_numpy(image_np)[None,]
        return torch.zeros((1, 512, 512, 3))
    def cprompt_send(self, oldprompt, input_prompt):
        generated_texts = []
        generated_texts = self.send_request(input_prompt)
        #clear beafore
        self.oldcprompt = ''
        for text in generated_texts:
            self.oldcprompt += text
        self.oldcprompt = self.oldcprompt.replace(", ", ",") 
        oldprompt = oldprompt + ',' + self.oldcprompt
        print(f"llama3: {self.oldcprompt}")
        return oldprompt
    
    def send_request(self, input_prompt, **kwargs):
        prime_directive = textwrap.dedent("""\
            Act as a prompt maker with the following guidelines:               
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt with the component format:
            1. Start with the subject and keyword description.
            2. Follow with motion keyword description.
            3. Follow with scene keyword description.
            4. Finish with background and keyword description.
            - Limit yourself to no more than 20 keywords per component  
            - Include all the keywords from the user's request verbatim as the main subject of the response.
            - Be varied and creative.
            - Always reply on the same line and no more than 100 words long. 
            - Do not enumerate or enunciate components.
            - Create creative additional information in the response.    
            - Response in English.
            - Response prompt only.                                                
            The followin is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and cosider the elemnts relationship.
            Example:
            Demon Hunter,Cyber City,A Demon Hunter,standing,lone figure,glow eyes,deep purple light,cybernetic exoskeleton,sleek,metallic,glowing blue accents,energy weapons,Fighting Demon,grotesque creature,twisted metal,glowing red eyes,sharp claws,towering structures,shrouded haze,shimmering energy,                            
            Make a prompt for the following Subject:
            """)
        data = {
                'model': self.settings["model"],
                'messages': [
                    {"role": "system", "content": prime_directive},
                    {"role": "user", "content": input_prompt + ";Response in English"}
                ],  
            }
        headers = kwargs.get('headers', {"Content-Type": "application/json", "Authorization": "Bearer " + self.settings["api_key"]})
        base_url = self.settings["base_url"]
        response = requests.post(base_url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            return []
    @classmethod
    def INPUT_TYPES(cls):
        hm_config_1_component, hm_config_2_component, localizations_component = load_settings()
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "character": (["None"]+ list(hm_config_1_component.keys()),),
                "character_zh":(["None"]+ list(localizations_component.keys()),),
                "action": (["None"]+ list(hm_config_2_component.keys()),),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "random_character": ("BOOLEAN", {"default": False}),
                "random_action": ("BOOLEAN", {"default": False}),
                "nsfw": ("BOOLEAN", {"default": False}),
                "quality": ("BOOLEAN", {"default": True}),
                "format_tags": ("BOOLEAN", {"default": True}),
                # ai填充
                "ai_fill": ("BOOLEAN", {"default": False}),
                "text": ("STRING", {"multiline": True, "placeholder": "Input text"}),
            }
        }
    
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING","STRING","IMAGE",)
    RETURN_NAMES = ("prompt","negative_prompt","image",)
    CATEGORY = "ComfyUI-pxtool"


    def execute(self, character, character_zh, action, seed, random_character, random_action, nsfw,  
                quality,  prompt, format_tags, text,ai_fill):
        # 初始化随机种子
        random.seed(seed)
        # 处理角色选择逻辑
        selected_character = None
        if character != "None" and character != "random":
            selected_character = self.hm_config_1_component[character]
        if character_zh != "None":
            selected_character = self.localizations_component[character_zh]
        if random_character or (selected_character == "random"):
            selected_character = random.choice(list(self.hm_config_1_component.values()))
        if selected_character is None:
            selected_character = ""
        else:
            for key, value in self.hm_config_1_component.items():
                if value == selected_character:
                    character = key
        # 处理动作选择逻辑
        selected_action = None
        if action != "None":
            selected_action = self.hm_config_2_component[action]
        else:
            selected_action = ""
        if random_action:
            selected_action = random.choice(list(self.hm_config_2_component.values())) 
            selected_action =selected_action + ","
        if format_tags:
            selected_character = format_str(selected_character) 
            selected_character= selected_character + ", "
        prompt = prompt  + selected_character + selected_action 
        prompt = self.func_setting(prompt,nsfw,quality)
        prompt = remove_duplicate_tags((prompt,))[0]
        result = self.hm1_setting(character, prompt)
        negative_prompt = self.settings["neg_prompt"]
        image = self.pil2tensor(result[0])
        if ai_fill:
            prompt = self.cprompt_send(prompt, text)
        return (prompt, negative_prompt, image)

NODE_CLASS_MAPPINGS5 = {"CharacterSelectLoader": CharacterSelectLoader}

NODE_DISPLAY_NAME_MAPPINGS5 = {"CharacterSelectLoader": "角色动作选择器"}