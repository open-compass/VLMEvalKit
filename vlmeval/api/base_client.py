import requests
import json
import base64
from typing import List, Dict, Any
import time


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:9100/v1/chat/completions", app_code: str = 'B0m6Tuglt5shfY7t3GyoJn1V5yVAm0Ba'):
        """
        初始化vLLM客户端
        
        Args:
            base_url: vLLM server地址
        """
        self.base_url = base_url
        self.app_code = app_code
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def create_messages_with_images(
        self, 
        prompt: str, 
        image_paths: List[str], 
        image_format: str = "base64"
    ) -> List[Dict]:
        """
        创建包含图片的消息
        
        Args:
            prompt: 文本提示词
            image_paths: 图片路径列表
            image_format: 图片格式，支持"base64"或"url"
            
        Returns:
            消息列表
        """
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 添加文本部分
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        
        # 添加图片部分
        for image_path in image_paths:
            if image_format == "base64":
                # 读取并编码图片
                base64_image = self.encode_image_to_base64(image_path)
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            elif image_format == "url":
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                }
            else:
                raise ValueError(f"不支持的图片格式: {image_format}")
            
            messages[0]["content"].append(image_content)
        
        return messages
    
    def stream_completion(
        self,
        prompt: str = None,
        messages: List[Dict] = None,
        image_paths: List[str] = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = None,
        stream: bool = True,
        **kwargs
    ):
        """
        流式输出请求
        
        Args:
            prompt: 文本提示词（如果使用messages参数，则忽略此参数）
            messages: 消息列表（支持多模态）
            image_paths: 图片路径列表
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            top_p: top-p采样参数
            stream: 是否使用流式输出
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        # 构建请求体
        
        request_data = {
            "model": model,
            "stream": stream,
            **kwargs
        }
        if temperature is not None:
            request_data["temperature"] = temperature
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens 
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens 
        print("request_data:",request_data)
        # 处理消息
        if messages is not None:
            request_data["messages"] = messages
        elif image_paths is not None:
            # 如果有图片路径，创建包含图片的消息
            if prompt is None:
                prompt = "请描述图片内容"
            request_data["messages"] = self.create_messages_with_images(prompt, image_paths)
        elif prompt is not None:
            # 纯文本消息
            request_data["messages"] = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            raise ValueError("必须提供prompt、messages或image_paths中的一个")
        
        # 发送请求
        response = requests.post(
            self.base_url,
            json=request_data,
            stream=True,
            headers={"Content-Type": "application/json",'Authorization':self.app_code}
        )
        
        if response.status_code != 200:
            raise Exception(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
        
        # 处理流式响应
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # 跳过SSE事件开始标记
                if line.startswith('data: '):
                    data = line[6:]  # 去掉"data: "前缀
                    
                    # 检查是否为结束标记
                    if data == '[DONE]':
                        break
                    
                    try:
                        # 解析JSON
                        json_data = json.loads(data)
                        
                        # 提取内容
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                full_response += content
                                yield content
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}, 原始数据: {data}")
        
        # 返回完整响应
        return full_response
    
    def non_stream_completion(
        self,
        prompt: str = None,
        messages: List[Dict] = None,
        image_paths: List[str] = None,
        model: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        非流式输出请求
        
        Args:
            参数同stream_completion
            
        Returns:
            完整的响应
        """
        # 构建请求体
        request_data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **kwargs
        }
        
        # 处理消息（同流式版本）
        if messages is not None:
            request_data["messages"] = messages
        elif image_paths is not None:
            if prompt is None:
                prompt = "请描述图片内容"
            request_data["messages"] = self.create_messages_with_images(prompt, image_paths)
        elif prompt is not None:
            request_data["messages"] = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            raise ValueError("必须提供prompt、messages或image_paths中的一个")
        
        # 发送请求
        response = requests.post(
            self.base_url,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
        
        return response.json()
