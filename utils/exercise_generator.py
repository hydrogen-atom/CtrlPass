from dashscope import Generation
from typing import List, Dict
import json
import logging
import streamlit as st

class ExerciseGenerator:
    def __init__(self, dashscope_api_key: str):
        self.api_key = dashscope_api_key
        self.model = "qwen-max"  # 使用通义千问大模型
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.exercise_template = """基于以下学习资料，生成3道练习题。每道题应该包含：
        1. 题目描述
        2. 选项（如果是选择题）
        3. 正确答案
        4. 解析
        
        学习资料：
        {content}
        
        请以JSON格式输出，格式如下：
        {{
            "exercises": [
                {{
                    "question": "题目描述",
                    "type": "选择题/填空题/简答题",
                    "options": ["选项A", "选项B", "选项C", "选项D"],  # 仅选择题需要
                    "answer": "正确答案",
                    "explanation": "解析"
                }}
            ]
        }}
        """

    def generate_exercises(self, content: str) -> List[Dict]:
        """生成练习题"""
        try:
            if not content or not content.strip():
                self.logger.error("错误：文档内容为空")
                return []
            
            self.logger.info(f"文档内容长度: {len(content)} 字符")
            self.logger.info(f"文档内容前100个字符: {content[:100]}")
            
            # 准备提示词
            prompt = self.exercise_template.format(content=content)
            
            # 调用阿里云API
            self.logger.info("正在调用通义千问API...")
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                api_key=self.api_key,
                result_format='message',
                temperature=0.7,
                max_tokens=2000,
                top_p=0.8,
                top_k=50,
                enable_search=True,
                incremental_output=False
            )
            
            self.logger.info(f"API响应状态码: {response.status_code}")
            self.logger.info(f"API响应类型: {type(response)}")
            
            if response.status_code == 200:
                # 从响应中提取JSON内容
                self.logger.info(f"Response output type: {type(response.output)}")
                self.logger.info(f"Response output attributes: {dir(response.output)}")
                
                # 获取响应文本
                try:
                    # 打印完整的响应结构以便调试
                    self.logger.info(f"Full response structure: {response.output}")
                    
                    # 尝试获取文本内容
                    if hasattr(response.output, 'choices') and response.output.choices:
                        response_text = response.output.choices[0].get('message', {}).get('content', '')
                    else:
                        response_text = str(response.output)
                    
                    if not response_text:
                        self.logger.error("无法从响应中获取文本内容")
                        return []
                    
                    self.logger.info(f"API响应文本: {response_text[:200]}...")  # 打印前200个字符
                    
                    # 尝试提取JSON部分
                    try:
                        # 查找JSON开始和结束的位置
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        
                        if start_idx == -1 or end_idx == -1:
                            self.logger.error("错误：未找到JSON格式的内容")
                            return []
                        
                        json_str = response_text[start_idx:end_idx]
                        self.logger.info(f"提取的JSON字符串: {json_str[:200]}...")  # 打印前200个字符
                        
                        exercises_data = json.loads(json_str)
                        
                        if not isinstance(exercises_data, dict):
                            self.logger.error(f"错误：解析后的数据不是字典类型，而是 {type(exercises_data)}")
                            return []
                        
                        exercises = exercises_data.get("exercises", [])
                        
                        if not exercises:
                            self.logger.error("错误：生成的练习题列表为空")
                            return []
                        
                        # 验证每个练习题的格式
                        valid_exercises = []
                        for i, exercise in enumerate(exercises):
                            if not isinstance(exercise, dict):
                                self.logger.error(f"错误：练习题 {i+1} 不是字典类型")
                                continue
                                
                            required_fields = ['question', 'type', 'answer', 'explanation']
                            if not all(field in exercise for field in required_fields):
                                self.logger.error(f"错误：练习题 {i+1} 缺少必要字段")
                                continue
                                
                            if exercise['type'] == '选择题' and 'options' not in exercise:
                                self.logger.error(f"错误：练习题 {i+1} 是选择题但缺少选项")
                                continue
                                
                            valid_exercises.append(exercise)
                        
                        if not valid_exercises:
                            self.logger.error("错误：没有有效的练习题")
                            return []
                        
                        self.logger.info(f"成功生成 {len(valid_exercises)} 道练习题")
                        return valid_exercises
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析失败: {str(e)}")
                        self.logger.info("尝试清理响应文本...")
                        # 如果直接解析失败，尝试清理文本
                        cleaned_text = response_text.replace('\n', ' ').strip()
                        try:
                            exercises_data = json.loads(cleaned_text)
                            if not isinstance(exercises_data, dict):
                                self.logger.error(f"错误：清理后解析的数据不是字典类型，而是 {type(exercises_data)}")
                                return []
                                
                            exercises = exercises_data.get("exercises", [])
                            if exercises:
                                self.logger.info(f"清理后成功解析，生成 {len(exercises)} 道练习题")
                                return exercises
                            else:
                                self.logger.error("清理后解析成功，但练习题列表为空")
                                return []
                        except Exception as e:
                            self.logger.error(f"清理后的文本仍然无法解析为JSON: {str(e)}")
                            return []
                except Exception as e:
                    self.logger.error(f"无法从响应中获取文本内容: {str(e)}")
                    return []
            else:
                self.logger.error(f"API调用失败: {response.status_code} - {response.message}")
                return []
            
        except Exception as e:
            self.logger.error(f"生成练习题时出错: {str(e)}")
            return []

    def preview_content(self, content: str):
        st.write("文档内容预览：")
        st.text(content[:500] + "..." if len(content) > 500 else content) 