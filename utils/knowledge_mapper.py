import requests
import json
from typing import Optional, Dict, List
from pyvis.network import Network
import networkx as nx

class KnowledgeMapper:
    def __init__(self, qwen_api_key: str):
        self.api_key = qwen_api_key
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_mindmap(self, content: str) -> Optional[Dict]:
        """生成思维导图的数据结构"""
        try:
            # 构建prompt
            prompt = f"""请根据以下内容生成一个知识点的思维导图，给出对应的pyvis代码。
要求：
1. 使用层级结构展示知识点
2. 每个节点包含id、label、level属性
3. 每个边包含from、to属性
4. 使用简洁的节点标签
5. 确保返回的是对应的pyvis代码
6. 不要包含任何额外的文本或解释
7. 不要使用markdown代码块标记

内容：
{content}"""

            # 调用通义千问API
            payload = {
                "model": "qwen-turbo",
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            }
            
            response = requests.post(self.url, headers=self.headers, json=payload)
            response_data = response.json()
            
            if response.status_code == 200:
                # 获取响应文本
                response_text = response_data.get("output", {}).get("text", "")
                
                # 清理可能的markdown代码块标记和其他非代码内容
                response_text = response_text.replace("```python", "").replace("```", "").strip()
                
                # 创建Network对象
                net = Network(height='1000px', width='100%', directed=True)
                
                # 执行pyvis代码
                try:
                    # 创建一个本地命名空间来执行代码
                    local_vars = {'net': net}
                    exec(response_text, globals(), local_vars)
                    
                    # 获取执行后的Network对象
                    net = local_vars['net']
                    
                    # 将Network对象转换为字典格式
                    nodes = []
                    edges = []
                    
                    for node in net.nodes:
                        nodes.append({
                            'id': node['id'],
                            'label': node['label'],
                            'level': node.get('level', 1)
                        })
                    
                    for edge in net.edges:
                        edges.append({
                            'from': edge['from'],
                            'to': edge['to']
                        })
                    
                    return {
                        'nodes': nodes,
                        'edges': edges
                    }
                    
                except Exception as e:
                    print(f"执行pyvis代码时出错: {str(e)}")
                    print(f"原始代码: {response_text}")
                    return None
            else:
                print(f"API调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"生成思维导图时出错: {str(e)}")
            return None

    def create_network(self, mindmap_data: Dict) -> Network:
        """创建交互式网络图"""
        # 创建网络图
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # 设置节点颜色映射
        colors = {
            0: "#FFB6C1",  # 浅粉色
            1: "#98FB98",  # 浅绿色
            2: "#87CEFA",  # 浅蓝色
            3: "#DDA0DD",  # 浅紫色
            4: "#F0E68C",  # 浅黄色
        }
        
        # 添加节点
        for node in mindmap_data["nodes"]:
            level = node["level"]
            color = colors.get(level, "#E6E6FA")  # 默认浅紫色
            net.add_node(
                node["id"],
                label=node["label"],
                color=color,
                size=30 - level * 5,  # 层级越高，节点越小
                font={"size": 16 - level * 2}  # 层级越高，字体越小
            )
        
        # 添加边
        for edge in mindmap_data["edges"]:
            net.add_edge(edge["from"], edge["to"])
        
        # 设置布局
        '''''
        net.set_options("""
        {
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 100,
                    "springConstant": 0.01,
                    "nodeDistance": 120,
                    "damping": 0.09
                },
                "solver": "hierarchicalRepulsion",
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            }
        }
        """)
        '''''
        net.set_options("""
{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",  # 方向：Up-Down (UD) 或 Left-Right (LR)
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "nodeDistance": 150
    }
  }
}
""")
        
        return net 