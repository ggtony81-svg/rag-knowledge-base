"""
Agent 演示：让 DeepSeek 能调用工具
模拟两个工具：查天气、算数学
"""
import requests
import json
from config import DEEPSEEK_API_KEY

# ============================================================
# 第1部分：定义工具
# ============================================================

def get_weather(city):
    """模拟查天气（实际项目里调天气 API）"""
    weather_data = {
        "北京": "25°C，晴转多云",
        "上海": "28°C，有雨",
        "广州": "32°C，闷热",
        "深圳": "30°C，阵雨",
    }
    return weather_data.get(city, f"{city}：暂时没有天气数据")

def calculate(expression):
    """计算数学表达式"""
    try:
        # 安全计算，只允许基本运算
        allowed = "0123456789+-*/(). "
        safe_expr = "".join(c for c in expression if c in allowed)
        result = eval(safe_expr)
        return f"{expression} = {result}"
    except:
        return f"无法计算：{expression}"

# 工具的"说明书"（告诉模型有哪些工具可以用）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 1+2*3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# 第2部分：调用 DeepSeek（支持 function calling）
# ============================================================

def call_deepseek_with_tools(messages):
    """调 DeepSeek，允许它返回工具调用指令"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": tools,        # 告诉模型有哪些工具可用
        "tool_choice": "auto"  # 让模型自己决定要不要用工具
    }
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    return resp.json()

# ============================================================
# 第3部分：运行 Agent
# ============================================================

def run_agent(user_input):
    """完整的 Agent 流程"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，必要时使用工具来回答用户的问题。"},
        {"role": "user", "content": user_input}
    ]

    print(f"🙋 用户：{user_input}")
    print()

    # 第1次调用模型
    response = call_deepseek_with_tools(messages)
    msg = response["choices"][0]["message"]

    # 如果模型不需要调工具，直接回答
    if not msg.get("tool_calls"):
        print(f"🤖 助手：{msg['content']}")
        return

    # 模型决定要调工具
    # 先把模型的 tool_calls 消息加到对话里（只加一次）
    messages.append(msg)

    for tool_call in msg["tool_calls"]:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        print(f"🔧 模型决定调工具：{func_name}")
        print(f"   参数：{args}")

        # 执行对应的工具函数
        if func_name == "get_weather":
            result = get_weather(**args)
        elif func_name == "calculate":
            result = calculate(**args)
        else:
            result = "未知工具"

        print(f"   结果：{result}")
        print()

        # 每个工具调用的结果分别加进去
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result
        })

    # 第2次调用模型（带着工具结果，让模型回答）
    final_response = call_deepseek_with_tools(messages)
    final_answer = final_response["choices"][0]["message"]["content"]
    print(f"🤖 助手：{final_answer}")
    print()

# ============================================================
# 第4部分：测试
# ============================================================

print("=" * 60)
print("🤖 Agent 功能演示")
print("=" * 60)
print()

run_agent("明天北京天气怎么样？")
print("-" * 60)
print()
run_agent("计算一下 (15+27)*3 等于多少？")
print("-" * 60)
print()
run_agent("北京天气怎么样？顺便算一下 25*4 等于多少？")
