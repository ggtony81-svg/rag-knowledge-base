"""
MCP 演示：先看你现在的方式，再看 MCP 方式
"""
import json

# ============================================================
# 你现在的方式：Function Calling（你已经会的）
# ============================================================
print("=" * 60)
print("你现在的 Function Calling 方式")
print("=" * 60)

# 工具函数
def get_weather(city):
    data = {"北京": "25°C", "上海": "28°C", "广州": "32°C"}
    return data.get(city, "无数据")

def calculate(expr):
    try: return str(eval(expr))
    except: return "无法计算"

# 手动编写工具描述
tools_manual = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "算数学",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        }
    }
]

# 手动路由
def route_tool(name, args):
    if name == "get_weather": return get_weather(**args)
    if name == "calculate": return calculate(**args)
    return "未知工具"

print(f"工具描述方式：手动编写 {len(tools_manual)} 个工具")
print(f"路由方式：if/else 判断")
print()

# ============================================================
# MCP 方式：服务端自动注册 + 客户端自动发现
# ============================================================
print("=" * 60)
print("MCP 方式（对比）")
print("=" * 60)

# MCP 核心思想：工具由服务器"广播"，客户端"自动发现"
# 下面模拟这个过程

class MCPServer:
    """模拟 MCP 服务器 — 自动注册和描述工具"""
    def __init__(self):
        self._tools = {}

    def tool(self, name, description):
        """装饰器：自动注册工具"""
        def decorator(func):
            self._tools[name] = {
                "func": func,
                "description": description,
                "schema": self._infer_schema(func)
            }
            return func
        return decorator

    def _infer_schema(self, func):
        """自动从函数签名推断参数格式"""
        import inspect
        sig = inspect.signature(func)
        props = {}
        for p_name, p_param in sig.parameters.items():
            props[p_name] = {"type": "string", "description": f"{p_name}参数"}
        return {
            "type": "object",
            "properties": props,
            "required": list(props.keys())
        }

    def list_tools(self):
        """MCP 标准接口：列出所有工具（自动生成描述）"""
        return [
            {
                "name": name,
                "description": info["description"],
                "input_schema": info["schema"]
            }
            for name, info in self._tools.items()
        ]

    def call_tool(self, name, args):
        """MCP 标准接口：调用工具"""
        if name in self._tools:
            return self._tools[name]["func"](**args)
        return "未知工具"

# 用 MCP 方式写同样的工具
server = MCPServer()

@server.tool("get_weather", "查询某个城市的天气")
def mcp_weather(city):
    data = {"北京": "25°C", "上海": "28°C", "广州": "32°C"}
    return data.get(city, "无数据")

@server.tool("calculate", "计算数学表达式")
def mcp_calculate(expression):
    try: return str(eval(expression))
    except: return "无法计算"

# MCP 自动做的两件事
tools_mcp = server.list_tools()
print(f"工具描述方式：自动生成（从函数签名推断）")
print(f"自动生成的工具列表：")
for t in tools_mcp:
    print(f"  - {t['name']}: {t['description']}")
    print(f"    参数: {list(t['input_schema']['properties'].keys())}")

print()
print(f"路由方式：server.call_tool(name, args) — 一行通用")
print(f"不用写 if/else")
print()

# ============================================================
# 核心区别
# ============================================================
print("=" * 60)
print("核心区别")
print("=" * 60)
print("""
你现在的方式（Function Calling）:
  ✓ 你自己能控制全部
  ✗ 加一个工具要写三段代码（函数 + 描述 + if/else）
  ✗ 别人用不了你的工具
  ✗ 你用不了别人的工具

MCP 方式:
  ✓ 工具即插即用，加工具只需要写函数
  ✓ 工具可以跨项目、跨语言复用
  ✓ 别人发布了 MCP 服务器，你直接连上就能用
  ✓ 客户端自动发现工具，不需要手动注册

比喻:
  Function Calling = 你自己焊电路板（每个零件自己接线）
  MCP = USB-C 接口（插上就能用）

你现在学 Function Calling 是打基础
MCP 是用标准方式把工具"包装"起来
两者不冲突，可以一起用
""")

# ============================================================
# 实际 MCP 用法（模拟客户端发现）
# ============================================================
print("=" * 60)
print("模拟 MCP 客户端：自动发现服务器上的工具")
print("=" * 60)

# 模拟：客户端连上服务器后，自动获取工具列表
print("\n客户端启动，连接 MCP 服务器...")
print(f"自动发现工具: {[t['name'] for t in server.list_tools()]}")

# 然后客户端可以自动把这些工具描述发给大模型
print("\n自动生成 tools 描述（不用手写！）：")
mcp_tools = server.list_tools()
for t in mcp_tools:
    print(f"  {t['name']}: schema 自动生成完成")

print()
print("用户问：北京天气怎么样？")
print(f"客户端调工具 → server.call_tool('get_weather', {{'city': '北京'}})")
print(f"结果：{server.call_tool('get_weather', {'city': '北京'})}")
