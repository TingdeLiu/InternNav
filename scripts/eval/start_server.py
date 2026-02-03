#!/usr/bin/env python
"""
脚本用途：启动 InternNav 的 `AgentServer` 服务进程，用于评估/调用导航 Agent。

主要流程：
- 设置 Python 模块搜索路径，确保项目与依赖可被导入。
- 通过 `--config` 动态加载评估配置（可覆盖端口等设置）。
- 解析命令行参数（host/port/reload 等）。
- 构造并运行 `AgentServer`。

AgentServer 是 InternNav 的服务端封装类，用于在指定 host/port 上启动进程，对外提供接口以驱动/评估导航 Agent。

职责：初始化通信层（如 HTTP/WebSocket/自定义 RPC）、接收请求（观测/指令/配置）、调用已注册的 Agent 推理、返回动作/状态、管理会话与日志/错误。
start_server.py 仅作为入口：解析参数与可选配置，实例化 AgentServer(host, port)，调用 run() 启动服务。
具体实现位于 internnav/utils 模块内。
"""
import sys

sys.path.append('.')
sys.path.append('./src/diffusion-policy')
# 以上两行将项目根目录与 diffusion-policy 源码目录加入模块搜索路径，
# 便于后续 `import` 正常找到内部包与第三方子仓库。

import argparse
import importlib
import importlib.util
import sys

# Import for agent registry side effects — do not remove
from internnav.agent import Agent  # noqa: F401
# 注意：该导入用于触发 Agent 注册的副作用（例如通过装饰器注册），不要删除。
# 实际运行过程中可能并不直接使用 `Agent` 符号，但其导入对系统初始化必要。
from internnav.utils import AgentServer


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    """
    动态加载评估配置模块，并返回其中名为 `attr_name` 的配置对象（默认 `eval_cfg`）。

    参数：
    - config_path: 配置文件的路径（.py），例如 scripts/eval/configs/h1_cma_cfg.py。
    - attr_name: 需要从模块中读取的属性名，缺省为 'eval_cfg'。

    过程：
    1) 通过 importlib.util.spec_from_file_location 创建模块规范。
    2) 基于规范实例化模块对象并注册到 sys.modules。
    3) 使用规范的 loader 执行模块（等价于导入）。
    4) 从模块中获取指定属性并返回。
    """
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


if __name__ == '__main__':
    print("Starting Agent Server...")

    parser = argparse.ArgumentParser()
    # 服务器监听地址（默认 localhost）
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    # 端口号（可被配置文件中的 eval_cfg.agent.server_port 覆盖）
    parser.add_argument('--port', type=int, default=8087)
    # 可选：热重载开关（如上层支持），当前脚本未直接使用，仅占位
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    if args.config:
        # 若提供配置文件，则动态加载并使用其中的端口设置
        eval_cfg = load_eval_cfg(args.config)
        args.port = eval_cfg.agent.server_port
    else:
        print(f"Warning: No config file provided, using port {args.port}")

    # 启动 AgentServer，开始监听并服务客户端请求
    server = AgentServer(args.host, args.port)
    server.run()
