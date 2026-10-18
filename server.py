from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:2b"
MAX_BODY_BYTES = 12_000


def clean_text(value: Any, limit: int, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    return re.sub(r"\s+", " ", value).strip()[:limit]


def build_prompt(city: str, facts: dict[str, Any]) -> str:
    assessment = clean_text(facts.get("assessment"), 900)
    growth = clean_text(facts.get("growth"), 32)
    growth_year = clean_text(facts.get("growthYear"), 32, "2025")
    growth_metric = clean_text(facts.get("growthMetric"), 80, "GDP 增长率")
    growth_period = clean_text(facts.get("growthPeriod"), 100, f"{growth_year} 年 GDP 同比")
    basis = clean_text(facts.get("basis"), 240)
    opportunities = facts.get("opportunities") if isinstance(facts.get("opportunities"), list) else []
    risks = facts.get("risks") if isinstance(facts.get("risks"), list) else []
    opportunities = [clean_text(item, 180) for item in opportunities[:4] if isinstance(item, str)]
    risks = [clean_text(item, 180) for item in risks[:4] if isinstance(item, str)]
    supplied_data = {
        "city": clean_text(city, 80),
        "period": growth_period,
        "year": growth_year,
        "indicator": growth_metric,
        "value": growth,
        "basis": basis,
        "curated_assessment": assessment,
        "opportunities_from_statistics": opportunities,
        "risks_to_monitor": risks,
    }
    return (
        "请只根据下面这组已收录统计数据，用中文写一段 100 至 180 字的城市发展评估。"
        "不得新增任何数字、城市事实、数据来源或预测；不要把分析判断伪装成统计事实。"
        "准确保留年份、指标名称、数值和统计口径；不可把就业增长、名义 GDP 增幅或缺失指标写成实际 GDP 增速。"
        "用一两句概括增长结构，再说明一个机会和一个观察风险。"
        "如果资料不足，明确指出不足，不要自行补全。仅输出评估正文，不要标题、Markdown 或思维过程。\n\n"
        + json.dumps(supplied_data, ensure_ascii=False)
    )


def local_ollama_request(city: str, facts: dict[str, Any]) -> str:
    model = clean_text(os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL), 100, DEFAULT_MODEL)
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "你是谨慎的城市统计资料整理助手。不得编造信息。"},
            {"role": "user", "content": build_prompt(city, facts)},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.2, "num_predict": 320},
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=150) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RuntimeError(f"本地没有找到模型 {model}。请先运行：ollama pull {model}") from None
        raise RuntimeError(f"本地 Ollama 返回错误（HTTP {error.code}）。") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("本机 Ollama 尚未启动。请先安装并启动 Ollama，再试一次。") from None
    except (json.JSONDecodeError, OSError):
        raise RuntimeError("本机 Ollama 返回内容无法读取，请重试。") from None
    message = payload.get("message", {}) if isinstance(payload, dict) else {}
    answer = clean_text(message.get("content") if isinstance(message, dict) else "", 1600)
    if not answer:
        raise RuntimeError("本机模型没有返回评估内容，请重试。")
    return answer


class Handler(BaseHTTPRequestHandler):
    server_version = "CityScopeLocal/1.0"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        try:
            body = (ROOT / "index.html").read_bytes()
        except OSError:
            self.send_error(500, "Cannot read index.html")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/city-assessment":
            self.send_json(404, {"error": "未找到此接口。"})
            return
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            self.send_json(403, {"error": "只允许本地页面发起请求。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "请求长度无效。"})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self.send_json(413, {"error": "请求内容过大或为空。"})
            return
        try:
            request_data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "请求格式无效。"})
            return
        if not isinstance(request_data, dict):
            self.send_json(400, {"error": "请求格式无效。"})
            return
        city = clean_text(request_data.get("city"), 80)
        facts = request_data.get("data")
        if not city or not isinstance(facts, dict):
            self.send_json(400, {"error": "缺少城市或统计数据。"})
            return
        try:
            assessment = local_ollama_request(city, facts)
        except RuntimeError as error:
            self.send_json(503, {"error": str(error)})
            return
        self.send_json(200, {"assessment": assessment})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {format % args}")


if __name__ == "__main__":
    print(f"城势本地服务：http://{HOST}:{PORT}")
    print("页面数据可离线使用；本地 AI 需先安装并启动 Ollama。按 Ctrl+C 停止服务。")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    webbrowser.open(f"http://{HOST}:{PORT}")
    server.serve_forever()
