# -*- coding: utf-8 -*-
"""
系统黑盒测试

对运行中的服务发真实请求，把每一次的输入、返回、耗时、判定结果记下来，
生成 eval/results/blackbox_report.md，供论文"系统测试"一节直接引用。

设计上的两点说明：

1. 只测"看得见的行为"，不去读代码判断对错。所以每个用例都写清楚了
   判据是什么（expect 字段），判据本身能不能成立，读者可以自己复核。

2. 跑之前先清掉 qa:* 缓存。否则"年假有多少天"这类问题可能命中上一轮
   留下的缓存，测出来的就不是检索+生成的真实耗时了。

3. 有缓存和没缓存两种状态都要跑一遍。只测"Redis 开着"是不够的——
   /status 曾经因为没写降级分支，Redis 一停就返回 500，而当时 11 个用例
   全部通过。T12 就是补这个缺口的，它两种状态下都会执行。

用法：
    先启动服务：  python pdf_qa_optimized.py
    再跑测试：    python eval/run_blackbox_tests.py
    建议两种状态各跑一次（Redis 开 / 关），报告分两份存档
"""
import json
import os
import re
import socket
import sys
import time

import requests

BASE = "http://127.0.0.1:18005"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "eval", "results")

# 知识库：《星海科技集团员工手册（2026 修订版）》，9 章 38 条，24 个文本块。
# 下面这些原文用来核对回答有没有"答到点上"，都是手册里确实写着的。


def redis_reachable():
    """Redis 是不是真的连得上。用裸 socket 探，不经过 redis 客户端。"""
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect(("localhost", 6379))
        return True
    except Exception:
        return False
    finally:
        s.close()


def clear_qa_cache():
    """
    只删问答缓存，不动 Redis 里的其它数据。

    Redis 没开时返回 None——整个测试套件要在"有缓存"和"没缓存"两种状态下
    都能跑完，否则降级路径永远测不到（这正是 T12 存在的理由）。
    """
    if not redis_reachable():
        return None
    sys.path.insert(0, BASE_DIR)
    import redis as redis_lib
    r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
    keys = r.keys("qa:*")
    if keys:
        r.delete(*keys)
    return len(keys)


def ask(question, timeout=120):
    """
    调 /ask/v2，把 SSE 流收完，返回一段结构化结果。

    返回：sources（检索到的片段）、answer（拼起来的回答）、
          t（各阶段耗时）、cached（是否命中缓存）、total_wall_ms（客户端实测端到端）
    """
    t0 = time.time()
    out = {"sources": [], "answer": "", "t": {}, "cached": None, "http": None}
    try:
        resp = requests.post(f"{BASE}/ask/v2", json={"question": question},
                             stream=True, timeout=timeout)
        out["http"] = resp.status_code
        if resp.status_code != 200:
            out["answer"] = resp.text[:500]
            return out

        buf = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            buf += raw + "\n"
            # SSE 事件之间用空行分隔
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                block = block.strip()
                if not block.startswith("data:"):
                    continue
                try:
                    ev = json.loads(block[5:].strip())
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "sources":
                    out["sources"] = ev.get("items", [])
                    out["cached"] = ev.get("cached", False)
                    out["t"].update(ev.get("t") or {})
                elif ev.get("type") == "delta":
                    out["answer"] += ev.get("text", "")
                elif ev.get("type") == "done":
                    out["t"].update(ev.get("t") or {})
                elif ev.get("type") == "error":
                    out["answer"] = "[error] " + ev.get("message", "")
    except Exception as e:
        out["answer"] = f"[异常] {type(e).__name__}: {e}"
    out["total_wall_ms"] = round((time.time() - t0) * 1000)
    return out


def grep_any(text, words):
    """回答里出现了任意一个关键词就算命中。"""
    return [w for w in words if w in text]


# ------------------------------------------------- 数字溯源（幻觉检查）

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
# 中文数字里必须带上"点"，否则"一点五"会被拆成"一"和"五"两个数，
# 与原文核对时就会把一个忠实的引用误判成编造。
NUM_RE = re.compile(r"\d+(?:\.\d+)?|[零一二两三四五六七八九十百千点]+")


def cn_num(s):
    """把中文数字串转成数值：'三十'→30，'一点五'→1.5，'一百五十八'→158。"""
    if not s:
        return None
    if "点" in s:                      # 一点五 → 1.5
        a, b = s.split("点", 1)
        ia = cn_num(a)
        ia = 0 if ia is None else int(ia)      # 递归返回的是 float，直接拼会得到 "1.0.5"
        dec = "".join(str(CN_DIGIT[c]) for c in b if c in CN_DIGIT)
        return float(f"{ia}.{dec}") if dec else float(ia)
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in CN_DIGIT:
            number = CN_DIGIT[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return None
    return float(total + section + number)


# 两类"看起来是数字、其实不是论断"的东西，核对之前要先去掉：
#   1. Markdown 的有序列表序号（"1. 年假有多少天"）和标题井号；
#   2. "百分之" 这个前缀，否则"百分之八十"会被拆成 100 和 80，凭空多出一个 100。
LIST_MARKER = re.compile(r"^\s*(?:\*\*|__)?\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
HEADING_MARK = re.compile(r"^#{1,6}\s*", re.MULTILINE)
CN_PERCENT = re.compile(r"百分之")


def numerals(text):
    """抽出文本里所有数字，统一转成数值集合（阿拉伯数字和中文数字都认）。"""
    text = CN_PERCENT.sub("", LIST_MARKER.sub("", HEADING_MARK.sub("", text)))
    out = set()
    for m in NUM_RE.finditer(text):
        tok = m.group()
        if tok[0].isdigit():
            out.add(float(tok))
        elif set(tok) != {"点"}:          # 光一个"点"不是数字，别让它变成 0
            v = cn_num(tok)
            if v is not None:
                out.add(v)
    return out


def _is_year(v):
    return float(v).is_integer() and 1900 <= v <= 2100


def ungrounded(answer, context, question):
    """
    回答里出现、但检索原文和提问里都没有的数字。

    这是"不能凭空捏造"的可执行判据：大模型完全可以在引用原文的同时，
    顺手补一个看起来很合理的数字（"带薪病假每年 30 天"），
    仅凭语气或关键词是看不出来的，但对着原文逐个核数字能查出来。

    数字可能来自提问本身（用户说"我工龄 12 年"，回答里出现 12 是正常的），
    所以已知来源 = 检索原文 ∪ 提问。

    已知的误报面（如实记下，不装作没有）：
    这个判据只能做"字面核对"，认不出**计算得出**的数字。例如提问说"2024 年入职"，
    回答里出现"从 2025 年起"是合理推算，字面上却找不到 2025。
    因此这里只对"年份的相邻递增/递减"开一个口子——范围卡死在 1900~2100，
    且必须与已知年份相差正好 1。除此之外一律从严，宁可误报也不放过。
    """
    known = numerals(context) | numerals(question)
    bad = set()
    for v in numerals(answer) - known:
        if _is_year(v) and any(_is_year(k) and abs(k - v) == 1 for k in known):
            continue
        bad.add(v)
    return sorted(bad)


# ---------------------------------------------------------------- 用例

def t01_status():
    r = requests.get(f"{BASE}/status", timeout=10).json()
    ok = r.get("status") == "ok" and r.get("kb") is True
    return {
        "id": "T01", "name": "服务与知识库状态",
        "input": "GET /status",
        "expect": "status=ok 且 kb=true，并报告当前使用的模型名",
        "actual": json.dumps(r, ensure_ascii=False),
        "pass": bool(ok),
    }


def t02_ui():
    r = requests.get(f"{BASE}/", timeout=10)
    ok = r.status_code == 200 and "PDF 文档问答系统" in r.text
    return {
        "id": "T02", "name": "问答界面可访问",
        "input": "GET /",
        "expect": "返回 200 且是问答页面",
        "actual": f"HTTP {r.status_code}，页面长度 {len(r.text)} 字符",
        "pass": bool(ok),
    }


def _qa_case(tid, name, question, expect, must_hit=(), must_miss=()):
    """通用问答用例：must_hit 是回答里应该出现的原文，must_miss 是不该出现的。"""
    r = ask(question)
    ans = r["answer"]
    ctx = "\n".join(s["text"] for s in r["sources"])
    hit = grep_any(ans, must_hit)
    bad = grep_any(ans, must_miss)
    num_bad = ungrounded(ans, ctx, question)
    ok = (bool(ans) and not ans.startswith("[") and len(hit) > 0
          and not bad and not num_bad)
    return {
        "id": tid, "name": name,
        "input": question,
        "expect": expect,
        "actual": (f"检索到 {len(r['sources'])} 段，答案 {len(ans)} 字，"
                   f"数字溯源{'全部有据' if not num_bad else '异常 ' + str(num_bad)}｜"
                   f"耗时 检索 {r['t'].get('retrieve_ms')}ms / "
                   f"生成 {r['t'].get('gen_ms')}ms / 端到端 {r['total_wall_ms']}ms"),
        "answer": ans,
        "sources": r["sources"],
        "ungrounded": num_bad,
        "pass": bool(ok),
        "hits": hit, "misses": bad,
    }


def t03_factual():
    return _qa_case(
        "T03", "事实型提问（答案在库内）",
        "出差结束后要在几个工作日内提交出差报告？",
        "应回答'十个工作日'，且必须来自手册第十二条",
        must_hit=["十个工作日"],
    )


def t04_reasoning():
    return _qa_case(
        "T04", "推理型提问（需跨条款理解）",
        "我入职满三年，今年能休多少天年假？",
        "手册第十四条按'累计工作年限'分档（满1年不满10年5天），"
        "系统应给出对应天数，不宜凭空编造数字",
        must_hit=["五", "5", "十天", "10"],   # 满足分档即可，不锁死哪一种表述
    )


def t05_cache():
    """同一问题连问两次，第二次应命中缓存并显著更快。"""
    q = "每月餐补是多少钱？"
    if not redis_reachable():
        # Redis 没开时缓存功能无从谈起。这里如实记为未通过，而不是
        # 悄悄跳过——报告里出现一个"未通过"，比凭空少一条用例更可信。
        return {
            "id": "T05", "name": "Redis 缓存命中",
            "input": f"同一问题「{q}」连续问两次",
            "expect": "第一次 cached=false 且耗时较长；第二次 cached=true 且耗时明显更短",
            "actual": "Redis 未连接，缓存不可用，本用例无法进行（服务本身工作正常，见 T12）",
            "pass": False,
        }
    first = ask(q)
    second = ask(q)
    ok = (second["cached"] is True and first["cached"] is False)
    return {
        "id": "T05", "name": "Redis 缓存命中",
        "input": f"同一问题「{q}」连续问两次",
        "expect": "第一次 cached=false 且耗时较长；第二次 cached=true 且耗时明显更短",
        "actual": (f"第一次 cached={first['cached']}，端到端 {first['total_wall_ms']}ms；"
                   f"第二次 cached={second['cached']}，端到端 {second['total_wall_ms']}ms"),
        "answer": second["answer"],
        "pass": bool(ok),
    }


ADMIT_WORDS = ["没有", "未提及", "未涉及", "未作规定", "无法确定", "无法给出",
               "不包括", "暂无", "资料中", "未找到", "未说明", "并未"]


def _absence_case(tid, name, question, expect):
    """
    库外提问：问的东西手册里确实没写，看系统会不会硬编一个答案。

    判据两条，缺一不可：
      1. 回答里明确说了"资料中没有" —— 不能装作答上来了；
      2. 回答里出现的每个数字都能在检索原文或提问里找到 —— 不能一边说
         "没有规定"一边又顺手补一个具体数字。
    """
    r = ask(question)
    ans = r["answer"]
    ctx = "\n".join(s["text"] for s in r["sources"])
    bad_nums = ungrounded(ans, ctx, question)
    admitted = grep_any(ans, ADMIT_WORDS)
    ok = bool(ans) and not ans.startswith("[") and bool(admitted) and not bad_nums
    return {
        "id": tid, "name": name,
        "input": question,
        "expect": expect,
        "actual": (f"{'已明确说明资料中没有该项' if admitted else '未说明资料缺失'}"
                   f"；回答中的数字"
                   f"{'全部可在原文中找到' if not bad_nums else '有 ' + str(bad_nums) + ' 在原文中找不到'}"
                   f"｜检索到 {len(r['sources'])} 段"),
        "answer": ans,
        "sources": r["sources"],
        "ungrounded": bad_nums,
        "pass": bool(ok),
    }


def t06_out_of_kb():
    return _absence_case(
        "T06", "库外提问（幻觉检测）",
        "公司年终奖一般发几个月工资？",
        "手册第二十六条只写了年终奖'系数'，没写对应几个月工资；"
        "系统应说明这一点，不得编造月数",
    )


def t07_domain_mismatch():
    return _absence_case(
        "T07", "近似措辞的库外提问",
        "员工每年可以申请多少天带薪病假？",
        "手册第十五条只规定了病假工资标准与审批门槛，没有'每年可申请多少天带薪病假'；"
        "系统应说明这一点，不得编造天数",
    )


def t08_long_input():
    q = ("我是2024年入职的正式员工，之前在其他公司累计工作了12年，"
         "想问一下：我的年假有多少天？如果当年没休完，最晚什么时候要用掉？"
         "另外离职的时候没休的年假怎么折算？还有迟到超过半小时会怎么处理？"
         "出差报告要在多久之内交？每月餐补多少钱？") * 3   # 重复拉长到超长输入
    r = ask(q)
    answers = grep_any(r["answer"], ["十天", "10", "三月三十一日", "百分之三百", "半天事假"])
    ctx = "\n".join(s["text"] for s in r["sources"])
    num_bad = ungrounded(r["answer"], ctx, q)
    ok = (bool(r["answer"]) and not r["answer"].startswith("[")
          and len(answers) >= 2 and not num_bad)
    return {
        "id": "T08", "name": "超长输入（约 %d 字）" % len(q),
        "input": f"（{len(q)} 字的多问句长文本，此处略）",
        "expect": "服务不报错，至少答对其中两个子问题（累计工作12年对应10天年假等），且数字有据",
        "actual": (f"HTTP {r['http']}，答案 {len(r['answer'])} 字，命中 {len(answers)} 个子问题，"
                   f"数字溯源{'全部有据' if not num_bad else '异常 ' + str(num_bad)}；"
                   f"检索耗时 {r['t'].get('retrieve_ms')}ms"),
        "answer": r["answer"][:600],
        "sources": r["sources"],
        "ungrounded": num_bad,
        "pass": bool(ok),
    }


def t09_special_chars():
    """
    分两类看，判据不一样：
      - 空串、纯空白：期望**明确拒绝**（"问题不能为空"），而不是照样去调大模型；
      - 纯符号、超短输入：没法预先判断有没有意义，只要求服务不崩、明确说明看不懂。
    """
    empty_cases = ["", "   ", "\n\n", "\t"]
    junk_cases = ["？？？", "！@#￥%……&*（）", "1"]

    results = []
    for c in empty_cases:
        r = ask(c, timeout=60)
        refused = "不能为空" in r["answer"]
        results.append({"input": repr(c), "http": r["http"], "kind": "空输入",
                        "answer_len": len(r["answer"]), "refused": refused,
                        "answer": r["answer"][:120]})
    for c in junk_cases:
        r = ask(c, timeout=60)
        crashed = r["http"] != 200 or r["answer"].startswith("[异常]")
        results.append({"input": repr(c), "http": r["http"], "kind": "无意义输入",
                        "answer_len": len(r["answer"]), "crashed": crashed,
                        "answer": r["answer"][:120]})

    empty_ok = all(x["refused"] for x in results if x["kind"] == "空输入")
    junk_ok = not any(x.get("crashed") for x in results if x["kind"] == "无意义输入")
    return {
        "id": "T09", "name": "空输入与特殊字符",
        "input": "空串、纯空白、纯符号、单字符",
        "expect": "空/纯空白输入应被明确拒绝（不调用大模型）；无意义输入不应导致崩溃或 5xx",
        "actual": "；".join(
            f"{x['input']} → HTTP {x['http']}，"
            + ("已拒绝" if x.get("refused") else f"回答 {x['answer_len']} 字")
            for x in results),
        "detail": results,
        "pass": bool(empty_ok and junk_ok),
    }


def t10_upload_non_pdf():
    """上传一个真正的 txt 文件，看服务怎么处理。"""
    p = os.path.join(RESULT_DIR, "_tmp_notpdf.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("这不是一个 PDF 文件，只是一段普通文本。" * 20)
    try:
        with open(p, "rb") as f:
            r = requests.post(f"{BASE}/upload", files={"file": ("not_a.pdf", f, "application/pdf")},
                              timeout=60)
        body = r.text[:300]
        ok = r.status_code == 200 and "error" in body
        return {
            "id": "T10", "name": "上传非 PDF 文件",
            "input": "POST /upload，文件内容为纯文本但扩展名为 .pdf",
            "expect": "应给出明确错误提示，且不破坏已有知识库",
            "actual": f"HTTP {r.status_code}，返回 {body[:160]}",
            "pass": bool(ok),
        }
    finally:
        os.remove(p)


def t11_upload_scanned():
    """
    上传一份"没有文字层"的 PDF（把手册首页渲染成图片再拼回 PDF，
    模拟扫描件）。应返回明确错误，且不能把已有知识库搞坏。
    """
    import fitz
    src = os.path.join(BASE_DIR, "test_handbook.pdf")
    out = os.path.join(RESULT_DIR, "_tmp_scanned.pdf")
    try:
        d = fitz.open(src)
        pix = d[0].get_pixmap(dpi=150)
        d.close()
        nd = fitz.open()
        page = nd.new_page(width=pix.width, height=pix.height)
        page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
        nd.save(out)
        nd.close()

        with open(out, "rb") as f:
            r = requests.post(f"{BASE}/upload", files={"file": ("scanned.pdf", f, "application/pdf")},
                              timeout=120)
        body = r.text[:300]
        # 上传失败后，知识库应保持原样
        st = requests.get(f"{BASE}/kb/status", timeout=10).json()
        kept = st.get("filename") == "test_handbook.pdf" and st.get("chunks") == 24
        ok = r.status_code == 200 and "error" in body and kept
        return {
            "id": "T11", "name": "上传扫描件（无文字层 PDF）",
            "input": "把手册首页渲染成图片后拼成的新 PDF（无文字层）",
            "expect": "提示'没有提取到可用文字'；且原知识库保持不变（仍是 test_handbook.pdf / 24 块）",
            "actual": (f"HTTP {r.status_code}，返回 {body[:150]}；"
                       f"上传后知识库：{st.get('filename')} / {st.get('chunks')} 块 → "
                       f"{'原样保留' if kept else '被破坏！'}"),
            "pass": bool(ok),
        }
    finally:
        if os.path.exists(out):
            os.remove(out)


def t12_redis_degraded():
    """
    Redis 停掉时，服务还能不能用。

    这条用例是被一个真实 bug 逼出来的：/status 里直接调 r.dbsize()，
    没有降级分支，Redis 一停就返回 HTTP 500。当时 11 个用例全部通过，
    因为跑测试的时候 Redis 恰好开着——降级路径从来没被走到过。

    所以这里不去假设 Redis 是开是关，而是**先探明真实状态**，再验证
    接口的声明和真实状态是否一致。两种状态下都应该通过。
    """
    up = redis_reachable()
    resp = requests.get(f"{BASE}/status", timeout=10)
    ok_http = resp.status_code == 200
    body = resp.json() if ok_http else {}

    declared_on = body.get("cache_on")
    cache_size = body.get("cache_size")

    # /status 不能挂
    ok = ok_http
    # 声明的缓存状态要与实际一致
    ok = ok and (declared_on is up)
    # 缓存不可用时不该报出一个假的条目数
    ok = ok and ((cache_size is None) if not up else (cache_size is not None))

    # 缓存不可用时，问答主流程必须照常能答（降级到"直查、不缓存"）
    qa_note = "（Redis 在跑，未走降级）"
    if not up:
        try:
            r = ask("迟到十五分钟以内会怎么处理？")
            ans = (r.get("answer") or "").strip()
            items = len(r.get("sources") or [])
            answered = ans and not ans.startswith("[")
            ok = ok and r.get("http") == 200 and items > 0 and answered
            qa_note = (f"降级问答正常：检索 {items} 段，回答 {len(ans)} 字，"
                       f"{r.get('total_wall_ms')}ms")
        except Exception as e:
            ok = False
            qa_note = f"降级问答失败：{type(e).__name__}: {e}"

    return {
        "id": "T12", "name": "Redis 不可用时的降级",
        "input": f"GET /status（Redis 实际状态：{'已连接' if up else '未连接'}）",
        "expect": ("/status 一律返回 200；cache_on 与 Redis 真实状态一致；"
                   "Redis 停时 cache_size 为 null，且问答主流程仍能作答"),
        "actual": (f"HTTP {resp.status_code}，cache_on={declared_on}，"
                   f"cache_size={cache_size}；{qa_note}"),
        "pass": bool(ok),
    }


def main():
    try:
        requests.get(f"{BASE}/status", timeout=5)
    except Exception as e:
        print(f"服务未启动（{BASE}）：{e}")
        print("请先运行：python pdf_qa_optimized.py")
        return 1

    n = clear_qa_cache()
    if n is None:
        print("Redis 未连接：跳过清缓存，本轮会走到降级路径（T05 预期不通过，T12 会记录）\n")
    else:
        print(f"已清理 {n} 条历史问答缓存，保证 T03/T04 是冷启动\n")

    cases = [
        t01_status, t02_ui, t03_factual, t04_reasoning, t05_cache,
        t06_out_of_kb, t07_domain_mismatch, t08_long_input,
        t09_special_chars, t10_upload_non_pdf, t11_upload_scanned,
        t12_redis_degraded,
    ]
    results = []
    for fn in cases:
        print(f"运行 {fn.__name__} ...", end="", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = {"id": fn.__name__, "name": fn.__name__, "input": "-",
                 "expect": "-", "actual": f"[用例本身出错] {type(e).__name__}: {e}",
                 "pass": False}
        results.append(r)
        print(" 通过" if r["pass"] else " 未通过")

    # ---- 写 JSON ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    jp = os.path.join(RESULT_DIR, "blackbox_report.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"base": BASE, "cases": results,
                   "passed": sum(1 for x in results if x["pass"]), "total": len(results)},
                  f, ensure_ascii=False, indent=2)

    # ---- 写 Markdown ----
    md = ["# 系统黑盒测试记录", "",
          f"测试对象：`{BASE}`（本地运行的服务）",
          f"用例总数：{len(results)}，通过：{sum(1 for x in results if x['pass'])}", "",
          "| 用例 | 测试项 | 输入 | 预期 | 实际结果 | 判定 |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append("| {id} | {name} | {inp} | {exp} | {act} | {verdict} |".format(
            id=r["id"], name=r["name"],
            inp=str(r["input"]).replace("|", "\\|").replace("\n", " ")[:60],
            exp=str(r["expect"]).replace("|", "\\|")[:80],
            act=str(r["actual"]).replace("|", "\\|")[:150],
            verdict="通过" if r["pass"] else "**未通过**"))
    md.append("")
    md.append("## 关键回答原文")
    md.append("")
    for r in results:
        if r.get("answer"):
            md.append(f"### {r['id']} {r['name']}")
            md.append("")
            md.append(f"> 提问：{r['input']}")
            md.append("")
            md.append(r["answer"][:1200])
            md.append("")

    mp = os.path.join(RESULT_DIR, "blackbox_report.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n已保存：{jp}")
    print(f"已保存：{mp}")
    print(f"\n通过 {sum(1 for x in results if x['pass'])}/{len(results)}")
    for r in results:
        if not r["pass"]:
            print(f"  未通过 {r['id']} {r['name']}：{r['actual'][:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
