/* PDF 文档问答系统 —— 前端逻辑
 *
 * 页面由 FastAPI 后端直接托管（同源），因此接口地址用相对路径，
 * 不需要写死 127.0.0.1:18005。
 *
 * 与 /ask/v2 的通信采用 SSE（Server-Sent Events）：
 * 因为 EventSource 只支持 GET，而提问需要 POST 传 JSON，
 * 所以这里用 fetch 读到响应流后手动按 "\n\n" 拆事件解析。
 */

const API = '';

const $ = (id) => document.getElementById(id);
const messagesEl = $('messages');
const convListEl = $('convList');
const inputEl    = $('input');
const btnSend    = $('btnSend');

let convId = null;      // 当前会话 id
let sending = false;    // 是否正在等待回答

/* ---------------- 初始化 ---------------- */

async function init() {
  await Promise.all([loadStatus(), loadConversations()]);
}

async function loadStatus() {
  try {
    const r = await fetch(`${API}/kb/status`);
    const d = await r.json();
    setKbStatus(d.loaded, d.filename, d.chunks);
  } catch (e) {
    setKbStatus(false, '', 0, '无法连接后端服务');
  }
}

function setKbStatus(loaded, filename, chunks, errMsg) {
  const box = $('kbStatus');
  const txt = $('kbText');
  box.classList.remove('on', 'off');
  if (errMsg) {
    box.classList.add('off');
    txt.textContent = errMsg;
    return;
  }
  box.classList.add(loaded ? 'on' : 'off');
  txt.textContent = loaded
    ? `知识库就绪 · ${filename} · ${chunks} 块`
    : '知识库为空，请先上传 PDF';
  $('infoFile').textContent   = loaded ? filename : '—';
  $('infoChunks').textContent = loaded ? chunks : '—';
}

async function loadConversations() {
  try {
    const r = await fetch(`${API}/conversations`);
    const d = await r.json();
    renderConversations(d.conversations || []);
  } catch (e) { /* 后端没起来时静默 */ }
}

function renderConversations(list) {
  convListEl.innerHTML = '';
  if (!list.length) {
    convListEl.innerHTML = '<li class="empty">暂无历史会话</li>';
    return;
  }
  list.forEach((c) => {
    const li = document.createElement('li');
    if (c.id === convId) li.classList.add('active');

    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = c.title || '新对话';
    title.onclick = () => openConversation(c.id);

    const del = document.createElement('span');
    del.className = 'del';
    del.textContent = '×';
    del.title = '删除该会话';
    del.onclick = (e) => { e.stopPropagation(); removeConversation(c.id); };

    li.append(title, del);
    convListEl.appendChild(li);
  });
}

/* ---------------- 上传 PDF ---------------- */

$('uploadBox').onclick = () => {
  if ($('uploadBox').classList.contains('busy')) return;
  $('fileInput').click();
};

$('fileInput').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const box = $('uploadBox');
  box.classList.add('busy');
  box.querySelector('p').textContent = '正在解析并建立索引…';

  const form = new FormData();
  form.append('file', file);

  try {
    const r = await fetch(`${API}/upload`, { method: 'POST', body: form });
    const d = await r.json();
    setKbStatus(true, d.filename, d.chunks);
    addSystemNote(`已上传《${d.filename}》：共 ${d.pages} 页，切分为 ${d.chunks} 个文本块。`);
  } catch (err) {
    addSystemNote('上传失败：' + err.message);
  } finally {
    box.classList.remove('busy');
    box.querySelector('p').textContent = '点击上传 PDF';
    e.target.value = '';
  }
};

/* ---------------- 发送提问 ---------------- */

btnSend.onclick = send;

inputEl.onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
};

inputEl.oninput = () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
};

document.querySelectorAll('.sample').forEach((el) => {
  el.onclick = () => { inputEl.value = el.textContent; send(); };
});

async function send() {
  const question = inputEl.value.trim();
  if (!question || sending) return;

  sending = true;
  btnSend.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  const welcome = $('welcome');
  if (welcome) welcome.remove();

  addUserMessage(question);
  const bot = addBotMessage();

  const t0 = performance.now();
  try {
    const resp = await fetch(`${API}/ask/v2`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation_id: convId }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 事件之间用空行分隔
      const parts = buffer.split('\n\n');
      buffer = parts.pop();            // 最后一段可能不完整，留到下一轮

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        let ev;
        try { ev = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
        handleEvent(ev, bot, t0);
      }
    }
  } catch (err) {
    appendAnswer(bot, `\n\n[请求失败] ${err.message}`);
  } finally {
    bot.bubble.classList.remove('streaming');
    sending = false;
    btnSend.disabled = false;
    inputEl.focus();
    loadConversations();
  }
}

function handleEvent(ev, bot, t0) {
  if (ev.type === 'meta') {
    if (ev.conv_id) convId = ev.conv_id;

  } else if (ev.type === 'sources') {
    renderSources(bot, ev);
    // 检索耗时是挂在 sources 上的，done 只带生成耗时，
    // 先攒进 bot.timing，等 done 到了再一起显示
    Object.assign(bot.timing, ev.t || {});

  } else if (ev.type === 'delta') {
    appendAnswer(bot, ev.text);
    messagesEl.scrollTop = messagesEl.scrollHeight;

  } else if (ev.type === 'done') {
    Object.assign(bot.timing, ev.t || {});
    renderTiming(bot, bot.timing, t0);

  } else if (ev.type === 'error') {
    bot.raw = ev.message;
    bot.text.innerHTML = mdToHtml(bot.raw);
  }
}

/* ---------------- 渲染 ---------------- */

/* 大模型返回的是 Markdown，直接塞进 textContent 会把 ** 和 - 这些记号
   原样显示出来。这里做一层极简渲染，只支持实际会用到的那几种语法。

   注意：一旦改用 innerHTML，文本里的 < > 就会被当成标签执行，
   所以必须先转义再拼标签 —— 顺序反了就是个 XSS 口子。 */
function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* 行内语法：**粗体**、`代码` */
function inlineMd(s) {
  return esc(s)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

/* 整段渲染：段落 / - 无序列表 / 1. 有序列表 / ### 标题 / 空行分段。
   流式输出时每来一小段就整体重渲染一次，所以渲染到一半的 ** 或没闭合的
   <ul> 都是正常的，下一段来了自然会补上。 */
function mdToHtml(md) {
  const html = [];
  let para = [];
  let list = null;

  const endPara = () => {
    if (para.length) { html.push('<p>' + para.join('<br>') + '</p>'); para = []; }
  };
  const endList = () => {
    if (list) { html.push('</' + list + '>'); list = null; }
  };
  const startList = (kind) => {
    if (list === kind) return;
    endList(); endPara();
    html.push('<' + kind + '>'); list = kind;
  };

  for (const raw of md.split('\n')) {
    const line = raw.trim();

    if (!line) { endPara(); endList(); continue; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      endPara(); endList();
      const lv = Math.min(h[1].length + 2, 4);     // # → h3，## 及以上 → h4
      html.push(`<h${lv}>${inlineMd(h[2])}</h${lv}>`);
      continue;
    }

    const ul = line.match(/^[-*+]\s+(.*)$/);
    if (ul) { startList('ul'); html.push('<li>' + inlineMd(ul[1]) + '</li>'); continue; }

    const ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ol) { startList('ol'); html.push('<li>' + inlineMd(ol[1]) + '</li>'); continue; }

    endList();
    para.push(inlineMd(line));
  }

  endPara(); endList();
  return html.join('');
}

/* 累积原始 Markdown，每来一段就整体重渲染。
   每次重算整段是 O(n²)，但回答只有几百字，肉眼无感；
   换来的是不用处理"半个 ** 怎么渲染"这种麻烦事。 */
function appendAnswer(bot, text) {
  bot.raw += text;
  bot.text.innerHTML = mdToHtml(bot.raw);
}

function addUserMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg user';
  wrap.innerHTML = '<div class="who">我</div>';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  wrap.appendChild(b);
  messagesEl.appendChild(wrap);
  scrollDown();
}

function addBotMessage() {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  wrap.innerHTML = '<div class="who">助手</div>';

  const bubble = document.createElement('div');
  // streaming 类负责在最后一个元素的行尾画闪烁光标（见 style.css）
  bubble.className = 'bubble streaming';
  // 用 div 不用 span：渲染出来的 <p>/<ul> 是块级元素，塞进 span 里不合法
  const text = document.createElement('div');
  text.className = 'md';
  bubble.appendChild(text);

  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollDown();
  return { wrap, text, bubble, raw: '', timing: {}, cached: false };
}

function renderSources(bot, ev) {
  bot.cached = !!ev.cached;    // 用后端给的标记，不去猜耗时是不是 0
  if (ev.cached) return;

  const det = document.createElement('details');
  det.className = 'sources';
  const sum = document.createElement('summary');
  sum.textContent = `检索依据（向量检索最相关的 ${ev.items.length} 段）`;
  det.appendChild(sum);

  ev.items.forEach((it, i) => {
    const div = document.createElement('div');
    div.className = 'src-item';
    const sc = document.createElement('div');
    sc.className = 'score';
    sc.textContent = `第 ${i + 1} 段 · 相关度 ${it.score.toFixed(4)}`;
    div.appendChild(sc);
    div.append(document.createTextNode(it.text));
    det.appendChild(div);
  });

  bot.wrap.appendChild(det);
  scrollDown();
}

function renderTiming(bot, t, t0) {
  const box = document.createElement('div');
  box.className = 'timing';

  const total = Math.round(performance.now() - t0);
  const add = (label, val) => {
    const s = document.createElement('span');
    s.className = 'tag';
    s.textContent = `${label} ${val} ms`;
    box.appendChild(s);
  };

  if (bot.cached) {
    const s = document.createElement('span');
    s.className = 'tag cache';
    s.textContent = '命中 Redis 缓存';
    box.appendChild(s);
  } else if (t.retrieve_ms !== undefined) {
    add('向量检索', t.retrieve_ms);
  }
  if (t.gen_ms !== undefined) add('模型生成', t.gen_ms);
  add('端到端', total);

  bot.wrap.appendChild(box);
  scrollDown();
}

function addSystemNote(text) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  div.appendChild(b);
  messagesEl.appendChild(div);
  scrollDown();
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------------- 会话管理 ---------------- */

async function openConversation(id) {
  convId = id;
  try {
    const r = await fetch(`${API}/conversation/${id}`);
    const d = await r.json();
    const welcome = $('welcome');
    if (welcome) welcome.remove();
    messagesEl.innerHTML = '';
    (d.messages || []).forEach((m) => {
      if (m.role === 'user') {
        addUserMessage(m.content);
      } else {
        const bot = addBotMessage();
        bot.raw = m.content;
        bot.text.innerHTML = mdToHtml(bot.raw);
        bot.bubble.classList.remove('streaming');   // 历史消息不显示光标
      }
    });
    loadConversations();
  } catch (e) { /* ignore */ }
}

async function removeConversation(id) {
  if (!confirm('确定删除这条会话记录？')) return;
  await fetch(`${API}/conversation/${id}`, { method: 'DELETE' });
  if (convId === id) newConversation();
  loadConversations();
}

function newConversation() {
  convId = null;
  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>开始提问</h2>
      <p>系统会先从 PDF 中找到与问题相关的段落，再用大模型据此作答。</p>
      <div class="samples">
        <span class="sample">年假有多少天？</span>
        <span class="sample">病假期间工资怎么算？</span>
        <span class="sample">加班费怎么计算？</span>
        <span class="sample">迟到会怎么处理？</span>
      </div>
    </div>`;
  document.querySelectorAll('.sample').forEach((el) => {
    el.onclick = () => { inputEl.value = el.textContent; send(); };
  });
  loadConversations();
}

$('btnNew').onclick = newConversation;

init();
