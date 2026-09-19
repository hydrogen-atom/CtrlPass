# 做题状态动态管理 · 第一版

已按四步落地：六张表和独立登录、资料与题目存储、作答状态流程、前端接入与验证。

## 启动

在 `D:\CtrlPass\RAG_finsh_fundamental_function\CtrlPass` 运行：

```powershell
python backend/app.py
```

另开终端进入 `frontend`，运行 `npm run dev`，访问 `http://localhost:5173`，在页面注册独立账号。
现有 WebForms 账号没有接入；历史 `data` / `vector_stores` 文件保持原样，没有自动分配给任何新用户。

数据库自动初始化到 `backend/storage/ctrlpass.sqlite3`，即方案指定的绝对位置。第一次启动不创建业务数据或默认账号。
已有依赖继续使用 `backend/requirements.txt` 和 `frontend/package.json`；新增持久化使用 Python 自带的 SQLite。
真实资料处理、出题、提示、主观题判分继续使用 Moonshot 接口，需要项目 `.env` 配置 `MOONSHOT_API_KEY`。
向量处理沿用现有本地嵌入模型，其首次使用仍可能需要下载模型文件。

## 文件分工

| 文件 | 负责内容 |
| --- | --- |
| `schema.sql` | 六张业务表、外键、唯一约束、知识点关联和快照不可覆盖约束 |
| `database.py` | SQLite 连接、UTC 时间、事务、数据库初始化 |
| `practice.py` | 登录、所属用户验证、资料、题目、计时、提交、提示、错因接口 |
| `learning_provider.py` | 调用现有文档、向量和模型组件 |
| `app.py` | Flask 启动和原有问答接口的账号适配 |
| `frontend/src/components/PracticeApp.tsx` | 登录、资料、做题、提示、历史、错因确认页面 |
| `tests/test_practice_lifecycle.py` | 使用真实 SQLite 和固定模型返回值的离线验收测试 |

六张业务表主键都是自增整数；SQLite 自带的 `sqlite_sequence` 属于内部表。
用户、资料、题目、作答之间采用外键；复合外键同时约束所属用户。提示、错因通过所属作答核验权限。
知识点暂存 JSON，编号由服务端分配；修改名称接口只改名称，不改变编号。
数据库字符串时间统一包含 `Z`，前端通过 `toLocaleString()` 转换为浏览器本地时间。布尔值用 `0/1`。

## 状态取值

数据库和接口使用稳定的英文值，页面显示中文：

| 对象 | 状态值 |
| --- | --- |
| 账号 | `active` 正常、`disabled` 停用 |
| 资料 | `uploaded` 已上传、`processing` 处理中、`ready` 可使用、`failed` 处理失败 |
| 题目 | `ready` 可使用、`expired` 内容过期、`invalid` 题目无效 |
| 作答 | `in_progress` 作答中、`pending_grading` 待判分、`completed` 已完成、`abandoned` 已放弃、`expired` 题目过期、`grading_failed` 判分失败 |
| 提示 | `requested` 请求中、`generated` 已生成、`viewed` 已展示、`failed` 失败 |
| 错因确认 | `pending` 待确认、`confirmed` 已确认、`rejected` 已否定 |
| 可信程度 | `high` 高、`medium` 中、`low` 低、`unknown` 未知 |

题型：`single_choice`、`multiple_choice`、`fill_blank`、`short_answer`。
提示类型：`concept`、`approach`、`step`、`answer`。
错因类型：`concept_missing`、`confusion`、`method`、`calculation`、`reading`、`expression`、`unknown`。
错因来源：`user`、`rule`、`model`。选择题错误只记录“具体原因未知”，不会据选项直接断定认知缺陷。

## 接口与请求

浏览器通过 Flask 签名 Cookie 保持登录。所有修改请求必须带 `X-CtrlPass-Request: 1`；跨域仅允许配置的前端来源。
旧 `X-Session-ID` 不再用作身份凭证。注册成功不算成功登录；只有 `/auth/login` 成功才更新 `last_login_at`。

| 方法 | 路径 | 请求内容 |
| --- | --- | --- |
| POST | `/api/auth/register` | `username,password,display_name?`，密码 8～256 字符 |
| POST | `/api/auth/login` | `username,password` |
| POST | `/api/auth/logout` | `{}` |
| GET | `/api/auth/me` | 当前账号 |
| POST | `/api/materials/upload` | 表单文件字段 `file`，TXT/PDF/DOCX，最大 32 MB |
| GET | `/api/materials` | 当前用户资料列表 |
| POST | `/api/materials/{id}/process` | `chunk_size?,chunk_overlap?,use_model_splitter?` |
| PATCH | `/api/materials/{id}/knowledge-points/{point_id}` | `name` |
| POST | `/api/questions/generate` | `material_id,primary_knowledge_point_id,question_type?,difficulty_level?,goal?` |
| GET | `/api/questions/{id}` | 题干与选项，不返回标准答案或评分标准 |
| POST | `/api/questions/{id}/attempts` | `{is_page_hidden:0}`，已有进行中作答时返回该记录 |
| PATCH | `/api/attempts/{id}/progress` | 下方进度对象 |
| POST | `/api/attempts/{id}/hints` | `request_key,hint_type?,hint_level?,request_text?,progress?` |
| POST | `/api/hints/{id}/viewed` | `{}`，前端实际展示后确认 |
| POST | `/api/attempts/{id}/submit` | `submission_key,answer,progress?` |
| POST | `/api/attempts/{id}/retry-grading` | `{}` |
| POST | `/api/attempts/{id}/abandon` | `{progress?:进度对象}` |
| GET | `/api/attempts?limit=50&offset=0` | 按开始时间倒序，单页最多 100 条 |
| GET | `/api/attempts/{id}` | 作答、题目、提示摘要和错因 |
| POST | `/api/attempts/{id}/errors` | `description,evidence_summary,knowledge_point_id?,error_type?`，新增用户自述 |
| PATCH | `/api/errors/{id}` | `review_status`，保留原来的推断内容与来源 |

资料上传和处理仅通过 `/api/materials` 接口完成，服务端只会访问当前用户拥有的资料记录。

进度对象示例：

```json
{"progress_seq": 1, "active_duration_ms": 15000, "is_manually_paused": 0, "is_page_hidden": 0}
```

单选答案为选项编号字符串，如 `"A"`；多选为编号列表，如 `["A","C"]`；填空和简答为非空字符串。
防重标识为前端随机字符串；提交标识在同一用户内唯一，提示标识在同一次作答内唯一。
重复提交同一答案和标识返回已有记录；同一标识用于不同答案或同一作答用不同标识覆盖答案，返回 `409`。

## 保存、计时和失败恢复

- 资料原文件存到 `storage/materials/{资料编号}/source.*`。向量存到该资料目录下；处理批次的随机键仅防止中断重试相互覆盖，不是业务编号。
- 未提交题目的完整内容暂存 `storage/cache/questions/{题目编号}.json`，默认有效期 24 小时；数据库只保存摘要。草稿不入库，刷新或离开作答页会丢失草稿。
- 第一次有效提交在一个事务中保存原题快照、答案和待判分状态。事务提交后才调用判分。选择题按选项比对；填空和简答由模型按评分标准判分。
- 判分异常或返回格式不合格时，状态为 `grading_failed`，正确性和分数保持空值，不记为答错。重试引用同一份快照；旧请求返回不能覆盖新一次判分。
- 重做新建作答记录并引用原快照；快照不能被更新或清空。原题失效不能通过重新生成冒充原题。
- 前端每 15 秒同步累计有效时间，暂停/隐藏/恢复时立即同步，提交和放弃携带最后进度。序号小于或等于已保存序号的进度被忽略。
- 后端拒绝倒退时间、暂停或隐藏期间新增用时，以及超过服务器经过时间加 2 秒容差的增长；单次同步最多补 120 秒。断网、电脑休眠或突然关闭页面可能丢失末尾未同步时间，恢复时以服务器进度为准。
- 提示展示确认后才计入依赖次数；失败不计入，查看完整答案另行统计。提示或判分请求卡住超过 3 分钟可重试；资料处理卡住超过 30 分钟可重试。
- 历史出题上下文保存最近十次完成记录的摘要、提示依赖和错因；为控制模型输入长度，当前模型实际读取其中最近三次的压缩摘要。已否定的错因不再用于新题；本版采用用户选择的预计难度，历史用于考查方式，不根据低可信度推断大幅调整难度。

新出题和历史查询会清理到期缓存；无访问期间可手动运行：

```powershell
python -m flask --app backend.app cleanup-question-cache
```

本版未新增练习分组表、统一知识点表、长期长解析或草稿表。

## 验证

```powershell
python -m pytest tests/test_practice_lifecycle.py tests/test_placeholder_protection.py test_document_processor.py -q
cd frontend
npm run build
node node_modules/eslint/bin/eslint.js src/components/PracticeApp.tsx src/api/practice.ts
```

自动化测试使用临时数据库与固定模型返回值，覆盖所属用户校验、事务回滚、并发防重、计时、缓存到期、判分与提示失败恢复、知识点改名、错因自述和确认。
本次新增 28 项状态管理测试，连同现有相关回归共 54 项通过；前端构建和新增前端文件的 ESLint 检查通过。现有问答测试的模拟对象已适配项目当前的检索参数和模型响应格式。
页面操作验证也使用独立临时数据库与本地模拟模型，已检查登录、出题、提示、暂停/继续、提交、历史统计、重做和刷新恢复。
真实模型服务、真实资料向量处理尚未做外部联调；这不影响上述离线流程验证，但需在配置有效模型密钥后验收。
