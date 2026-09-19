# CtrlPass

基于 LangGraph 的学习助手 Agent。它可以处理学习资料、检索原文、回答资料问题、生成知识图谱、生成练习、提供提示、判分并记录错因。

## Agent 循环

自动模式通过 `POST /api/agent/run` 执行一个有上限的循环：

1. 观察资料、知识点和近期学习记录；
2. 由模型决定下一步动作；
3. 调用白名单工具（分析学习历史、生成练习题、资料问答或生成知识图谱）；
4. 把工具结果作为新观察，再次决策；
5. 目标结果生成并被再次观察到后结束。

页面中的“让 Agent 自主安排”会启动该流程，可以选择自动判断、自主出题、资料问答或知识图谱。问答和图谱均返回原文片段编号；图谱结果包含节点、关系和可视化。模型只能选择已注册工具，不能执行任意代码；循环默认最多调用 4 次工具。

相关接口：

- `POST /api/agent/run`：统一 Agent 入口，参数为 `material_id、task?、goal?`。
- `POST /api/materials/{id}/ask`：直接资料问答，参数为 `question`。
- `POST /api/materials/{id}/knowledge-graph`：直接生成知识图谱，参数为 `focus?`。

## 启动

后端需要在项目 `.env` 中配置 `MOONSHOT_API_KEY`、`DASHSCOPE_API_KEY` 或 `QWEN_API_KEY`：

```powershell
python backend/app.py
```

另开终端启动前端：

```powershell
cd frontend
npm run dev
```
