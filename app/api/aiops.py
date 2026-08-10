"""
AIOps 智能运维接口
"""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service
from app.observability import observability_store, metrics_collector

router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 自动获取当前系统的活动告警
    - 使用 Plan-Execute-Replan 模式进行智能诊断
    - 流式返回诊断过程和结果

    **SSE 事件类型：**

    1. `status` - 状态更新
       ```json
       {
         "type": "status",
         "stage": "fetching_alerts",
         "message": "正在获取系统告警信息..."
       }
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {
         "type": "plan",
         "stage": "plan_created",
         "message": "诊断计划已制定，共 6 个步骤",
         "target_alert": {...},
         "plan": ["步骤1: ...", "步骤2: ..."]
       }
       ```

    3. `step_complete` - 步骤执行完成
       ```json
       {
         "type": "step_complete",
         "stage": "step_executed",
         "message": "步骤执行完成 (2/6)",
         "current_step": "查询系统日志",
         "result_preview": "...",
         "remaining_steps": 4
       }
       ```

    4. `report` - 最终诊断报告
       ```json
       {
         "type": "report",
         "stage": "final_report",
         "message": "最终诊断报告已生成",
         "report": "# 故障诊断报告\\n...",
         "evidence": {...}
       }
       ```

    5. `complete` - 诊断完成
       ```json
       {
         "type": "complete",
         "stage": "diagnosis_complete",
         "message": "诊断流程完成",
         "diagnosis": {...}
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "诊断过程发生错误: ..."
       }
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    **前端使用示例：**
    ```javascript
    const eventSource = new EventSource('/api/aiops');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'plan') {
        console.log('诊断计划:', data.plan);
      } else if (data.type === 'step_complete') {
        console.log('步骤完成:', data.current_step);
      } else if (data.type === 'report') {
        console.log('最终报告:', data.report);
      } else if (data.type === 'complete') {
        console.log('诊断完成');
        eventSource.close();
      }
    };
    ```

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        try:
            async for event in aiops_service.diagnose(session_id=session_id):
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@router.get("/aiops/metrics")
async def get_metrics():
    """获取可观测聚合指标

    返回工具调用统计（成功率/延迟/Token）+ 节点执行统计 + 最近 trace 列表。
    用于运维仪表盘和面试演示。
    """
    try:
        summary = await metrics_collector.get_summary()
        return {"status": "success", "data": summary}
    except Exception as e:
        logger.error(f"获取指标失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/aiops/traces/{trace_id}")
async def get_trace(trace_id: str):
    """查询单条 trace 的完整链路（含 spans + tool_metrics）

    用于全链路复现：输入什么 → planner 生成什么计划 → executor 调了哪些工具 →
    replanner 做了什么决策 → memory_writer 写了什么 → 最终响应。
    """
    try:
        trace = await observability_store.get_trace(trace_id)
        if trace is None:
            return {"status": "not_found", "message": f"trace_id {trace_id} 不存在"}
        return {"status": "success", "data": trace}
    except Exception as e:
        logger.error(f"查询 trace 失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/aiops/traces")
async def list_traces(limit: int = 20):
    """列出最近的 trace（不含 spans 明细）"""
    try:
        traces = await observability_store.list_traces(limit=limit)
        return {"status": "success", "data": traces, "count": len(traces)}
    except Exception as e:
        logger.error(f"列出 trace 失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
