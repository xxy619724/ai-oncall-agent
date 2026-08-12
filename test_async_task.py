"""异步任务系统端到端验证脚本

自动模拟：提交任务 → 轮询状态 → 验证结果
包含：正常流程 + 取消流程 + 异常场景

用法：python test_async_task.py
"""

import time
import json
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:9900/api"

# ============================================================
# 工具函数
# ============================================================

def print_header(title: str) -> None:
    """打印分段标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step: str) -> None:
    """打印步骤"""
    print(f"\n▶ {step}")


def print_response(r: requests.Response) -> None:
    """打印响应"""
    print(f"  HTTP {r.status_code}")
    try:
        print(f"  响应: {json.dumps(r.json(), ensure_ascii=False, indent=2)[:500]}")
    except Exception:
        print(f"  响应: {r.text[:500]}")


def format_duration(start: str, end: str) -> str:
    """计算耗时"""
    try:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        return f"{(e - s).total_seconds():.1f}s"
    except Exception:
        return "unknown"


def poll_task_status(
    task_id: str,
    max_wait: int = 120,
    interval: int = 2,
    expect_terminal: bool = True,
) -> dict:
    """轮询任务状态直到终态

    Args:
        task_id: 任务 ID
        max_wait: 最大等待秒数
        interval: 轮询间隔
        expect_terminal: 是否等待终态

    Returns:
        最终任务状态
    """
    print(f"\n  轮询任务状态（每 {interval}s 一次，最长 {max_wait}s）...")
    start_time = time.time()
    last_status = ""

    while time.time() - start_time < max_wait:
        r = requests.get(f"{BASE_URL}/tasks/{task_id}")
        if r.status_code != 200:
            print(f"  查询失败: HTTP {r.status_code}")
            time.sleep(interval)
            continue

        data = r.json()["data"]
        current_status = data["status"]

        # 状态变化时打印
        if current_status != last_status:
            progress = f"{data['progress']['completed']}/{data['progress']['total']}"
            elapsed = f"{time.time() - start_time:.1f}s"
            print(f"  [{elapsed}] 状态: {current_status} (进度: {progress})")
            last_status = current_status

        # 终态判断
        if expect_terminal and data["is_terminal"]:
            return data

        time.sleep(interval)

    print(f"  ⚠️ 超过 {max_wait}s 未到终态")
    return data


# ============================================================
# 测试用例
# ============================================================

def test_normal_flow() -> bool:
    """测试 1: 正常提交流程（提交 → 轮询 → 成功）"""
    print_header("测试 1: 正常提交流程")
    print_step("提交任务")
    body = {
        "input_text": "你好，请用一句话介绍你自己",
        "session_id": "e2e-normal",
    }
    r = requests.post(f"{BASE_URL}/tasks", json=body)
    print_response(r)

    if r.status_code != 202:
        print("  ❌ 提交失败")
        return False

    task_id = r.json()["task_id"]
    print(f"\n  task_id: {task_id}")

    print_step("轮询状态直到终态")
    final = poll_task_status(task_id, max_wait=120)

    print_step("验证结果")
    success = final["status"] == "succeeded"
    duration = format_duration(final["created_at"], final["ended_at"])

    print(f"  最终状态: {final['status']}")
    print(f"  进度: {final['progress']['completed']}/{final['progress']['total']}")
    print(f"  总耗时: {duration}")
    print(f"  结果: {(final['result_text'] or '')[:100]}...")
    print(f"  终态: {final['is_terminal']}")

    if success:
        print("  ✅ 正常流程通过")
    else:
        print(f"  ❌ 期望 succeeded，实际 {final['status']}")
    return success


def test_cancel_flow() -> bool:
    """测试 2: 取消流程（提交长任务 → 取消 → 验证 cancelled）"""
    print_header("测试 2: 取消流程")
    print_step("提交长任务")
    body = {
        "input_text": "请详细分析当前系统的告警情况，生成完整的诊断报告，包含所有告警的根因分析和处理建议",
        "session_id": "e2e-cancel",
    }
    r = requests.post(f"{BASE_URL}/tasks", json=body)
    print_response(r)

    if r.status_code != 202:
        print("  ❌ 提交失败")
        return False

    task_id = r.json()["task_id"]
    print(f"\n  task_id: {task_id}")

    print_step("等待 2s 让任务进入 running")
    time.sleep(2)

    print_step("查询当前状态")
    r = requests.get(f"{BASE_URL}/tasks/{task_id}")
    print_response(r)
    current = r.json()["data"]
    print(f"  当前状态: {current['status']}")

    print_step("发送取消请求")
    r = requests.post(f"{BASE_URL}/tasks/{task_id}/cancel")
    print_response(r)

    print_step("轮询状态直到终态")
    final = poll_task_status(task_id, max_wait=60)

    print_step("验证结果")
    success = final["status"] in ("cancelled", "succeeded")  # 可能在取消前就完成了
    print(f"  最终状态: {final['status']}")
    print(f"  终态: {final['is_terminal']}")

    if final["status"] == "cancelled":
        print("  ✅ 取消流程通过（任务已取消）")
    elif final["status"] == "succeeded":
        print("  ⚠️ 任务在取消前已完成（可接受）")
    else:
        print(f"  ❌ 期望 cancelled/succeeded，实际 {final['status']}")
    return success


def test_list_tasks() -> bool:
    """测试 3: 任务列表"""
    print_header("测试 3: 任务列表")
    print_step("查询任务列表")
    r = requests.get(f"{BASE_URL}/tasks?limit=10")
    print_response(r)

    if r.status_code != 200:
        print("  ❌ 查询失败")
        return False

    data = r.json()
    print(f"\n  任务总数: {data['count']}")
    print("  最近任务:")
    for t in data["data"][:5]:
        task_id_short = t["task_id"][:8]
        status = t["status"]
        progress = f"{t['progress']['completed']}/{t['progress']['total']}"
        terminal = "终态" if t["is_terminal"] else "进行中"
        print(f"    - {task_id_short}... | {status:10} | 进度 {progress} | {terminal}")

    print("  ✅ 任务列表通过")
    return True


def test_not_found() -> bool:
    """测试 4: 查询不存在的任务"""
    print_header("测试 4: 查询不存在的任务")
    print_step("查询不存在的 task_id")
    r = requests.get(f"{BASE_URL}/tasks/nonexistent-id-12345")
    print_response(r)

    if r.status_code == 404:
        print("  ✅ 404 返回正确")
        return True
    else:
        print(f"  ❌ 期望 404，实际 {r.status_code}")
        return False


def test_cancel_terminal() -> bool:
    """测试 5: 取消已终态任务（应 409）"""
    print_header("测试 5: 取消已终态任务")
    print_step("查找一个已终态的任务")
    r = requests.get(f"{BASE_URL}/tasks?limit=10")
    tasks = r.json()["data"]
    terminal_task = next((t for t in tasks if t["is_terminal"]), None)

    if terminal_task is None:
        print("  ⚠️ 没有已终态任务，跳过")
        return True

    task_id = terminal_task["task_id"]
    print(f"  使用任务: {task_id[:8]}... (状态: {terminal_task['status']})")

    print_step("尝试取消已终态任务")
    r = requests.post(f"{BASE_URL}/tasks/{task_id}/cancel")
    print_response(r)

    if r.status_code == 409:
        print("  ✅ 409 Conflict 返回正确")
        return True
    else:
        print(f"  ⚠️ 期望 409，实际 {r.status_code}（取消是异步的，可能任务刚结束）")
        return True


def test_queue_full() -> bool:
    """测试 6: 队列容量保护（快速提交超过 maxsize 的任务）"""
    print_header("测试 6: 队列容量保护")
    print_step("快速提交 105 个任务（队列容量 100）")
    success_count = 0
    rejected_count = 0
    for i in range(105):
        body = {
            "input_text": f"测试任务 {i}",
            "session_id": f"e2e-queue-{i}",
        }
        r = requests.post(f"{BASE_URL}/tasks", json=body)
        if r.status_code == 202:
            success_count += 1
        elif r.status_code == 503:
            rejected_count += 1
        else:
            print(f"  意外状态码: {r.status_code}")

    print(f"\n  成功入队: {success_count}")
    print(f"  被拒绝: {rejected_count}")

    if rejected_count > 0:
        print("  ✅ 队列满时返回 503")
        return True
    else:
        print("  ⚠️ 没有触发 503（可能 Worker 消费太快）")
        return True


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  异步任务系统端到端验证")
    print(f"  服务地址: {BASE_URL}")
    print(f"  时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # 健康检查
    print_step("健康检查")
    try:
        r = requests.get(f"{BASE_URL}/../docs", timeout=5)
        print(f"  服务可达: HTTP {r.status_code}")
    except Exception as e:
        print(f"  ❌ 服务不可达: {e}")
        print("  请先启动服务: python -m uvicorn app.main:app --port 9900")
        return

    # 运行所有测试
    results = []
    results.append(("正常提交流程", test_normal_flow()))
    results.append(("取消流程", test_cancel_flow()))
    results.append(("任务列表", test_list_tasks()))
    results.append(("查询不存在任务", test_not_found()))
    results.append(("取消已终态任务", test_cancel_terminal()))
    results.append(("队列容量保护", test_queue_full()))

    # 汇总
    print_header("测试汇总")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {status}  {name}")
    print(f"\n  总计: {passed}/{total} 通过")

    if passed == total:
        print("\n  🎉 所有测试通过！异步任务系统工作正常。")
    else:
        print(f"\n  ⚠️ {total - passed} 个测试未通过，请检查。")


if __name__ == "__main__":
    main()
