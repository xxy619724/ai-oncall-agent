"""记忆写入门控单元测试

直接测试 memory_writer 的四个门控函数，不依赖完整 AIOps 流程。
运行: .venv\Scripts\python.exe test_memory_gate.py
"""

import asyncio
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_assess_confidence():
    """测试门控4：置信度评估"""
    from app.agent.aiops.memory_writer import _assess_confidence

    print("\n" + "=" * 60)
    print("测试门控4：置信度评估 (_assess_confidence)")
    print("=" * 60)

    # 场景1：纯模型推理（无工具调用）→ medium
    past_steps_1 = []
    response_1 = "根据分析，系统运行正常"
    result_1 = _assess_confidence(past_steps_1, response_1)
    print(f"\n场景1 - 纯模型推理:")
    print(f"  past_steps: {past_steps_1}")
    print(f"  期望: medium, 实际: {result_1}", "✓" if result_1 == "medium" else "✗")

    # 场景2：工具调用成功（非外部、非错误）→ high
    past_steps_2 = [("查询Prometheus告警", "查询完成，当前无活跃告警")]
    result_2 = _assess_confidence(past_steps_2, "")
    print(f"\n场景2 - 工具调用成功:")
    print(f"  past_steps: {past_steps_2}")
    print(f"  期望: high, 实际: {result_2}", "✓" if result_2 == "high" else "✗")

    # 场景3：外部工具返回（含 mcp 关键词）→ low
    past_steps_3 = [("调用MCP工具", "mcp 工具返回: 服务状态正常")]
    result_3 = _assess_confidence(past_steps_3, "")
    print(f"\n场景3 - 外部工具返回:")
    print(f"  past_steps: {past_steps_3}")
    print(f"  期望: low, 实际: {result_3}", "✓" if result_3 == "low" else "✗")

    # 场景4：工具调用失败（含错误关键词）→ medium（非外部，非成功）
    past_steps_4 = [("查询日志", "查询失败: connection error")]
    result_4 = _assess_confidence(past_steps_4, "")
    print(f"\n场景4 - 工具调用失败:")
    print(f"  past_steps: {past_steps_4}")
    print(f"  期望: medium, 实际: {result_4}", "✓" if result_4 == "medium" else "✗")

    # 场景5：第三方接口调用 → low
    past_steps_5 = [("调用第三方接口", "接口调用成功，返回数据")]
    result_5 = _assess_confidence(past_steps_5, "")
    print(f"\n场景5 - 第三方接口调用:")
    print(f"  past_steps: {past_steps_5}")
    print(f"  期望: low, 实际: {result_5}", "✓" if result_5 == "low" else "✗")

    passed = sum([
        result_1 == "medium",
        result_2 == "high",
        result_3 == "low",
        result_4 == "medium",
        result_5 == "low",
    ])
    print(f"\n置信度评估: {passed}/5 通过")
    return passed == 5


async def test_detect_conflict():
    """测试门控3：冲突检测"""
    from app.agent.aiops.memory_writer import _detect_conflict

    print("\n" + "=" * 60)
    print("测试门控3：冲突检测 (_detect_conflict)")
    print("=" * 60)

    # 构造模拟的旧经验 Document
    class MockDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    # 场景1：has_error 不同 → 冲突
    old_doc_1 = MockDoc({"has_error": False, "task_preview": "系统正常"})
    is_conflict_1, reason_1 = await _detect_conflict(
        new_input="诊断告警", new_response="发现异常",
        new_has_error=True, old_doc=old_doc_1
    )
    print(f"\n场景1 - has_error 不同（旧=False, 新=True）:")
    print(f"  期望: 冲突, 实际: {'冲突' if is_conflict_1 else '无冲突'}", "✓" if is_conflict_1 else "✗")
    print(f"  原因: {reason_1}")

    # 场景2：关键词矛盾（旧说"正常"，新说"异常"）→ 冲突
    old_doc_2 = MockDoc({"has_error": False, "task_preview": "系统运行正常"})
    is_conflict_2, reason_2 = await _detect_conflict(
        new_input="诊断告警", new_response="系统出现异常",
        new_has_error=False, old_doc=old_doc_2
    )
    print(f"\n场景2 - 关键词矛盾（旧='正常', 新='异常'）:")
    print(f"  期望: 冲突, 实际: {'冲突' if is_conflict_2 else '无冲突'}", "✓" if is_conflict_2 else "✗")
    print(f"  原因: {reason_2}")

    # 场景3：无冲突（has_error 相同，无矛盾关键词）
    old_doc_3 = MockDoc({"has_error": True, "task_preview": "CPU使用率高"})
    is_conflict_3, reason_3 = await _detect_conflict(
        new_input="诊断CPU告警", new_response="CPU使用率达90%",
        new_has_error=True, old_doc=old_doc_3
    )
    print(f"\n场景3 - 无冲突:")
    print(f"  期望: 无冲突, 实际: {'冲突' if is_conflict_3 else '无冲突'}", "✓" if not is_conflict_3 else "✗")

    # 场景4：成功/失败矛盾
    old_doc_4 = MockDoc({"has_error": False, "task_preview": "操作成功"})
    is_conflict_4, reason_4 = await _detect_conflict(
        new_input="测试操作", new_response="操作失败",
        new_has_error=False, old_doc=old_doc_4
    )
    print(f"\n场景4 - 成功/失败矛盾:")
    print(f"  期望: 冲突, 实际: {'冲突' if is_conflict_4 else '无冲突'}", "✓" if is_conflict_4 else "✗")
    print(f"  原因: {reason_4}")

    passed = sum([is_conflict_1, is_conflict_2, not is_conflict_3, is_conflict_4])
    print(f"\n冲突检测: {passed}/4 通过")
    return passed == 4


async def test_extract_errors():
    """测试错误提取"""
    from app.agent.aiops.memory_writer import _extract_errors

    print("\n" + "=" * 60)
    print("测试辅助函数：错误提取 (_extract_errors)")
    print("=" * 60)

    past_steps = [
        ("查询告警", "查询成功，无活跃告警"),
        ("查询日志", "查询失败: connection refused"),
        ("检查磁盘", "发现异常: 磁盘使用率 95%"),
        ("正常步骤", "检查完成，一切正常"),
    ]

    errors = _extract_errors(past_steps)
    print(f"\n输入 {len(past_steps)} 个步骤:")
    for step, result in past_steps:
        print(f"  - {step}: {result}")
    print(f"\n提取到 {len(errors)} 个错误步骤:")
    for e in errors:
        print(f"  - {e['step']}: {e['error'][:60]}...")

    passed = len(errors) == 2  # 应该提取出"查询失败"和"发现异常"两个
    print(f"\n期望提取 2 个错误, 实际提取 {len(errors)} 个", "✓" if passed else "✗")
    return passed


async def test_check_duplicate():
    """测试门控2：查重（需要 Milvus 连接）"""
    from app.agent.aiops.memory_writer import _check_duplicate
    from app.config import config

    print("\n" + "=" * 60)
    print("测试门控2：查重 (_check_duplicate)")
    print("=" * 60)

    # 用一个普通查询文本测试（预期不重复）
    test_query = "测试查询_这是一条全新的任务描述_用于验证查重功能"
    print(f"\n查询文本: {test_query}")
    print(f"查重阈值: {config.memory_dedup_threshold}")
    print(f"冲突阈值: {config.memory_conflict_threshold}")

    try:
        is_duplicate, similar_docs = await _check_duplicate(
            test_query, config.memory_dedup_threshold
        )
        print(f"\n结果: is_duplicate={is_duplicate}, 相似文档数={len(similar_docs)}")
        print("✓ 查重函数执行成功（Milvus 连接正常）")
        return True
    except Exception as e:
        print(f"✗ 查重函数执行失败: {e}")
        return False


async def test_build_experience_text():
    """测试经验文本构建"""
    from app.agent.aiops.memory_writer import _build_experience_text

    print("\n" + "=" * 60)
    print("测试辅助函数：经验文本构建 (_build_experience_text)")
    print("=" * 60)

    input_text = "诊断系统CPU高使用率告警"
    response = "CPU使用率高的原因是由于数据库查询效率低下导致"
    errors = [{"step": "查询日志", "error": "连接超时"}]
    past_steps = [
        ("查询告警", "发现CPU使用率90%"),
        ("查询日志", "连接超时"),
        ("分析原因", "数据库查询效率低"),
    ]

    experience_text = _build_experience_text(input_text, response, errors, past_steps)
    print(f"\n构建的经验文本:\n{experience_text}")

    # 验证包含各部分
    has_task = "## 任务" in experience_text
    has_solution = "## 最终方案" in experience_text
    has_steps = "## 执行步骤摘要" in experience_text
    has_errors = "## 踩坑记录" in experience_text

    print(f"\n包含任务: {'✓' if has_task else '✗'}")
    print(f"包含方案: {'✓' if has_solution else '✗'}")
    print(f"包含步骤: {'✓' if has_steps else '✗'}")
    print(f"包含踩坑: {'✓' if has_errors else '✗'}")

    passed = all([has_task, has_solution, has_steps, has_errors])
    print(f"\n经验文本构建: {'全部通过 ✓' if passed else '有缺失 ✗'}")
    return passed


async def test_init_table():
    """测试经验表初始化（含 ALTER TABLE 兼容旧表）"""
    import aiosqlite
    from app.config import config
    from app.agent.aiops.memory_writer import init_experiences_table

    print("\n" + "=" * 60)
    print("测试：经验表初始化 (init_experiences_table)")
    print("=" * 60)

    try:
        conn = await aiosqlite.connect(config.sqlite_db_path)
        await init_experiences_table(conn)

        # 验证表结构
        cursor = await conn.execute("PRAGMA table_info(aiops_experiences)")
        columns = await cursor.fetchall()
        await cursor.close()

        print(f"\n表结构（{len(columns)} 列）:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]}) DEFAULT={col[4]}")

        # 验证关键列存在
        col_names = [col[1] for col in columns]
        required_cols = ["confidence", "status", "ttl_days", "version"]
        all_exist = all(c in col_names for c in required_cols)

        print(f"\n门控字段检查:")
        for c in required_cols:
            print(f"  - {c}: {'✓' if c in col_names else '✗'}")

        await conn.close()
        print(f"\n经验表初始化: {'全部通过 ✓' if all_exist else '有缺失 ✗'}")
        return all_exist
    except Exception as e:
        print(f"✗ 表初始化失败: {e}")
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("记忆写入门控 - 单元测试")
    print("=" * 60)

    results = []

    # 1. 辅助函数测试（不依赖外部服务）
    results.append(("错误提取", await test_extract_errors()))
    results.append(("经验文本构建", await test_build_experience_text()))

    # 2. 门控函数测试（不依赖外部服务）
    results.append(("置信度评估", await test_assess_confidence()))
    results.append(("冲突检测", await test_detect_conflict()))

    # 3. 依赖 Milvus 的测试
    results.append(("查重(Milvus)", await test_check_duplicate()))

    # 4. 依赖 SQLite 的测试
    results.append(("经验表初始化", await test_init_table()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
