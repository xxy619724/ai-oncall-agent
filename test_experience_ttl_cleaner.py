"""ExperienceTtlCleaner 单元测试

覆盖场景:
1. 过期经验被标记 deprecated
2. 未过期经验保持 active
3. 已 deprecated 经验不再重复处理
4. 自定义 TTL 生效（每条经验独立 ttl_days）
5. 分批处理（batch_size）
6. get_stats 统计准确
7. 异常降级（表不存在不崩溃）
8. start/stop 生命周期
"""

import asyncio
import os
import sys
import aiosqlite
from datetime import datetime, timedelta

# Windows 控制台 UTF-8 输出（避免 GBK 编码崩溃）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.experience_ttl_cleaner import ExperienceTtlCleaner


async def setup_test_table(conn: aiosqlite.Connection):
    """初始化测试表结构（对齐 memory_writer.init_experiences_table）"""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aiops_experiences (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            final_solution TEXT,
            steps_json TEXT,
            errors_json TEXT,
            task_type TEXT,
            has_error INTEGER DEFAULT 0,
            created_at TEXT,
            confidence TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'active',
            ttl_days INTEGER DEFAULT 90,
            version INTEGER DEFAULT 1
        )
        """
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aiops_experiences_status "
        "ON aiops_experiences(status)"
    )
    await conn.commit()


async def insert_experience(
    conn: aiosqlite.Connection,
    exp_id: str,
    days_ago: int,
    ttl_days: int,
    status: str = "active",
):
    """插入测试经验记录

    Args:
        days_ago: 创建时间距今天几天前
        ttl_days: TTL 天数
        status: 初始状态
    """
    created_at = (datetime.now() - timedelta(days=days_ago)).isoformat()
    await conn.execute(
        """
        INSERT INTO aiops_experiences
            (id, task, created_at, status, ttl_days, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (exp_id, f"test task {exp_id}", created_at, status, ttl_days, "medium"),
    )
    await conn.commit()


async def get_status(conn: aiosqlite.Connection, exp_id: str) -> str:
    """查询经验当前状态"""
    cursor = await conn.execute(
        "SELECT status FROM aiops_experiences WHERE id = ?", (exp_id,)
    )
    row = await cursor.fetchone()
    return row[0] if row else None


async def test_expired_marked_deprecated():
    """场景 1: 过期经验被标记 deprecated"""
    print("\n=== 场景 1: 过期经验被标记 deprecated ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 100 天前创建，TTL=90 天 → 过期
        await insert_experience(conn, "exp_1", days_ago=100, ttl_days=90)

        cleaner = ExperienceTtlCleaner(conn)
        marked = await cleaner.cleanup_expired()

        status = await get_status(conn, "exp_1")
        assert marked == 1, f"期望标记 1 条，实际 {marked}"
        assert status == "deprecated", f"期望 deprecated，实际 {status}"
        print(f"  ✓ 100天前+TTL90=过期，已标记 deprecated (marked={marked})")


async def test_not_expired_keeps_active():
    """场景 2: 未过期经验保持 active"""
    print("\n=== 场景 2: 未过期经验保持 active ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 30 天前创建，TTL=90 天 → 未过期
        await insert_experience(conn, "exp_2", days_ago=30, ttl_days=90)

        cleaner = ExperienceTtlCleaner(conn)
        marked = await cleaner.cleanup_expired()

        status = await get_status(conn, "exp_2")
        assert marked == 0, f"期望标记 0 条，实际 {marked}"
        assert status == "active", f"期望 active，实际 {status}"
        print(f"  ✓ 30天前+TTL90=未过期，保持 active (marked={marked})")


async def test_deprecated_not_reprocessed():
    """场景 3: 已 deprecated 经验不再重复处理"""
    print("\n=== 场景 3: 已 deprecated 经验不重复处理 ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 已 deprecated 的过期经验，不应被重复处理
        await insert_experience(
            conn, "exp_3", days_ago=200, ttl_days=90, status="deprecated"
        )

        cleaner = ExperienceTtlCleaner(conn)
        marked = await cleaner.cleanup_expired()

        assert marked == 0, f"期望标记 0 条，实际 {marked}"
        print(f"  ✓ 已 deprecated 的过期经验不被重复处理 (marked={marked})")


async def test_custom_ttl_per_experience():
    """场景 4: 每条经验独立 TTL"""
    print("\n=== 场景 4: 每条经验独立 TTL ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 经验 A: 50 天前 + TTL=30 → 过期
        # 经验 B: 50 天前 + TTL=90 → 未过期
        await insert_experience(conn, "exp_a", days_ago=50, ttl_days=30)
        await insert_experience(conn, "exp_b", days_ago=50, ttl_days=90)

        cleaner = ExperienceTtlCleaner(conn)
        marked = await cleaner.cleanup_expired()

        status_a = await get_status(conn, "exp_a")
        status_b = await get_status(conn, "exp_b")
        assert marked == 1, f"期望标记 1 条，实际 {marked}"
        assert status_a == "deprecated", f"exp_a 应过期，实际 {status_a}"
        assert status_b == "active", f"exp_b 应未过期，实际 {status_b}"
        print(f"  ✓ 同样 50 天前，TTL30 过期/TTL90 未过期 (marked={marked})")


async def test_batch_size_limit():
    """场景 5: 分批处理限制"""
    print("\n=== 场景 5: 分批处理限制 ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 插入 5 条过期经验
        for i in range(5):
            await insert_experience(conn, f"exp_{i}", days_ago=200, ttl_days=90)

        # batch_size=3，只处理 3 条
        cleaner = ExperienceTtlCleaner(conn, batch_size=3)
        marked = await cleaner.cleanup_expired()

        assert marked == 3, f"期望标记 3 条，实际 {marked}"
        print(f"  ✓ 5 条过期+batch=3，只处理 3 条 (marked={marked})")


async def test_get_stats():
    """场景 6: get_stats 统计准确"""
    print("\n=== 场景 6: get_stats 统计准确 ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        # 2 条 active 未过期 + 1 条 active 已过期 + 1 条 deprecated
        await insert_experience(conn, "active_1", days_ago=10, ttl_days=90)
        await insert_experience(conn, "active_2", days_ago=10, ttl_days=90)
        await insert_experience(conn, "expired_1", days_ago=200, ttl_days=90)
        await insert_experience(
            conn, "dep_1", days_ago=200, ttl_days=90, status="deprecated"
        )

        cleaner = ExperienceTtlCleaner(conn)
        stats = await cleaner.get_stats()

        assert stats["active_count"] == 3, f"active 数量错误: {stats}"
        assert stats["expired_pending_deprecate"] == 1, f"待过期标记数错误: {stats}"
        assert stats["deprecated_count"] == 1, f"deprecated 数量错误: {stats}"
        print(
            f"  ✓ 统计准确: active={stats['active_count']}, "
            f"pending={stats['expired_pending_deprecate']}, "
            f"deprecated={stats['deprecated_count']}"
        )


async def test_table_not_exist_no_crash():
    """场景 7: 表不存在不崩溃（降级安全）"""
    print("\n=== 场景 7: 表不存在不崩溃 ===")
    async with aiosqlite.connect(":memory:") as conn:
        # 不建表，直接跑清理
        cleaner = ExperienceTtlCleaner(conn)
        try:
            marked = await cleaner.cleanup_expired()
            print(f"  ✗ 应该抛异常但没抛 (marked={marked})")
            assert False
        except Exception as e:
            print(f"  ✓ 表不存在抛异常但被调用方捕获: {type(e).__name__}")

        # get_stats 也应抛异常，不崩溃调用方
        try:
            stats = await cleaner.get_stats()
            print(f"  ✗ 应该抛异常但没抛 (stats={stats})")
            assert False
        except Exception as e:
            print(f"  ✓ get_stats 异常被捕获: {type(e).__name__}")


async def test_start_stop_lifecycle():
    """场景 8: start/stop 生命周期"""
    print("\n=== 场景 8: start/stop 生命周期 ===")
    async with aiosqlite.connect(":memory:") as conn:
        await setup_test_table(conn)
        await insert_experience(conn, "exp_life", days_ago=200, ttl_days=90)

        cleaner = ExperienceTtlCleaner(
            conn, cleanup_interval_hours=1  # 1 小时间隔（测试用）
        )
        # start 会立即跑一次清理
        await cleaner.start_periodic_cleanup()
        await asyncio.sleep(0.2)  # 等首次清理完成

        status = await get_status(conn, "exp_life")
        assert status == "deprecated", f"启动后应已清理，实际 {status}"

        # stop 应正常停止
        await cleaner.stop()
        print(f"  ✓ start 立即清理 + stop 正常停止 (status={status})")


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("ExperienceTtlCleaner 单元测试")
    print("=" * 60)

    tests = [
        test_expired_marked_deprecated,
        test_not_expired_keeps_active,
        test_deprecated_not_reprocessed,
        test_custom_ttl_per_experience,
        test_batch_size_limit,
        test_get_stats,
        test_table_not_exist_no_crash,
        test_start_stop_lifecycle,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 异常: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过 / {failed} 失败 / 共 {len(tests)} 个场景")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
