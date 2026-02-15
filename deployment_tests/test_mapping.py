#!/usr/bin/env python3
"""
单元测试：验证 ICAO 代码映射表功能
"""

from icao_to_fullname_mapping import get_fullname, ICAO_TO_FULLNAME

def test_mapping():
    """测试 ICAO 代码映射"""
    print("=" * 60)
    print("测试 ICAO 代码映射表")
    print("=" * 60)
    
    # 测试常见映射
    test_cases = [
        ("A332", "A330-200"),
        ("A333", "A330-300"),
        ("B77W", "777-300ER"),
        ("B788", "787-8"),
        ("A320", "A320"),
        ("A321", "A321"),
        ("B738", "737-800"),
    ]
    
    passed = 0
    failed = 0
    
    for icao, expected in test_cases:
        result = get_fullname(icao)
        if result == expected:
            print(f"✓ {icao} -> {result}")
            passed += 1
        else:
            print(f"✗ {icao} -> {result} (expected: {expected})")
            failed += 1
    
    print()
    print(f"总计: {len(test_cases)} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print()
    
    # 显示映射表大小
    print(f"映射表包含 {len(ICAO_TO_FULLNAME)} 个 ICAO 代码")
    
    # 显示前 20 个映射
    print()
    print("前 20 个映射:")
    for i, (icao, fullname) in enumerate(list(ICAO_TO_FULLNAME.items())[:20], 1):
        print(f"  {i}. {icao:8} -> {fullname}")
    
    print()
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = test_mapping()
    if success:
        print("✅ 所有测试通过")
        exit(0)
    else:
        print("❌ 部分测试失败")
        exit(1)
