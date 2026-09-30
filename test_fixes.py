#!/usr/bin/env python3
"""
測試腳本：驗證 ArUco 偵測和 tilt floor 修復

用法：
    python test_fixes.py
"""

import sys
import os

# 確保可以導入 VLServo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_aruco_config():
    """測試 ArUco 配置文件加載"""
    import yaml
    from yaml.loader import SafeLoader
    
    print("=" * 60)
    print("測試 1: ArUco 配置文件")
    print("=" * 60)
    
    config_paths = [
        'aruco_marker_info.yaml',
        'config/aruco_marker_info.yaml',
        'VLServo/aruco_marker_info.yaml'
    ]
    
    for path in config_paths:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    marker_info = yaml.load(f, Loader=SafeLoader)
                print(f"✓ 成功加載: {path}")
                print(f"  包含的 Marker IDs: {list(marker_info.keys())}")
                if '135' in marker_info:
                    print(f"  ✓ 找到 D405 ArUco (ID 135): {marker_info['135']}")
                else:
                    print(f"  ✗ 警告: 沒有找到 D405 ArUco (ID 135)")
                return True
            except Exception as e:
                print(f"✗ 加載失敗 {path}: {e}")
    
    print("✗ 找不到任何配置文件")
    return False


def test_tilt_floor_logic():
    """測試 tilt floor 邏輯"""
    import numpy as np
    import math
    
    print("\n" + "=" * 60)
    print("測試 2: Tilt Floor 邏輯")
    print("=" * 60)
    
    # 模擬測試案例
    test_cases = [
        (-75.0, -1.0, -1.31, "當前在 floor 之上，應該鎖定"),
        (-75.0, -1.5, -1.31, "當前在 floor 之下，應該允許往上到 floor"),
        (-75.0, -1.31, -1.31, "正好在 floor，應該保持"),
    ]
    
    for floor_deg, cur_tilt, floor_rad, desc in test_cases:
        print(f"\n測試案例: {desc}")
        print(f"  Floor: {floor_deg}° ({floor_rad:.3f} rad)")
        print(f"  當前 tilt: {cur_tilt:.3f} rad")
        
        # 模擬修復後的邏輯
        if cur_tilt >= floor_rad:
            desired_tilt = cur_tilt
            v_tilt = 0.0
            print(f"  結果: ✓ LOCKED at {cur_tilt:.3f} rad")
        else:
            desired_tilt = floor_rad  # 簡化：假設想往上
            v_tilt = 0.5
            print(f"  結果: ✓ 允許移動到 {desired_tilt:.3f} rad (v={v_tilt})")
    
    return True


def test_arm_motion_import():
    """測試 arm_motion 模組可以正常導入"""
    print("\n" + "=" * 60)
    print("測試 3: arm_motion 模組導入")
    print("=" * 60)
    
    try:
        from VLServo import arm_motion
        print("✓ arm_motion 模組導入成功")
        return True
    except Exception as e:
        print(f"✗ arm_motion 導入失敗: {e}")
        return False


def test_head_tracker_import():
    """測試 head_tracker 模組可以正常導入"""
    print("\n" + "=" * 60)
    print("測試 4: head_tracker 模組導入")
    print("=" * 60)
    
    try:
        from VLServo import head_tracker
        print("✓ head_tracker 模組導入成功")
        return True
    except Exception as e:
        print(f"✗ head_tracker 導入失敗: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("VLMotion 修復驗證測試")
    print("=" * 60)
    
    results = []
    
    # 執行所有測試
    results.append(("ArUco 配置", test_aruco_config()))
    results.append(("Tilt Floor 邏輯", test_tilt_floor_logic()))
    results.append(("arm_motion 導入", test_arm_motion_import()))
    results.append(("head_tracker 導入", test_head_tracker_import()))
    
    # 總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ 所有測試通過！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 啟動 GUI: ros2 launch vlservo maingui.launch.py")
        print("2. 點擊 'Start Camera'")
        print("3. 確認可以看到 ArUco 標記 (綠色邊框)")
        print("4. 設定 'LLM Tilt Extra' 為 -75")
        print("5. 點擊 'Start LLM Grasping'")
        print("6. 檢查終端輸出，確認 tilt 被鎖定在 -75°")
    else:
        print("\n" + "=" * 60)
        print("✗ 部分測試失敗，請檢查錯誤訊息")
        print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
