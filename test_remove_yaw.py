from holomotion.src.utils.isaac_utils.rotations import *


def test_remove_yaw_component():
    """Test cases to verify the correctness of remove_yaw_component function."""
    import math

    print("Testing remove_yaw_component function...")

    def rad(degrees):
        return math.radians(degrees)

    def approx_equal(a, b, tolerance=1e-3):
        return torch.allclose(a, b, atol=tolerance)

    def quat_to_euler_degrees(q):
        """Helper to convert quaternion to euler angles in degrees for easy reading."""
        euler_tensor = get_euler_xyz_in_tensor(
            q
        )  # Returns single tensor with shape (..., 3)
        roll, pitch, yaw = (
            euler_tensor[..., 0],
            euler_tensor[..., 1],
            euler_tensor[..., 2],
        )
        return torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw)

    # Test 1: Identity case - same quaternions should result in roll/pitch only
    print("\nTest 1: Identity case")
    quat_init = quat_from_euler_xyz(
        torch.tensor(rad(10)), torch.tensor(rad(20)), torch.tensor(rad(45))
    )
    quat_raw = quat_init.clone()
    result = remove_yaw_component(quat_raw, quat_init, w_last=True)

    # Result should be roll=10°, pitch=20°, yaw=0° (not identity!)
    expected = quat_from_euler_xyz(
        torch.tensor(rad(10)), torch.tensor(rad(20)), torch.tensor(rad(0))
    )
    print(f"Result: {result}")
    print(f"Expected (roll=10°, pitch=20°, yaw=0°): {expected}")
    print(f"✓ Passed: {approx_equal(result, expected)}")

    # Test 2: Pure yaw difference - should result in 30° yaw difference
    print("\nTest 2: Pure yaw difference")
    quat_init = quat_from_euler_xyz(
        torch.tensor(rad(0)), torch.tensor(rad(0)), torch.tensor(rad(30))
    )
    quat_raw = quat_from_euler_xyz(
        torch.tensor(rad(0)), torch.tensor(rad(0)), torch.tensor(rad(60))
    )
    result = remove_yaw_component(quat_raw, quat_init, w_last=True)

    # Since raw has 60° yaw and init has 30° yaw, result should be 30° yaw
    expected_30deg_yaw = quat_from_euler_xyz(
        torch.tensor(rad(0)), torch.tensor(rad(0)), torch.tensor(rad(30))
    )

    result_euler = quat_to_euler_degrees(result.unsqueeze(0))
    expected_euler = quat_to_euler_degrees(expected_30deg_yaw.unsqueeze(0))

    print(f"Init: roll=0°, pitch=0°, yaw=30°")
    print(f"Raw: roll=0°, pitch=0°, yaw=60°")
    print(
        f"Result euler: roll={result_euler[0].item():.1f}°, pitch={result_euler[1].item():.1f}°, yaw={result_euler[2].item():.1f}°"
    )
    print(
        f"Expected euler: roll={expected_euler[0].item():.1f}°, pitch={expected_euler[1].item():.1f}°, yaw={expected_euler[2].item():.1f}°"
    )
    print(f"✓ Passed: {approx_equal(result, expected_30deg_yaw)}")

    # Test 3: Roll/pitch preservation with yaw removal
    print("\nTest 3: Roll/pitch preservation")
    quat_init = quat_from_euler_xyz(
        torch.tensor(rad(0)), torch.tensor(rad(0)), torch.tensor(rad(45))
    )
    quat_raw = quat_from_euler_xyz(
        torch.tensor(rad(15)), torch.tensor(rad(25)), torch.tensor(rad(90))
    )
    result = remove_yaw_component(quat_raw, quat_init, w_last=True)

    # Expected: roll=15°, pitch=25°, yaw≈45° (90°-45°=45°)
    expected = quat_from_euler_xyz(
        torch.tensor(rad(15)), torch.tensor(rad(25)), torch.tensor(rad(45))
    )

    result_euler = quat_to_euler_degrees(result.unsqueeze(0))
    expected_euler = quat_to_euler_degrees(expected.unsqueeze(0))

    print(f"Init: roll=0°, pitch=0°, yaw=45°")
    print(f"Raw: roll=15°, pitch=25°, yaw=90°")
    print(
        f"Result euler: roll={result_euler[0].item():.1f}°, pitch={result_euler[1].item():.1f}°, yaw={result_euler[2].item():.1f}°"
    )
    print(
        f"Expected euler: roll={expected_euler[0].item():.1f}°, pitch={expected_euler[1].item():.1f}°, yaw={expected_euler[2].item():.1f}°"
    )
    print(f"✓ Passed: {approx_equal(result, expected)}")

    # Test 4: Zero yaw offset - should preserve original roll/pitch, remove yaw
    print("\nTest 4: Zero yaw offset removal")
    # Robot starts with 0° yaw offset, current reading has 30° yaw + 10° roll
    quat_init = quat_from_euler_xyz(
        torch.tensor(rad(0)), torch.tensor(rad(0)), torch.tensor(rad(0))
    )
    quat_raw = quat_from_euler_xyz(
        torch.tensor(rad(10)), torch.tensor(rad(0)), torch.tensor(rad(30))
    )
    result = remove_yaw_component(quat_raw, quat_init, w_last=True)

    # Should result in roll=10°, pitch=0°, yaw=30° (since no offset to remove)
    expected_simple = quat_from_euler_xyz(
        torch.tensor(rad(10)), torch.tensor(rad(0)), torch.tensor(rad(30))
    )

    result_euler_simple = quat_to_euler_degrees(result.unsqueeze(0))
    expected_euler_simple = quat_to_euler_degrees(expected_simple.unsqueeze(0))

    print(f"Init: roll=0°, pitch=0°, yaw=0°")
    print(f"Raw: roll=10°, pitch=0°, yaw=30°")
    print(
        f"Result euler: roll={result_euler_simple[0].item():.1f}°, pitch={result_euler_simple[1].item():.1f}°, yaw={result_euler_simple[2].item():.1f}°"
    )
    print(
        f"Expected euler: roll={expected_euler_simple[0].item():.1f}°, pitch={expected_euler_simple[1].item():.1f}°, yaw={expected_euler_simple[2].item():.1f}°"
    )
    print(f"✓ Passed: {approx_equal(result, expected_simple)}")

    # Test 5: Normalization check
    print("\nTest 5: Quaternion normalization")
    quat_init_norm = quat_from_euler_xyz(
        torch.tensor(rad(30)), torch.tensor(rad(45)), torch.tensor(rad(60))
    )
    quat_raw_norm = quat_from_euler_xyz(
        torch.tensor(rad(10)), torch.tensor(rad(20)), torch.tensor(rad(90))
    )
    result_norm = remove_yaw_component(
        quat_raw_norm, quat_init_norm, w_last=True
    )

    # Check that result is normalized (magnitude = 1)
    magnitude = torch.norm(result_norm)
    print(f"Result quaternion magnitude: {magnitude:.6f}")
    print(f"✓ Passed: {approx_equal(magnitude, torch.tensor(1.0))}")

    print("\n" + "=" * 50)
    print("All tests completed!")
    print(
        "If all tests show ✓ Passed: True, the function is working correctly."
    )


if __name__ == "__main__":
    test_remove_yaw_component()
