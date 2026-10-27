# Class-agnostic grouping prototype

This experiment uses base SAM3 to propose generic surgical-instrument instances,
then groups them using CLIP appearance features, mask shape, and a weak spatial
prior. Results are anonymous (`Group A`, `Group B`, ...) rather than medical
instrument names.

```bash
python3 group_count.py /workspace/counting_test.png
```

Outputs are stored under `/workspace/Outputs/class_agnostic_counting/`. It does
not modify the ROS worker, GUI, calibrated thresholds, or deployed weights.
