import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from ai_vital_analysis import AIVitalAnalysis


class ReferenceSynchronizationTests(unittest.TestCase):
    def _write_samples(self, values, start):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            for index, value in enumerate(values):
                stamp = start + timedelta(seconds=index)
                handle.write(f"{stamp.strftime('%d.%m.%Y %H:%M:%S')},{stamp.microsecond // 1000:03d};{value}\n")
            return handle.name
        finally:
            handle.close()

    def test_absolute_timestamps_crop_zero_pulse_and_interpolate(self):
        radar_start = datetime(2024, 1, 1, 0, 0, 0).timestamp()
        ppg_path = self._write_samples(range(7), datetime(2024, 1, 1, 0, 0, 1))
        pulse_path = self._write_samples([0, 0, 70, 72, 74, 76, 78], datetime(2024, 1, 1, 0, 0, 0))
        try:
            analysis = AIVitalAnalysis()
            self.assertTrue(analysis.load_reference_txt_csv(ppg_path, config={"fs": 128}, signal_kind="ppg"))
            ppg = analysis.reference_data
            self.assertTrue(ppg["has_absolute_timestamps"])
            self.assertTrue(analysis.load_reference_txt_csv(pulse_path, config={"fs": 4}, signal_kind="pulse"))
            analysis.pulse_data = analysis.reference_data
            analysis.reference_data = ppg

            result = analysis.synchronize_reference_with_radar(
                radar_start, 6.0, radar_time_s=np.arange(0, 6.01, 1.0)
            )

            self.assertAlmostEqual(result["common_start_s"], 2.0)
            self.assertAlmostEqual(result["common_end_s"], 6.0)
            self.assertEqual(len(analysis.reference_data["signal"]), 5)
            self.assertEqual(analysis.pulse_data["hr_ref"][0], 70)
            np.testing.assert_allclose(analysis.pulse_data["radar_values"], [70, 72, 74, 76, 78])
        finally:
            os.unlink(ppg_path)
            os.unlink(pulse_path)

    def test_manual_offset_is_applied_to_reference_axis(self):
        radar_start = datetime(2024, 1, 1, 0, 0, 0).timestamp()
        ppg_path = self._write_samples(range(6), datetime(2024, 1, 1, 0, 0, 1))
        try:
            analysis = AIVitalAnalysis()
            self.assertTrue(analysis.load_reference_txt_csv(ppg_path, config={"fs": 128}, signal_kind="ppg"))
            result = analysis.synchronize_reference_with_radar(radar_start, 5.0, manual_offset=-1.0)
            self.assertAlmostEqual(result["manual_offset_s"], -1.0)
            self.assertAlmostEqual(result["common_start_s"], 0.0)
            self.assertAlmostEqual(analysis.reference_data["time_sync"][0], 0.0)
        finally:
            os.unlink(ppg_path)


if __name__ == "__main__":
    unittest.main()