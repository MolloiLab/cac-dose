# cac-dose
Study measuring dose reduction capabiltiies of the new calcium scoring technique, volume fraction calcium mass vs. traditional Agatston scoring

## Phantom Set Up
**Overview**
- 3 measurement inserts are included for every scan
- 2 calibration inserts are also included on the endpoints for every scan
  - Useful for automatic segmentation and for calcium mass calibration
- 2 background inserts are in between the measurement and calibration inserts
  - Useful to avoid any partial volume effect from the calibration inserts (e.g. false-positives)

*NOTE: To minimize partial volume effect of erroneous air surrounding the inserts, the cardiac phantom cavity is filled with putty (~ -250 HU). This putty minimizes the partial volume effect of non-anatomical air (~ -900 HU) and also loosely simulates surrounding pericoronary adipose tissue (-30 to -190 HU)*

**Measurement inserts**
- "A": density => 0.050 mg/mm^3, diameter => 1.2 mm
- "B": density => 0.050 mg/mm^3, diameter => 3.0 mm
- "C": density => 0.050 mg/mm^3, diameter => 5.0 mm
- "D": density => 0.100 mg/mm^3, diameter => 1.2 mm
- "E": density => 0.100 mg/mm^3, diameter => 3.0 mm
- "F": density => 0.100 mg/mm^3, diameter => 5.0 mm

**Calibration (endpoint) inserts**
- "N": density => 0.400 mg/mm^3, diameter => 5.0 mm

**Motion**
- Study included all of the above inserts at 3 different beats per minute (bpm):
  - Static: 0 bpm
  - Normal: 60 bpm
  - High: 90 bpm
- This information is included in the DICOM scan name (e.g. "F_0bpm", "D_60bpm", etc.) as it cannot be determined by the DICOM header

**kV & mAs**
- Various tube voltages (80, 100, 120 kV) were studied
- Various tube current exposure times (10, 15, 20, 25, 30, 40, 50, 100, 150, 250 mAs) were studied

## Repository Structure
- `measure_cac.jl`: notebook with rich visualizations for quality control cac measurements of one singular scan
- `measure_cac_script.jl`: notebook wtihout visualizations for complete cac measurements for one set of scans (various mAs)
- `analysis.j`: notebook for analyzing all of the various cac measurements across inserts and mAs
- `data/*`: folder containing the CSV files across the various cac measurements (`measure_cac_script.jl`)