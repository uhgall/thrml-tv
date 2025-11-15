
Where:

- **Az** = bearing  
- **FV** = normalized field value (1.0 = max gain)

Example start of your row:

| Azimuth | Field Value | Interpretation |
|--------|-------------|----------------|
| 0° | 0.987 | Slight attenuation |
| 10° | 1.000 | Max gain |
| 20° | 0.977 | ~2% reduction |

The final FV in this record corresponds to **350°**.  
Because no additional pattern points are provided, **ExtraFVs** is empty.

---

## Summary

A TVStudy station row encodes:

- All FCC identity and licensing data  
- All facility technical parameters  
- Three types of modeled coverage  
- A 36-point directional antenna gain table  

This row contains **every value required** for interference and coverage computation in FCC repack and TVStudy simulations.

