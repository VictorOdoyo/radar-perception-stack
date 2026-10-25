# Architecture

| Layer | Modules | Function |
|---|---|---|
| Physical signals | waveform, channel, frontend | frame geometry, coherent IQ, clutter and impairments |
| Spectral/spatial | processing, array, spectral, advanced | FFTs, angle estimates, ambiguity and optional extensions |
| Perception | detection, geometry, clustering, tracking, pipeline | CFAR, covariance, clustering, association and lifecycle |
| Motion and quality | egomotion, diagnostics, metrics | robust ego fit, sensor checks and independent scoring |
| External data | records, detections_io | strict IQ and polar detection replay |
| Evidence | scenes, studies_frame, studies_tracking, studies_analysis | seeded physical scenes and reproducible studies |
| Artifacts | results, reporting, experiments, batch, cli | provenance, sealed exports, rendering and verification |

Artifact infrastructure is adapted from this author's MIT-licensed ev-powertrain-lab and adaptive-grid-protection repositories. Radar signal, detector and tracking implementations are independently authored here. Main perception does not use the optional MUSIC, ego-motion or ambiguity-resolution benches as hidden truth inputs.
