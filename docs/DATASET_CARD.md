# Dataset Card: AxiomBridge-SeverityLab

## Dataset Description

**Dataset Name:** AxiomBridge-SeverityLab Video Severity Dataset  
**Version:** 0.1.0  
**Last Updated:** December 2025  
**Contact:** [Your contact information]

### Summary

This dataset contains video sequences labeled with severity classifications for autonomous risk understanding applications. The dataset is designed to train and evaluate machine learning models that can assess the severity level of situations captured in video form.

## Dataset Structure

### Labels

The dataset uses a 3-class severity taxonomy:

- **Stable (0):** Situations with minimal or no risk indicators. Normal operational conditions with no immediate concerns.
- **Critical (1):** Situations requiring attention or intervention. Elevated risk indicators that warrant monitoring or response.
- **Terminal (2):** Situations requiring immediate action. High-risk scenarios with potential for significant adverse outcomes.

### Data Splits

The dataset is divided into three stratified splits to ensure balanced representation across classes:

- **Training Set:** 80% of data (used for model training)
- **Validation Set:** 10% of data (used for hyperparameter tuning and early stopping)
- **Test Set:** 10% of data (used for final model evaluation, never seen during training)

Stratification ensures each split maintains the same class distribution as the overall dataset.

### File Organization

```
data/
├── raw/                    # Original video files
│   ├── stable/
│   ├── critical/
│   └── terminal/
├── interim/                # Extracted frames and preprocessing artifacts
│   └── frames/
│       ├── stable/
│       ├── critical/
│       └── terminal/
└── processed/              # Final train/val/test splits
    ├── train/
    │   ├── stable/
    │   ├── critical/
    │   └── terminal/
    ├── val/
    │   ├── stable/
    │   ├── critical/
    │   └── terminal/
    └── test/
        ├── stable/
        ├── critical/
        └── terminal/
```

## Data Collection

### Collection Protocol

**[IMPORTANT: Fill in your specific collection methodology]**

Example template:
- **Collection Period:** [Start date] to [End date]
- **Collection Method:** [Describe how videos were sourced]
- **Geographic Coverage:** [Regions/locations covered]
- **Temporal Coverage:** [Time periods, seasons, conditions]
- **Capture Devices:** [Camera types, resolutions, frame rates]
- **Collection Team:** [Team composition, training]

### Inclusion Criteria

Videos were included if they met the following criteria:
- Minimum duration: [X] seconds
- Minimum resolution: [X]p
- Minimum frame rate: [X] fps
- Clear visibility of relevant risk indicators
- Sufficient context for severity assessment

### Exclusion Criteria

Videos were excluded if:
- Quality was too poor for analysis
- Contained excessive noise or artifacts
- Lacked sufficient context for labeling
- Contained personally identifiable information (PII) that could not be anonymized

## Annotation Process

### Labeling Guidelines

**[IMPORTANT: Document your specific labeling criteria]**

Each video was annotated by trained annotators following a detailed rubric:

**Stable:**
- [Specific criteria for "stable" classification]
- [Observable indicators]
- [Example scenarios]

**Critical:**
- [Specific criteria for "critical" classification]
- [Observable indicators]
- [Example scenarios]

**Terminal:**
- [Specific criteria for "terminal" classification]
- [Observable indicators]
- [Example scenarios]

### Annotator Training

- Annotators underwent [X] hours of training
- Training included [describe training materials, sessions]
- Annotators achieved [X]% inter-annotator agreement on calibration set

### Quality Control

- Each video was labeled by [N] independent annotators
- Disagreements were resolved through [consensus meeting / expert adjudication / majority vote]
- Final labels reflect [consensus / adjudicated / averaged] decisions
- Inter-annotator agreement (Cohen's Kappa): [X.XX]

### Annotation Tools

- Primary tool: [Tool name and version]
- Annotation interface: [Description]
- Metadata captured: [List fields]

## Dataset Statistics

### Size

- **Total Videos:** [N]
- **Total Duration:** [X] hours
- **Average Video Length:** [X] seconds (SD: [X])
- **Total Frames:** [N] (after extraction at [Y] fps)

### Class Distribution

| Class    | Count | Percentage |
| -------- | ----- | ---------- |
| Stable   | [N]   | [X.X]%     |
| Critical | [N]   | [X.X]%     |
| Terminal | [N]   | [X.X]%     |

### Technical Specifications

- **Resolution Range:** [min] to [max]
- **Frame Rate Range:** [min] to [max] fps
- **Video Formats:** [list formats]
- **Total Size (raw):** [X] GB
- **Total Size (processed):** [X] GB

## Ethical Considerations

### Consent and Privacy

**[CRITICAL: Document consent and privacy measures]**

- **Consent:** [Describe consent process for human subjects]
- **Anonymization:** All personally identifiable information (PII) has been removed or obscured:
  - Faces: [blurred / removed / not present]
  - License plates: [blurred / removed / not present]
  - Addresses: [redacted / not visible]
  - Other identifiers: [list measures]
- **Privacy Review:** Dataset underwent review by [ethics board / privacy officer]

### Sensitive Content

This dataset may contain:
- [List any sensitive content categories present]
- [Describe content warnings]
- Users should review [ETHICS.md](ETHICS.md) before use

### Usage Restrictions

- This dataset is intended for research and development purposes only
- Commercial use: [permitted / requires permission / prohibited]
- Redistribution: [permitted with attribution / prohibited]
- Derivative works: [permitted / requires permission]

See [LICENSE](../LICENSE) for full terms.

## Limitations and Biases

### Known Limitations

1. **Geographic Bias:** [Describe any geographic concentration]
2. **Temporal Bias:** [Describe any temporal patterns]
3. **Demographic Bias:** [Describe any demographic skew]
4. **Scenario Coverage:** [Describe what scenarios are underrepresented]
5. **Technical Limitations:** [Image quality, lighting conditions, etc.]

### Potential Biases

- **Selection Bias:** [How data was selected may introduce bias]
- **Annotation Bias:** [Annotator backgrounds may influence labels]
- **Representation Bias:** [Some populations/scenarios may be underrepresented]

### Recommended Mitigations

- Use with awareness of geographic and temporal limitations
- Supplement with additional data sources when possible
- Validate model performance on diverse test sets
- Implement fairness metrics during evaluation

## Updates and Versioning

### Version History

- **v0.1.0 (December 2025):** Initial release

### Planned Updates

- [Describe any planned additions or changes]

### How to Report Issues

Please report data quality issues, labeling errors, or other concerns by:
- Opening an issue on [GitHub repository]
- Emailing [contact email]
- Include video ID, issue description, and any relevant screenshots

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{axiombridge_severitylab_2025,
  title={AxiomBridge-SeverityLab Video Severity Dataset},
  author={[Your Name/Team]},
  year={2025},
  version={0.1.0},
  url={https://github.com/yourusername/axiombridge-severitylab}
}
```

## Acknowledgments

- [Acknowledge data contributors]
- [Acknowledge annotators]
- [Acknowledge funding sources]
- [Acknowledge any source datasets or inspirations]

## Related Resources

- Project Repository: [GitHub link]
- Model Card: [MODEL_CARD.md](MODEL_CARD.md)
- Ethics Documentation: [ETHICS.md](ETHICS.md)
- Technical Documentation: [README.md](../README.md)

---

**Last Reviewed:** December 2025  
**Next Review:** [Date]
