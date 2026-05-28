# Release Management
This document describes the management approach by which official versions of the TeleCom-Bench benchmark dataset and evaluation code are released.
## Versioning
The TeleCom-Bench project follows the specification of Semantic Versioning 2.0.0 with the following supplemental notes:
1. **Dataset Subset**: To prevent evaluation dataset leakage and ensure benchmark integrity, this repository publicly releases only examples of a subset of the benchmark dataset. For the usage or access method of the complete dataset, please refer to the relevant documentation.
2. **Evaluation Code**: All evaluation code will be released with the version. APIs or evaluation scripts marked as EXPERIMENTAL or DEPRECATED may change at any time, though we aim for stability.
3. **Contribution to Stability**: If you wish for a specific evaluation task or data loader to be more stable, please contribute relevant unit tests.
## Release Criteria
A new version release must meet the following criteria:
1. **Code Quality**: The evaluation code shall not have any critical errors that cause failure of key evaluation workflows.
2. **Test Coverage**: Changes to all evaluation tasks and core scripts must be tested, ensuring that the test code coverage of the core evaluation workflow does not decrease.
3. **Documentation Review**: All documentation (especially `README.md`) must be reviewed to ensure there is no misleading description, and all critical links in the documentation (such as links to dataset subsets or code directories) are valid.
4. **Benchmark Reproducibility**: Core evaluation results (for example, the performance of representative models on the public data subset) should be stably reproducible using the evaluation code released in this repository.
## Release Process
TeleCom-Bench plans to release 2 major versions per year. The specific cadence will be dynamically adjusted based on dataset updates, evaluation task expansion, or community feedback.
Releases are managed by the following personnel (based on repository contribution information):
- xiao.jieting@zte.com.cn
