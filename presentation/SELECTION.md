# Demo image selection

Chosen from 403 DermaMNIST test images by comparing the plaintext model (A) with the block2-encrypted model (variant D). Predictions below are real, from the saved checkpoints.

| Role | File | True class | Plaintext model | Encrypted-image model |
|---|---|---|---|---|
| safe (both correct) | `safe_melanoma_test846` | melanoma | melanoma (62%) | melanoma (65%) |
| safe (both correct) | `safe_melanocytic_nevi_test30` | melanocytic nevi | melanocytic nevi (34%) | melanocytic nevi (35%) |
| safe (both correct) | `safe_basal_cell_carcinoma_test436` | basal cell carcinoma | basal cell carcinoma (85%) | basal cell carcinoma (83%) |
| honest failure | `failure_basal_cell_carcinoma_test1214` | basal cell carcinoma | basal cell carcinoma (41%) | actinic keratoses and intraepithelial carcinoma (85%) |

Images are 28x28 test images from the public DermaMNIST dataset, enlarged 8x for display. Research prototype; not for clinical use.
