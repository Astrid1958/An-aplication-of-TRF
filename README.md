# An-aplication-of-TRF
EEG signals were paired with binary stimuli, aligned to the same sample length, and split into train (64%), validation (16%), and test (20%) sets. Ridge regularization (λ) was selected by maximizing Pearson correlation on validation data. A TRF model was then trained for each subject, condition, and channel and evaluated on test data.
