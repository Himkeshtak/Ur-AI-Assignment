# Test data

`simple_invoice.json` contains synthetic Japanese invoice values authored for
this repository. Tests generate a small vector PDF locally, render it with
PDFium, and inject known structure/OCR/reviewer outputs where stated.

These fixtures test orchestration and validation. They are not an OCR benchmark
or evidence of Japanese invoice recognition accuracy. No customer invoice is
included in the repository.
