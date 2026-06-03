# Decision log

- Use a static cache because repo generation has fixed `max_len`.
- Cross K/V is original-batch indexed to avoid duplicating memory across beams.
- Self K/V and ARM sums are active-hypothesis indexed because they depend on generated prefix.
- ARM sums are fp32 to avoid drift and must be updated after computing current attention.
- Keep full-prefix code as the oracle and fallback for regression tests.
- Do not use PyTorch 2.x attention APIs because repo environment is PyTorch 1.8.1 and custom ARM needs weights.
