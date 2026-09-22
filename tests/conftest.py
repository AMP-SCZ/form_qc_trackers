"""Shared package imports for tests against the complete deployed checkout.

Legacy sliced-workspace tests create placeholder packages at collection time
when these modules are missing. Import the real modules first so test order
cannot change a checker class's base. Importing them performs no Utils
construction, configuration reads, or pipeline work; fixtures remain responsible
for every runtime dependency.
"""

import main  # noqa: F401 -- supplies the deployed sibling-package paths.
import qc_forms.form_check  # noqa: F401
import utils.utils  # noqa: F401
