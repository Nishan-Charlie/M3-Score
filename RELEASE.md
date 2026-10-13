# Publication release

This repository is a cleaned, pre-publication companion release for the M3-Score research project.

## Included

- Core metric, evaluation, model, and data-loading code.
- Reproducibility scripts and configuration.
- Publication figures and result reports.
- Experimental logs retained for auditability.
- Supporting publication figures and retained logs in `paper_release/`; the ArXiv link will be added later.

## Deliberately excluded

Raw medical datasets, model checkpoints, generated training outputs, temporary image
batches, exploratory notebooks, superseded manuscript drafts, LaTeX source files, and
LaTeX build artifacts are not part of the release. They should be obtained or generated
locally using the documented scripts and external dataset instructions.

The repository does not contain patient data. Do not add datasets, checkpoints, or other
private artifacts before publishing.

## Python package

The `m3-score` distribution contains only `src/m3score/`. Experiment code, figures and
result reports stay in the repository and are not part of the wheel.

### One-time setup

1. Create an account on https://pypi.org and enable two-factor authentication.
2. Under "Publishing", add a pending trusted publisher with:
   - PyPI project name: `m3-score`
   - Owner: `Nishan-Charlie`
   - Repository: `M3-Score`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repository, create an environment named `pypi` (Settings -> Environments).

### Each release

1. Set `version` in `pyproject.toml` and `CITATION.cff`, and merge to `main` with CI green.
2. Locally: `python -m build` and `twine check dist/*`.
3. On GitHub, publish a release whose tag is `v` followed by the version, for example
   `v0.1.0`. The "Publish to PyPI" workflow builds and uploads it. The workflow refuses
   to run when the tag and the version differ.

A version number can be published only once, and a project name cannot be released
by anyone else afterwards. Check the built wheel before publishing the release.

### Changing the protocol

Any change to a constant in `src/m3score/protocol.py`, or to the arithmetic in
`src/m3score/stats.py`, changes what an M3-Score means. Give it a new
`PROTOCOL_VERSION`, regenerate the golden values with `python tools/make_golden.py`,
and say so in the release notes.
